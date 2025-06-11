__all__ = ['attention_mask']
import torch
from torch import Tensor

"""
Args:
    P : patch_num, number of patches, (qlen_uni)
    M : number of patches in memory cached, (mlen_uni)
    PM : P + M
    qlen = P*N
    mlen = M*N
    klen = qlen + mlen = PM*N
"""

def _create_mask(qlen, mlen, k_dim, dtype, same_length=False):
    """create causal attention mask."""
    # attn_mask = torch.ones([qlen, qlen], dtype=dtype)
    # mask_u = torch.triu(attn_mask)  # Upper triangular part.
    # mask_dia = torch.tril(attn_mask) & torch.triu(attn_mask)  # Diagonal. Figure 2(c)
    # attn_mask_ = mask_u - mask_dia
    attn_mask_ = torch.triu(torch.ones([qlen, qlen], dtype=dtype), diagonal=1)
    attn_mask_pad = torch.zeros([qlen, mlen], dtype=dtype)
    
    for i in range(qlen):
        for j in range(qlen):
            if j<((i)//k_dim+1)*k_dim:
                attn_mask_[i,j] = 0
    ret = torch.cat([attn_mask_pad, attn_mask_], dim=1)  # [qlen, klen]
    
    if same_length:
        attn_mask = torch.ones([qlen, qlen], dtype=dtype)
        mask_dia = torch.tril(attn_mask) & torch.triu(attn_mask)  # Diagonal. Figure 2(c)
        mask_l = torch.tril(attn_mask)  # Lower triangular part.
        ret = torch.cat([ret[:, :qlen] + mask_l - mask_dia, ret[:, qlen:]], dim=1)

    return ret.type(dtype=torch.float32)  # [qlen, klen]

def _generate_data_mask(perm_mask:Tensor, input_mask:Tensor=None, mlen:int=0, batch_size:int=1, attn_module:str='Mul'):
    """
    Args:
        Mul:
            perm_mask : [bsz, qlen, qlen]
        Uni:
            perm_mask : [bsz, n_vars, P, P]
    Return:
        Mul:
            data_mask : [bsz, qlen, klen]
        Uni:
            data_mask : [bsz, n_vars, P, PM]
    """
    assert attn_module in ['Mul', 'Uni'], f"Unrecognized dset (`{attn_module}`). Options include: {['Mul', 'Uni']}"
    ## data mask: input mask & perm mask
    if input_mask is not None and perm_mask is not None: data_mask = input_mask[None] + perm_mask
    elif input_mask is not None and perm_mask is None: data_mask = input_mask[None]
    elif input_mask is None and perm_mask is not None: data_mask = perm_mask
    else: data_mask = None

    if data_mask is not None:
        ## all mems can be attended to
        if attn_module == 'Mul':
            if mlen>0:
                mems_mask = torch.zeros([batch_size, perm_mask.shape[1], mlen], dtype=torch.float32).to(perm_mask.device)
                data_mask = torch.cat([mems_mask, data_mask], dim=-1)  # data_mask : [bsz, q_len, k_len]
        elif attn_module == 'Uni':
            mems_mask = torch.zeros([batch_size, perm_mask.shape[1], perm_mask.shape[2], mlen], dtype=torch.float32).to(perm_mask.device)
            data_mask = torch.cat([mems_mask, data_mask], dim=-1)  # data_mask : [bsz, n_vars, q_len_uni, k_len_uni]
    return data_mask

def attention_mask(batch_size:int, num_vars:int, 
                   qlen_mul:int, qlen_uni:int, mlen_mul:int, mlen_uni:int,
                   input_mask_mul:Tensor, input_mask_uni:Tensor,
                   perm_mask_mul:Tensor, perm_mask_uni:Tensor,
                   attn_mask_mul_cache:Tensor, attn_mask_uni_cache:Tensor,
                   n_cls:int=None, efficient=False, device='cpu',
                   attn_direction='uni', same_length=False):
    
    """
    'attn_mask' is used in Query-stream, where tokens can NOT attend to itself
    'non_tgt_mask' is used in Content-stream, where tokens can attend to itself
    INput:
        perm_mask : [bsz, qlen, qlen]
        perm_mask_uni : [bsz, n_vars, P, P]
    Return:
        attn_mask : [bsz, 1, qlen, klen]
        attn_mask_uni : [bsz, n_vars, 1, P, PM]
        non_tgt_mask : [bsz, 1, qlen, klen]
        non_tgt_mask_uni : [bsz, n_vars, 1, P, PM]
    """
    attn_mask, attn_mask_uni, non_tgt_mask, non_tgt_mask_uni = None, None, None, None
    # attn_mask_mul_cache, attn_mask_uni_cache = attn_mask_mul_cache, attn_mask_uni_cache
    ## causal attention mask
    if attn_direction == 'uni':
        if attn_mask_uni_cache is not None:
            attn_mask_uni = attn_mask_uni_cache
        else:
            attn_mask_uni = _create_mask(qlen_uni, mlen_uni, k_dim=1, dtype=torch.int64, same_length=same_length).to(device)
            attn_mask_uni_cache = attn_mask_uni
        if efficient and (n_cls is not None):
            PM = qlen_uni + mlen_uni
            PM_mul = int((qlen_mul + mlen_mul)/num_vars)
            if PM>=PM_mul:
                attn_mask = attn_mask_uni[:,None,:].repeat(1,n_cls,1).reshape(-1, PM)
                attn_mask = attn_mask[:, -PM_mul:]
                attn_mask = attn_mask[:,:,None].repeat(1,1,n_cls).reshape(-1, PM_mul*n_cls)
            else:
                attn_mask = _create_mask(qlen_uni, int(mlen_mul/num_vars), k_dim=1, dtype=torch.int64, same_length=same_length).to(device)
            non_tgt_mask = -torch.eye(qlen_uni*n_cls, dtype=torch.float32)
            if mlen_uni>0:
                non_tgt_mask = torch.cat([torch.zeros([qlen_uni*n_cls, int(mlen_mul/num_vars)*n_cls], dtype=torch.float32), 
                                          non_tgt_mask], dim=-1).to(device)
            else:
                non_tgt_mask = non_tgt_mask.to(device)
            non_tgt_mask = (attn_mask + non_tgt_mask).gt(0)
    elif attn_direction != 'bi':
        raise ValueError('Unsupported attention type: {}'.format(attn_direction))
    
    if attn_direction == 'uni':
        if attn_mask_mul_cache is not None:
            attn_mask = attn_mask_mul_cache
        else:
            attn_mask = _create_mask(qlen_mul, mlen_mul, k_dim=num_vars, dtype=torch.int64, same_length=same_length).to(device)
            attn_mask_mul_cache = attn_mask
    
    ## data mask: input mask & perm mask
    """
        data_mask : [bsz, qlen, klen]
        data_mask_uni : [bsz, n_vars, P, PM]
    """

    if not efficient:
        data_mask = _generate_data_mask(perm_mask=perm_mask_mul, input_mask=input_mask_mul, mlen=mlen_mul, batch_size=batch_size)
        if data_mask is not None:
            if attn_mask is None:
                attn_mask = data_mask[:, None, :, :]
            else:
                attn_mask = data_mask[:, None, :, :] + attn_mask

    if (not efficient) and (attn_mask is not None):
        non_tgt_mask = -torch.eye(qlen_mul, dtype=torch.float32) 
        non_tgt_mask = torch.cat([torch.zeros([qlen_mul, mlen_mul], dtype=torch.float32), non_tgt_mask], dim=-1).to(device)
        non_tgt_mask = (attn_mask + non_tgt_mask).gt(0) # .type(dtype=torch.float32)

    data_mask_uni = _generate_data_mask(perm_mask=perm_mask_uni, input_mask=input_mask_uni,
                                       mlen=mlen_uni, batch_size=batch_size, attn_module='Uni')
    if data_mask_uni is not None:
        if attn_mask_uni is None:
            attn_mask_uni = data_mask_uni[:, :, None, :, :]
        else:
            attn_mask_uni = data_mask_uni[:, :, None, :, :] + attn_mask_uni

    if attn_mask_uni is not None:
        non_tgt_mask_uni = -torch.eye(qlen_uni, dtype=torch.float32)
        non_tgt_mask_uni = torch.cat([torch.zeros([qlen_uni, mlen_uni], dtype=torch.float32), non_tgt_mask_uni], dim=-1).to(device)
        non_tgt_mask_uni = (attn_mask_uni + non_tgt_mask_uni).gt(0)
        attn_mask_uni = attn_mask_uni.gt(0)
    if attn_mask is not None:
        attn_mask = attn_mask.gt(0)

    return attn_mask, attn_mask_uni, non_tgt_mask, non_tgt_mask_uni, attn_mask_mul_cache, attn_mask_uni_cache
    
