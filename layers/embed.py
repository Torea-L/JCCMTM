import torch

def positional_embedding(pos_seq, inv_freq, bsz=None, batch_first=False):
    """
    Args:
        pos_seq  : [k_len]
        inv_freq : [d_model/2]
    Return:
        pos_emb  : [bsz, k_len, d_model]
    """

    sinusoid_inp = torch.einsum('i,d->id', pos_seq, inv_freq)
    pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
    if batch_first:
        pos_emb = pos_emb[None, :, :]
    else:
        pos_emb = pos_emb[:, None, :]
        
    if (bsz is not None) and (bsz > 1):
        if batch_first:
            return pos_emb.repeat(bsz, 1, 1)
        else:
            return pos_emb.repeat(1, bsz, 1)

    return pos_emb

def relative_positional_encoding(qlen, klen, k_dim, d_model, clamp_len, attn_direction,
                                 bi_data:bool=False, batch_first:bool=False, 
                                 h_number:int=None, bsz:int=None, dtype=None):
    """
    create relative positional encoding.
        
    Return:
        pos_emb : [k_len, bsz, d_model]
        freq_seq : [d_model/2,]
    """
    # 长度为d_model/2
    freq_seq = torch.arange(0, d_model, 2.0)
    if dtype is not None and dtype != torch.float32:
        freq_seq = freq_seq.type(dtype)
    inv_freq = 1 / (10000 ** (freq_seq / d_model))

    if attn_direction == 'bi':
        # beg, end = klen - 1, -qlen
        beg, end = klen, -qlen
    elif attn_direction == 'uni':
        if h_number is not None:
            # beg, end = h_number-1, -1
            beg, end = (klen//k_dim)-1, -1
        else:
            beg, end = klen - 1, -1
            # beg, end = klen, -1
    else:
        raise ValueError('Unknown `attn_type` {}.'.format(attn_direction))

    if bi_data and bsz % 2 == 0:
        fwd_pos_seq = torch.arange(beg, end, -1.0)
        bwd_pos_seq = torch.arange(-beg, -end, 1.0)

        if dtype is not None and dtype != torch.float32:
            fwd_pos_seq = fwd_pos_seq.type(dtype=dtype)
            bwd_pos_seq = bwd_pos_seq.type(dtype=dtype)

        if clamp_len > 0:
            fwd_pos_seq = torch.clamp(fwd_pos_seq, -clamp_len, clamp_len)
            bwd_pos_seq = torch.clamp(bwd_pos_seq, -clamp_len, clamp_len)

        fwd_pos_emb = positional_embedding(fwd_pos_seq, inv_freq, bsz // 2)
        bwd_pos_emb = positional_embedding(bwd_pos_seq, inv_freq, bsz // 2)

        pos_emb = torch.cat([fwd_pos_emb, bwd_pos_emb], dim=1)
    else:
        fwd_pos_seq = torch.arange(beg, end, -1.0)
        if h_number is not None:
            fwd_pos_seq = fwd_pos_seq.repeat(k_dim,1).permute(1,0).reshape(-1)
        if dtype is not None and dtype != torch.float32:
            fwd_pos_seq = fwd_pos_seq.type(dtype=dtype)
        if clamp_len > 0:
            fwd_pos_seq = torch.clamp(fwd_pos_seq, -clamp_len, clamp_len)
        if bsz is not None:
            pos_emb = positional_embedding(fwd_pos_seq, inv_freq, bsz, batch_first=batch_first)
        else:
            pos_emb = positional_embedding(fwd_pos_seq, inv_freq, batch_first=batch_first)

    return pos_emb*0.05
