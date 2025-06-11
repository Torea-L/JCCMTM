__all__ = ['MultiheadRelAttention', 'positionwise_ffn']

import torch
from torch import Tensor
import torch.nn as nn
from typing import Optional

import numpy as np
import einops

from layers.basics import get_activation_fn, get_norm_fn

"""
Args:
            B : batch size
            N : n_vars, number of variates in MTS
            G : n_cls, number of group tokens
            NG : n_vars + n_cls
            P : patch_num, number of patches, (qlen_uni)
            M : number of patches cached in memory
            PM : P + M
            D : d_model
            qlen = P*N
            mlen = M*N
            klen = qlen + mlen = PM*N
"""

def head_projection(h:Tensor, name:str, proj_weight:Tensor=None, attn_type='Mul'):
    """
    Project hidden states to a specific head with a 4D-shape.
    Args:
        proj_weight : [D, n_head, d_head]
        Mul:
            stream='h':
                h(efficient) : [B*PM, NG, D]
                h(not-efficient) : [B, qlen, D]
            stream='g':
                h : [B, num_mask, D]
        Uni:
            stream='h':
                h : [B, N, P, D]
            stream='g':
                h : [B, N, num_mask, D]
    """
    if proj_weight is None:
            raise ValueError('`{}_proj_weight` is {}.'.format(name, proj_weight))
    if name not in ['q', 'k', 'v', 'r']:
        raise ValueError('Unknown `name` {}.'.format(name))
    
    if attn_type=='Mul':
        head = torch.einsum('bkh,hnd->bknd', h, proj_weight)
    else:
        head = torch.einsum('bkih,hnd->bkind', h, proj_weight)
    return head

class positionwise_ffn(nn.Module):
    def __init__(self, d_model:int, d_ff:int=128, 
                 dropout=0.1, use_norm=True, bias=True,
                 norm_h='layer_norm', norm_g='layer_norm', norm_shape_h=None, norm_shape_g=None, 
                 activation_type='gelu'):
        super(positionwise_ffn, self).__init__()
        self.activation = activation_type
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation_type),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))
        ## Add & Norm
        self.use_norm = use_norm
        self.dropout_ffn = nn.Dropout(dropout)
        if self.use_norm:
            self.norm_h = get_norm_fn(norm_h, norm_shape=norm_shape_h)
            self.norm_g = get_norm_fn(norm_g, norm_shape=norm_shape_g)
    
    @torch.no_grad()
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if ('relu' in self.activation) or ('gelu' in self.activation):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(m.weight)

    def forward(self, inp:Tensor, stream='h'):
        ## Position-wise Feed-Forward
        output = self.ffn(inp)
        ## Add & Norm
        output = inp + self.dropout_ffn(output)
        if self.use_norm:
            if stream=='h':
                output = self.norm_h(output)
            else:
                output = self.norm_g(output)
        return output
    
class RelAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, attention_dropout=0.1, scale=None, output_attention=False):
        super(RelAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _rel_shift(self, x, batch_first=True):
        """perform relative shift to form the relative attention score."""
        if batch_first:
            # x: [bsz, n_head, qlen, klen]
            zero_pad = torch.zeros((*x.size()[:-2], x.size(-2), 1), device=x.device, dtype=x.dtype)
            x_padded = torch.cat([zero_pad, x], dim=-1)
            x_padded = x_padded.view(*x.size()[:-2], x.size(-1) + 1, x.size(-2))
            x = x_padded[:,:,1:].view_as(x)
        else:
            # x: [qlen, klen, bsz, n_head]
            zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]), device=x.device, dtype=x.dtype)
            x_padded = torch.cat([zero_pad, x], dim=1)
            x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])
            x = x_padded[1:].view_as(x)
        return x
    
    def forward(self, queries, keys, values, attn_mask=None, 
                k_head_r:Optional[Tensor]=None, r_w_bias:Optional[Tensor]=None, r_r_bias:Optional[Tensor]=None, 
                tau=None, delta=None):
        """
        Input:
            queries  : [bsz, P*NG, n_head, d_head]
            keys     : [bsz, PM*NG, n_head, d_head]
            values   : [bsz, PM*NG, n_head, d_head]
            k_head_r : [bsz, PM*NG, n_head, d_head] if not None
            attn_mask: [bsz, 1, P*NG, PM*NG] if not None

            scores   : [bsz, n_head, P*NG, PM*NG]

        Return:
            V        : [bsz, P*NG, n_head, d_head]
        """
        B, I, H, D = queries.shape
        scale = self.scale or 1. / (D ** 0.5)

        if k_head_r is not None:
            #### content based attention score
            # print('queries.shape = ', queries.shape)
            # print('r_w_bias.shape = ', r_w_bias.shape)
            # print('keys.shape = ', keys.shape)
            ac = torch.einsum('bind,bjnd->bnij', queries + r_w_bias, keys)
            
            #### position based attention score
            # print('queries.shape = ', queries.shape)
            # print('r_r_bias.shape = ', r_r_bias.shape)
            # print('k_head_r.shape = ', k_head_r.shape)
            bd = torch.einsum('bind,bjnd->bnij', queries + r_r_bias, k_head_r)
            bd = self._rel_shift(bd)
        else:
            #### content based attention score
            ac = torch.einsum('bind,bjnd->bnij', queries, keys)
            #### position based attention score
            bd = 0
        scores = (ac + bd) * scale
        if self.mask_flag:
            if attn_mask is not None:
                # scores = scores - (1e30 * attn_mask).to(queries.device)
                scores.masked_fill_(attn_mask, -np.inf)
        A = self.dropout(torch.softmax(scores, dim=-1))
        V = torch.einsum("bnij,bjnd->bind", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()
        
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, pos_emb=None, r_w_bias=None, r_r_bias=None, tau=None, delta=None):
        B, I, _ = queries.shape
        _, J, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, I, H, -1)
        keys = self.key_projection(keys).view(B, J, H, -1)
        values = self.value_projection(values).view(B, J, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask=attn_mask,
            k_head_r=pos_emb,
            r_w_bias=r_w_bias,
            r_r_bias=r_r_bias,
            tau=tau,
            delta=delta
        )

        return out, attn

class _twoStream_rel_attn(nn.Module):
    def __init__(self, configs, scale:float, lsa=False, res_attention=True, attn_dropout=0.,
                 eps=1e-7, threshold=0., use_ef=False, use_threshold=False, attn_type='Mul', 
                 vec_norm=True, efficient=False):
        super(_twoStream_rel_attn, self).__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        self.scale = nn.Parameter(torch.tensor(scale), requires_grad=lsa)
        self.lsa = lsa
        self.n_vars = configs.data.n_vars
        self.configs = configs

        self.use_ef = use_ef
        self.use_threshold = use_threshold
        self.attn_type = attn_type
        self.efficient = efficient
        self.cosattn = vec_norm
        self.eps = torch.tensor(eps)
        self.threshold = threshold

        if efficient:
            self.group_proj = nn.Parameter(
            nn.init.xavier_normal_(
                torch.randn(configs.model.d_model, configs.model.n_heads, configs.model.d_head)))
            self.group_attn = AttentionLayer(attention=RelAttention(mask_flag=True, attention_dropout=0.1), 
                                             d_model=configs.model.d_model, n_heads=configs.model.n_heads)
        else:
            self.group_proj, self.group_attn = None, None

    def _rel_shift(self, x, batch_first=True):
        """perform relative shift to form the relative attention score."""
        if batch_first:
            # x: [bsz, n_head, qlen, klen]
            zero_pad = torch.zeros((*x.size()[:-2], x.size(-2), 1), device=x.device, dtype=x.dtype)
            x_padded = torch.cat([zero_pad, x], dim=-1)
            x_padded = x_padded.view(*x.size()[:-2], x.size(-1) + 1, x.size(-2))
            x = x_padded[:,:,1:].view_as(x)
        else:
            # x: [qlen, klen, bsz, n_head]
            zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]), device=x.device, dtype=x.dtype)
            x_padded = torch.cat([zero_pad, x], dim=1)
            x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])
            x = x_padded[1:].view_as(x)
        return x
    
    def forward(self, q_head:Tensor, k_head_h:Tensor, v_head_h:Tensor, k_head_r:Optional[Tensor]=None, group_cls:Tensor=None,
                r_w_bias:Optional[Tensor]=None, r_r_bias:Optional[Tensor]=None, r_s_bias:Optional[Tensor]=None,
                seg_embed:Optional[Tensor]=None, seg_mat:Optional[Tensor]=None, g_stream=False, 
                prev:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None, sparse_attn:Optional[Tensor]=None):
        """
        Input shape:
            Mul:
                q_head(efficient/not-efficient):    [B*PM, NG, n_head, d_head] / [B, qlen, n_head, d_head]
                k_head_h(efficient/not-efficient):  [B*PM, NG, n_head, d_head] / [B, klen, n_head, d_head]
                v_head_h(efficient/not-efficient):  [B*PM, NG, n_head, d_head] / [B, klen, n_head, d_head]
                k_head_r(efficient/not-efficient):  [B, PM*G, n_head, d_head] / [B, klen, n_head, d_head]
                r_w_bias:    [n_head, d_head] vector 'u' in relative positional attention if not 'None'
                r_r_bias:    [n_head, d_head] vector 'v' in relative positional attention if not 'None'
                attn_mask(efficient/not-efficient): [B, 1, p*G, PM*G] / [B, 1, qlen, klen]
                sparse_attn: [1, n_head, qlen, klen] if 'not-efficient' else None
            Uni:
                q_head:      [B, N, P, n_head, d_head]
                k_head_h:    [B, N, klen_uni, n_head, d_head]
                v_head_h:    [B, N, klen_uni, n_head, d_head]
                k_head_r:    [B, klen_uni, n_head, d_head]
                r_w_bias:    [n_head, d_head] vector 'u' in relative positional attention
                r_r_bias:    [n_head, d_head] vector 'v' in relative positional attention
                attn_mask:   [B*N, 1, P, PM]
                sparse_attn: [1, 1, n_head, qlen, klen]
        Returns:
            attn_vec:
                Mul(efficient) : [B*PM, NG, n_head, d_head]
                Mul(not-efficient) : [B, qlen, n_head, d_head]
                Uni : [B, N, P, n_head, d_head]
            attn_score/attn_prob:
                Mul(efficient) : [B*PM, n_head, NG, NG]
                Mul(not-efficient) : [B, n_head, qlen, klen]
                Uni : [B*N, n_head, qlen_uni, klen_uni]
        """
        with torch.no_grad():
            N = self.n_vars
            # B = self.configs.data.batch_size
            B = self.configs.data.bsz_efficient
            if self.attn_type=='Mul' and self.efficient:
                PB, NG, H, D = q_head.shape
                PM = int(PB/B)
                G = NG - N
        if self.cosattn:
            q_head = q_head/torch.max(q_head.norm(p=2, dim=-1, keepdim=True), self.eps)
            k_head_h = k_head_h/torch.max(k_head_h.norm(p=2, dim=-1, keepdim=True), self.eps)
            if k_head_r is not None:
                k_head_r = k_head_r/torch.max(k_head_r.norm(p=2, dim=-1, keepdim=True), self.eps)
                r_w_bias = r_w_bias/torch.max(r_w_bias.norm(p=2, dim=-1, keepdim=True), self.eps)
                r_r_bias = r_r_bias/torch.max(r_r_bias.norm(p=2, dim=-1, keepdim=True), self.eps)
        
        if self.attn_type=='Mul':
            if self.efficient:
                #### content based attention score
                ac = torch.einsum('bind,bjnd->bnij', q_head, k_head_h)
                #### position based attention score
                bd = 0
            elif k_head_r is not None:
                #### content based attention score
                ac = torch.einsum('bind,bjnd->bnij', q_head + r_w_bias, k_head_h)
                #### position based attention score
                bd = torch.einsum('bind,bjnd->bnij', q_head + r_r_bias, k_head_r)
                bd = self._rel_shift(bd)
        else:
            q_head = einops.rearrange(q_head, 'b k i n d -> (b k) i n d')     # [B*N, qlen_uni, n_head, d_head]
            k_head_h = einops.rearrange(k_head_h, 'b k j n d -> (b k) j n d') # [B*N, klen_uni, n_head, d_head]
            v_head_h = einops.rearrange(v_head_h, 'b k j n d -> (b k) j n d') # [B*N, klen_uni, n_head, d_head]

            if k_head_r is not None:
                k_head_r = einops.repeat(k_head_r, 'b j n d -> (b k) j n d', k=N) # [B*N, klen_uni, n_head, d_head]
                #### content based attention score
                ac = torch.einsum('bind,bjnd->bnij', q_head + r_w_bias, k_head_h)
                #### position based attention score
                bd = torch.einsum('bind,bjnd->bnij', q_head + r_r_bias, k_head_r)
                bd = self._rel_shift(bd)
            else:
                #### content based attention score
                ac = torch.einsum('bind,bjnd->bnij', q_head, k_head_h)
                #### position based attention score
                bd = 0
        
        #### merge attention scores and perform masking
        attn_score = (ac + bd) * self.scale
        ## Add pre-softmax attention scores from the previous layer (optional)
        if self.res_attention and (prev is not None): 
            attn_score = attn_score + prev
        if (not self.efficient) or self.attn_type=='Uni':
            if attn_mask is not None:
                if sparse_attn is not None:
                    attn_mask = torch.maximum(attn_mask, sparse_attn)
                attn_score = attn_score - (1e30 * attn_mask.float())
                # attn_score.masked_fill_(attn_mask, -np.inf)
        
        #### attention probability
        attn_prob = self.attn_dropout(torch.softmax(attn_score, dim=-1))

        if self.use_threshold:
            attn_zero = (torch.zeros_like(attn_prob)).to(attn_score.device)
            attn_prob = torch.where(attn_prob>self.threshold, attn_prob ,attn_zero)
        # compute the new values given the attention weights
        # Mul : [bsz*PM, n_cls+n_vars, n_head, d_head]
        # Uni : [bsz*n_vars, klen_uni, n_head, d_head]
        attn_vec = torch.einsum('bnij,bjnd->bind', attn_prob, v_head_h)

        if self.attn_type=='Mul' and self.efficient:
            if not g_stream:
                ## update group tokens
                attn_vec = attn_vec.view(B, PM, NG, H, D)
                group_cls = attn_vec[:,:,-G:]
                group_cls = group_cls.reshape(B, -1, H, D) #[bsz, PM*n_cls, n_head, d_head]
                group_cls = torch.einsum('bind,hnd->bih', group_cls, self.group_proj) #[bsz, PM*n_cls, d_model]
                group_cls, _ = self.group_attn(queries=group_cls, keys=group_cls, values=group_cls, attn_mask=attn_mask,
                                               pos_emb=k_head_r, r_w_bias=r_w_bias, r_r_bias=r_r_bias)
                group_cls = einops.rearrange(group_cls, 'b (p g) n d -> b p g n d', g=G, p=PM)
                attn_vec[:,:,-G:] = group_cls
                attn_vec = attn_vec.contiguous().view(B*PM, -1, H, D)
        elif self.attn_type=='Uni':
            attn_vec = einops.rearrange(attn_vec, '(b k) i n d -> b k i n d', k=N)

        return attn_vec, attn_prob, attn_score

class MultiheadRelAttention(nn.Module):
    def __init__(self, configs, d_k=None, d_v=None, attn_type='Mul', 
                 proj_drop=0., out_drop=0., attn_drop=0., threshold=0.,
                 qkv_bias=True, lsa=False, use_threshold=False,
                 norm_h='layer_norm', norm_g='layer_norm',
                 norm_shape_h=None, norm_shape_g=None,
                 res_attention=True,
                 cos_attn=True, efficient=False):
        super(MultiheadRelAttention, self).__init__()
        self.n_vars = configs.data.n_vars
        self.d_head = configs.model.d_head
        self.n_head = configs.model.n_heads
        self.d_model = configs.model.d_model
        self.configs = configs
        d_k = self.d_head if d_k is None else d_k
        d_v = self.d_head if d_v is None else d_v
        ## scale for dot-product scaling in attention
        scale = 1 / (self.d_head ** 0.5)
        self.attn_type = attn_type
        self.efficient = efficient

        self.q_proj_weight = nn.Parameter(nn.init.xavier_normal_(torch.randn(self.d_model, self.n_head, self.d_head)))
        self.k_proj_weight = nn.Parameter(nn.init.xavier_normal_(torch.randn(self.d_model, self.n_head, d_k)))
        self.v_proj_weight = nn.Parameter(nn.init.xavier_normal_(torch.randn(self.d_model, self.n_head, d_v)))
        self.r_proj_weight = nn.Parameter(nn.init.xavier_normal_(torch.randn(self.d_model, self.n_head, self.d_head)))
        self.proj_out = nn.Parameter(nn.init.xavier_normal_(torch.randn(self.d_model, self.n_head, self.d_head)))

        self.rel_attn = _twoStream_rel_attn(configs=configs, scale=scale, lsa=lsa, res_attention=res_attention, threshold=threshold, use_threshold=use_threshold,
                                            attn_dropout=attn_drop, attn_type=self.attn_type, vec_norm=cos_attn,
                                            efficient=efficient)
        self.proj_drop = nn.Dropout(proj_drop)
        self.out_dropout = nn.Dropout(out_drop)

        self.norm_h = get_norm_fn(norm_h, norm_shape=norm_shape_h)
        self.norm_g = get_norm_fn(norm_g, norm_shape=norm_shape_g)

    def post_attention(self, h, attn_vec, proj_o, stream='h'):
        """
        Post-attention processing.
        
        Input shape:
            stream='h':
                h(Mul-Model)(efficient): [bsz*PM, n_vars + n_cls, d_model]
                h(Mul-Model)(not-efficient): [bsz, qlen, d_model]
                h(Uni-Model): [bsz, n_vars, P, d_model]
                attn_vec(Mul-Model)(efficient): [bsz*PM, n_vars + n_cls, n_head, d_head]
                attn_vec(Mul-Model)(not-efficient): [bsz, qlen, n_head, d_head]
                attn_vec(Uni-Model): [bsz, n_vars, P, n_head, d_head]
            stream='g':
                g(Mul-Model): [bsz, num_mask, d_model]
                g(Uni-Model): [bsz, n_vars, num_mask, d_model]
                attn_vec(Mul-Model): [bsz, num_mask, n_head, d_head]
                attn_vec(Uni-Model): [bsz, n_vars, num_mask, n_head, d_head]
        Return:
            stream='h':
                attn_out(Mul-Model)(efficient): [bsz*PM, n_vars + n_cls, d_model]
                attn_out(Mul-Model)(not-efficient): [bsz, q_len, d_model]
                attn_out(Uni-Model): [bsz, n_vars, P, d_model]
            stream='g':
                attn_out(Mul-Model): [bsz, num_mask, d_model]
                attn_out(Uni-Model): [bsz, n_vars, num_mask, d_model]
        """
        ## post-attention projection (back to `d_model`)
        if self.attn_type=='Mul':
            attn_out = torch.einsum('bind,hnd->bih', attn_vec, proj_o)     # attn_out: [q_len, bsz, d_model]
        else:
            attn_out = torch.einsum('bkind,hnd->bkih', attn_vec, proj_o)   # attn_out: [n_vars, patch_num, bsz, d_model]

        ## Add & Norm
        # Add: residual connection with residual dropout
        output = self.out_dropout(attn_out) + h
        if stream=='h':
            output = self.norm_h(output)
        else:
            output = self.norm_g(output)
        return output
    
    def forward(self, h:Tensor, g:Tensor, r_h:Optional[Tensor]=None, r_g:Optional[Tensor]=None,
                mems:Optional[Tensor]=None, 
                r_w_bias:Optional[Tensor]=None, r_r_bias:Optional[Tensor]=None, r_s_bias:Optional[Tensor]=None, 
                seg_embed:Optional[Tensor]=None, seg_mat:Optional[Tensor]=None, 
                attn_mask_h:Optional[Tensor]=None, attn_mask_g:Optional[Tensor]=None,
                target_mapping:Optional[Tensor]=None, 
                sparse_attn:Optional[Tensor]=None, sparse_attn_mem:Optional[Tensor]=None,
                pretrain=False, prev:Optional[Tensor]=None):
        """
        Input:
            Mul:
                h(efficient) : [B*PM, NG, D]; 
                h(not-efficient) : [B, qlen, D];
                g : [B, num_mask, D]
                r_h(efficient) : [B, PM*G, D]
                r_h(not-efficient) : [B, klen, D]
                r_g : [B, klen, D]
                mems(efficient) : [B*PM, NG, D] if not None
                mems(not-efficient) : [B, mlen, D] if not None
                attn_mask_h(efficient) : [B, P*G, PM*G]
                attn_mask_h(not-efficient) : [B, qlen, klen]
                target_mapping : [B, num_mask, qlen]
                sparse_attn : [1, n_head, qlen, qlen]
                sparse_attn_mem : [1, n_head, qlen, mlen]
            Uni:
                h : [B, N, P, D]
                g : [B, N, num_mask, D]
                r_h : [B, klen_uni, D]
                r_g : [B, klen_uni, D]
                mems : [B, N, M, D]
                target_mapping_uni : [B, N, num_mask, P]
                sparse_attn : [1, n_head, P, P]
                sparse_attn_mem : [1, n_head, P, PM]
        """
        with torch.no_grad():
            B = self.configs.data.batch_size
            N = self.n_vars
            P = self.configs.data.patch_num
            if self.attn_type=='Mul' and self.efficient:
                PB, NG, D = h.shape
                G = NG - N
                PM = int(PB/B)
                M = int(mems.shape[1]) if (mems is not None) else 0

        if mems is not None and len(mems.size()) > 1:
            if self.attn_type=='Mul' and self.efficient:
                cat = torch.cat([mems.view(B, PM, NG, D), h.view(B, PM, NG, D)[:,M:,:,:]], dim=1)
                cat = cat.view(B*PM, NG, D)
            else:
                if self.attn_type=='Mul':
                    cat = torch.cat([mems, h], dim=1)
                else:
                    cat = torch.cat([mems, h], dim=2)
        else:
            cat = h
        
        if prev is not None:
            [attn_score_h_prev, attn_score_g_prev] = prev
        else:
            attn_score_h_prev, attn_score_g_prev = None, None

        if sparse_attn is not None:
            # print(f'MultiheadRelAttention, NOT None, sparse_attn.shape = {sparse_attn.shape}')
            if sparse_attn_mem is not None:
                sparse_attn = torch.cat([sparse_attn_mem, sparse_attn], dim=-1)
        
        # content-based key head
        k_head_h = self.proj_drop(head_projection(cat, 'k', proj_weight=self.k_proj_weight, attn_type=self.attn_type))
        # content-based value head
        v_head_h = self.proj_drop(head_projection(cat, 'v', proj_weight=self.v_proj_weight, attn_type=self.attn_type))
        # position-based key head
        if r_h is not None:
            k_head_r = self.proj_drop(head_projection(r_h, 'r', proj_weight=self.r_proj_weight))
        else:
            k_head_r = None

        ##### h-stream #####
        ## content-stream query head
        q_head_h = self.proj_drop(head_projection(h, 'q', proj_weight=self.q_proj_weight, attn_type=self.attn_type))
        #### core attention ops
        #### hˆ(m)_zt = LayerNorm(h^(m-1)_zt + RelAttn(h^(m-1)_zt + [h~^(m-1), hT(m-1)_z<=t]))
        attn_vec_h, attn_prob_h, attn_score_h = self.rel_attn(q_head=q_head_h, k_head_h=k_head_h, v_head_h=v_head_h, k_head_r=k_head_r, 
                                                              r_w_bias=r_w_bias, r_r_bias=r_r_bias, r_s_bias=r_s_bias, 
                                                              attn_mask=attn_mask_h, 
                                                              seg_embed=seg_embed, seg_mat=seg_mat, 
                                                              prev=attn_score_h_prev, 
                                                              sparse_attn=sparse_attn)
        output_h = self.post_attention(h, attn_vec_h, self.proj_out, stream='h')

        ##### g-stream #####
        output_g, attn_prob_g, attn_score_g = None, None, None
        if pretrain:
            assert (target_mapping is not None)
            ## query-stream query head
            q_head_g = self.proj_drop(head_projection(g, 'q', proj_weight=self.q_proj_weight, attn_type=self.attn_type))   
            # [bsz, num_mask, n_head, d_head]
            #### core attention ops
            #### gˆ(m)_zt = LayerNorm(g^(m-1)_zt + RelAttn(g^(m-1)_zt + [h~^(m-1), hT(m-1)_z<=t]))
            if self.attn_type=='Mul':                                                   # target_mapping : [bsz, num_mask, q_len]
                q_head_g = torch.einsum('bmnd,bml->blnd', q_head_g, target_mapping)     # q_head_g       : [bsz, qlen, n_head, d_head]
            else:                                                                       # target_mapping : [bsz, n_vars, num_mask, patch_num]
                q_head_g = torch.einsum('bkmnd,bkml->bklnd', q_head_g, target_mapping)  # q_head_g       : [bsz, n_vars, patch_num, n_head, d_head]
                
            if self.attn_type=='Mul' and self.efficient: 
                k_head_h = einops.rearrange(k_head_h[:,:N], '(b p) k n d -> b (p k) n d', b=B)
                v_head_h = einops.rearrange(v_head_h[:,:N], '(b p) k n d -> b (p k) n d', b=B)
                # k_head_r = self.h2g(k_head_r, n_vars=N, n_cls=G, name='r')
            if r_g is not None: # [bsz, klen, n_head, d_head] for 'Mul'; [bsz, klen_uni, n_head, d_head] for 'Uni'
                k_head_r = self.proj_drop(head_projection(r_g, 'r', proj_weight=self.r_proj_weight)) 
            else:
                k_head_r = None
            attn_vec_g, attn_prob_g, attn_score_g = self.rel_attn(q_head=q_head_g, k_head_h=k_head_h, v_head_h=v_head_h, k_head_r=k_head_r, 
                                                                  r_w_bias=r_w_bias, r_r_bias=r_r_bias, r_s_bias=r_s_bias, 
                                                                  attn_mask=attn_mask_g,
                                                                  seg_embed=seg_embed, seg_mat=seg_mat,
                                                                  g_stream=True,
                                                                  prev=attn_score_g_prev,
                                                                  sparse_attn=sparse_attn)
            if self.attn_type=='Mul':
                attn_vec_g = torch.einsum('blnd,bml->bmnd', attn_vec_g, target_mapping)  # attn_vec_g : [bsz, num_mask, n_head, d_head]
            else:
                attn_vec_g = torch.einsum('bklnd,bkml->bkmnd', attn_vec_g, target_mapping)  # attn_vec_g : [bsz, n_vars, num_mask, n_head, d_head]
            output_g = self.post_attention(g, attn_vec_g, self.proj_out, stream='g')

        return output_h, output_g, [attn_prob_h, attn_prob_g], [attn_score_h, attn_score_g]
