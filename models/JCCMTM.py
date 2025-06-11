from __future__ import absolute_import, division, print_function

__all__ = ['Pretrain_Model']

from typing import Optional

import einops
import torch
import torch.nn as nn
from layers.revin import RevIN
from torch import Tensor

"""
Args:
            B : batch size
            N : n_vars, number of variates in MTS
            G : n_cls, number of group tokens
            NG : n_vars + n_cls
            P : patch_num, number of patches, (qlen_uni)
            M : number of patches cached in memory in 'Mul-Module'
            PM : P + M; number of patches in 'Mul-Module'
            D : d_model
            qlen = P*N
            mlen = M*N
            klen = qlen + mlen = PM*N
            mlen_uni : number of patches cached in memory in 'Uni-Module'
"""
def data_revin(x:Tensor, revin:RevIN, mode:str, stream='h', target_masked_idx:Optional[Tensor]=None):
    """
    B : batch size
    P : number of patches in each channel
    L : length of patches (patch_len)
    M : number of masked tokens
    param:
        x : [bsz, patch_num, n_vars, patch_len] in content stream
        x : [bsz, num_mask, patch_len] in query stream
    """
    assert stream in ['h', 'g'], f"Unrecognized dset (`{stream}`). Options include: {['h', 'g']}"
    if stream == 'h':
        B, P, N, L = x.shape
        x = einops.rearrange(x, 'b p k l -> b (p l) k', p=P)
        x = revin(x, mode)
        x = einops.rearrange(x, 'b (p l) k -> b p k l', p=P)
    else:
        B, M, L = x.shape
        x = torch.einsum('bmh,bkm->bkmh', x, target_masked_idx)      # x : [bsz, n_vars, num_mask, patch_len]
        x = einops.rearrange(x, 'b k m l -> b (m l) k')
        x = revin(x, mode)
        x = einops.rearrange(x, 'b (m l) k -> b k m l', m=M)
        x = torch.einsum('bkmh,bkm->bmh', x, target_masked_idx)      # x : [bsz, num_mask, patch_len]
    
    return x

class Model(nn.Module):
    def __init__(self, configs, encoder, cross_domain=False, n_vars_cross_domian=7):
        super(Model, self).__init__()
        self.configs = configs
        self.n_vars = configs.data.n_vars
        '''if cross_domain:
            self.revin = RevIN(num_features=n_vars_cross_domian)
        else:
            self.revin = RevIN(num_features=self.n_vars)'''
        self.revin = RevIN(num_features=self.n_vars)
        self.encoder = encoder
        self.norm_layer = nn.LayerNorm(configs.model.d_model)

    def forward(self, inp_k:Tensor, inp_decomp:Tensor=None, seg_id:Tensor=None, 
                sparse_attn_mask:Optional[Tensor]=None, sparse_attn_mem_mask:Optional[Tensor]=None,
                mems_mul:list=None, mems_uni:list=None,
                input_mask_mul:Tensor=None, input_mask_uni:Tensor=None,
                perm_mask_mul:Tensor=None, perm_mask_uni:Tensor=None,
                target_mapping_mul:Tensor=None, target_mapping_uni:Tensor=None, 
                target_masked_idx:Tensor=None,
                pretrain:bool=False, strategy='CICD'):
        """
        B : batch size
        P : number of patches in each channel
        L : length of patches (patch_len)
        N : number of variables/channels
        """
        # Normalization from Non-stationary Transformer
        inp_k = data_revin(x=inp_k, revin=self.revin, mode='norm', stream='h')

        if inp_decomp is None:
            inp_decomp = inp_k
        else:
            inp_decomp = data_revin(x=inp_decomp, revin=self.revin, mode='norm', stream='h')
        
        output_g_mul, output_h_mul, output_g_uni, output_h_uni, mem_lst, attn_prob_lst, attn_score_lst = \
            self.encoder(inp_k, inp_decomp, seg_id, 
                         sparse_attn_mask, sparse_attn_mem_mask, mems_mul, mems_uni, 
                         input_mask_mul, input_mask_uni, perm_mask_mul, perm_mask_uni, 
                         target_mapping_mul, target_mapping_uni, target_masked_idx, 
                         pretrain, strategy)
        return output_g_mul, output_h_mul, output_g_uni, output_h_uni, mem_lst, attn_prob_lst, attn_score_lst
    
