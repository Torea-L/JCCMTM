from __future__ import absolute_import, division, print_function

__all__ = ['Pretrain_Model']

from typing import Optional

import einops
import torch
import torch.nn as nn
from layers.attention import MultiheadRelAttention, positionwise_ffn
from layers.basics import get_activation_fn, get_norm_fn, series_decomp
from layers.embed import relative_positional_encoding
from layers.revin import RevIN
from torch import Tensor
from utils.masking import attention_mask

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

class AttrProxy(object):
    """Translates index lookups into attribute lookups."""
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))
    
class JCC_backbone(nn.Module):
    def __init__(self, attn_direction, clamp_len=-1, same_length=False, use_seg_emb=False, learn_pe:bool=True, 
                 reuse_len:int=0, reuse_len_uni:int=0, mem_len:int=0, mem_len_uni:int=0,
                 sparse_attn_:torch.Tensor=None, sparse_attn_mem_:torch.Tensor=None,
                 mul_uni_ratio:int=1, group_token_num=10,
                 configs=None, efficient=False, details=False):
        super(JCC_backbone, self).__init__()
        
        self.configs = configs
        # model Hyperparameters #
        self.n_layer = configs.model.n_layers
        self.ratio = mul_uni_ratio
        self.n_layer_uni = int(self.n_layer*mul_uni_ratio)
        self.n_head = configs.model.n_heads
        self.d_head = configs.model.d_head
        self.d_inner = configs.model.d_ff
        self.d_model = configs.model.d_model
        self.dropout = configs.model.dropout
        self.dropout_attn = configs.model.dropout_att
        self.group_token_num = group_token_num
        # self.n_vars = configs.data.n_vars

        self.reuse_len = reuse_len
        self.reuse_len_uni = reuse_len_uni
        self.mem_len = mem_len
        self.mem_len_uni = mem_len_uni
        # model settings #
        self.clamp_len = clamp_len
        self.attn_direction = attn_direction
        self.same_length = same_length
        self.use_seg_emb = use_seg_emb
        self.efficient = efficient
        self.details = details
        # learnable vector `u` in relative position encoding
        self.r_w_bias = nn.Parameter(nn.init.xavier_normal_(
            torch.randn(self.n_layer, self.n_head, self.d_head)))  
        self.r_w_bias_uni = nn.Parameter(nn.init.xavier_normal_(
            torch.randn(self.n_layer_uni, self.n_head, self.d_head)))
        # learnable vector `v` in relative position encoding
        self.r_r_bias = nn.Parameter(nn.init.xavier_normal_(
            torch.randn(self.n_layer, self.n_head, self.d_head)))
        self.r_r_bias_uni = nn.Parameter(nn.init.xavier_normal_(
            torch.randn(self.n_layer_uni, self.n_head, self.d_head)))
        
        self.mask_emb = nn.Parameter(nn.init.xavier_normal_(torch.randn(1, 1, self.d_model)))
        self.cls = nn.Parameter(nn.init.xavier_normal_(torch.randn(1, 1, 1, self.d_model)))

        self.pos_emb_mul = None
        self.pos_emb_uni = None
        self.pos_emb = None
        self.attn_mask_cache = None
        self.attn_mask_uni_cache = None

        self.klen_uni = configs.data.patch_num
        self.klen = configs.data.patch_num*configs.data.n_vars
        
        if use_seg_emb:
            ##### Segment embedding
            self.r_s_bias = nn.Parameter(nn.init.xavier_normal_(
                torch.randn(self.n_layer, self.n_head, self.d_head)))
            self.r_s_bias_uni = nn.Parameter(nn.init.xavier_normal_(
                torch.randn(self.n_layer_uni, self.n_head, self.d_head)))
            self.seg_embed = nn.Parameter(nn.init.xavier_normal_(
                torch.randn(self.n_layer, 2, self.n_head, self.d_head)))
            self.seg_embed_uni = nn.Parameter(nn.init.xavier_normal_(
                torch.randn(self.n_layer_uni, 2, self.n_head, self.d_head)))
        
        self.Dropout_h = nn.Dropout(p=self.dropout)
        self.Dropout_g = nn.Dropout(p=self.dropout)
        self.Dropout_pos_mul = nn.Dropout(p=self.dropout)
        self.Dropout_pos_uni = nn.Dropout(p=self.dropout)

        self.inp_projection_mul = nn.Linear(configs.data.patch_len, self.d_model, bias=True)
        self.inp_projection_uni = nn.Linear(configs.data.patch_len, self.d_model, bias=True)
        
        for i in range(self.n_layer):
            self.add_module('TS_Attn_mul_' + str(i), MultiheadRelAttention(configs=self.configs, attn_type='Mul', 
                                                                           proj_drop=0.1, out_drop=self.dropout, attn_drop=self.dropout_attn,
                                                                        #    use_threshold=True, threshold=1e-2,
                                                                           norm_h='layer_norm', norm_shape_h=self.d_model,
                                                                           norm_g='layer_norm', norm_shape_g=self.d_model,
                                                                           cos_attn=True, res_attention=True, efficient=efficient))
            self.add_module('PW_FFN_mul_' + str(i), positionwise_ffn(d_model=self.d_model, d_ff=self.d_inner, dropout=self.dropout, 
                                                                     norm_h='layer_norm', norm_shape_h=self.d_model, 
                                                                     norm_g='layer_norm', norm_shape_g=self.d_model, 
                                                                     activation_type='gelu'))
            self.add_module('Norm_' +str(i), nn.LayerNorm(normalized_shape=self.d_model))
        for j in range(self.n_layer_uni):
            self.add_module('TS_Attn_uni_' + str(j), MultiheadRelAttention(configs=self.configs, attn_type='Uni', 
                                                                           proj_drop=0.1, out_drop=self.dropout, attn_drop=self.dropout_attn,
                                                                        #    use_threshold=True, threshold=1e-2,
                                                                           norm_h='layer_norm', norm_shape_h=self.d_model,
                                                                           norm_g='layer_norm', norm_shape_g=self.d_model,
                                                                           cos_attn=False, res_attention=True))
            self.add_module('PW_FFN_uni_' + str(j), positionwise_ffn(d_model=self.d_model, d_ff=self.d_inner, dropout=self.dropout, 
                                                                     norm_h='layer_norm', norm_shape_h=self.d_model, 
                                                                     norm_g='layer_norm', norm_shape_g=self.d_model, 
                                                                     activation_type='gelu'))
        self.TS_Attn_mul = AttrProxy(self, 'TS_Attn_mul_')
        self.TS_Attn_uni = AttrProxy(self, 'TS_Attn_uni_')
        
        self.PW_FFN_mul = AttrProxy(self, 'PW_FFN_mul_')
        self.PW_FFN_uni = AttrProxy(self, 'PW_FFN_uni_')

        self.Norm = AttrProxy(self, 'Norm_')

        self.sparse_attn_mask = sparse_attn_ # [1, n_head, qlen, qlen]
        self.sparse_attn_mask_mem = sparse_attn_mem_ # [1, n_head, qlen, mlen]

    @torch.no_grad()
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.details:
                    print('In Transformer Model, Linear layer: ', m)
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def _cache_mem(self, curr_out, prev_mem, mem_len, reuse_len=None, attn_module='Mul', efficient=False):
        """
        Cache hidden states into memory.
        cached mem does not participate in gradient calculation.
        Args:
            Mul attn_module:
                curr_out(not-efficient) : [bsz, seq_len, d_model]
                curr_out(efficient) : [bsz*PM, n_vars+n_cls, d_model]
                prev_mem : [bsz, seq_len, d_model]
            Uni attn_module:
                curr_out : [bsz, n_vars, patch_num, d_model]
                prev_mem : [bsz, n_vars, patch_num, d_model]
        """
        assert attn_module in ['Mul', 'Uni'], f"Unrecognized dset (`{attn_module}`). Options include: {['Mul', 'Uni']}"
        new_mem = None
        with torch.no_grad():
            if mem_len is None or mem_len == 0:
                return None
            else:
                if attn_module=='Uni':
                    curr_out = einops.rearrange(curr_out, 'b k p d -> (b k) p d')
                    if prev_mem is not None:
                        prev_mem = einops.rearrange(prev_mem, 'b k p d -> (b k) p d')
                if attn_module=='Mul' and efficient:
                    curr_out = einops.rearrange(curr_out, '(b p) k d -> b (p k) d', p=self.configs.data.patch_num)
                
                if reuse_len is not None and reuse_len > 0:
                    curr_out = curr_out[:,:reuse_len]

                if prev_mem is None:
                    new_mem = curr_out[:,-mem_len:]
                else:
                    new_mem = torch.cat([prev_mem, curr_out], dim=1)[:, -mem_len:]
            if attn_module=='Mul' and efficient:
                new_mem = einops.rearrange(new_mem, 'b (p k) d -> (b p) k d', p=mem_len)
            elif attn_module=='Uni':
                new_mem = einops.rearrange(new_mem, '(b k) p d -> b k p d', k=self.configs.data.n_vars)
        return new_mem

    def forward(self, inp_k:Tensor, inp_k_trend:Tensor, seg_id:Tensor=None, 
                input_mask:Tensor=None, input_mask_uni:Tensor=None, 
                mems:list=None, mems_uni:list=None,
                perm_mask:Tensor=None, perm_mask_uni:Tensor=None,
                target_mapping:Tensor=None, target_mapping_uni:Tensor=None, 
                target_masked_idx:Tensor=None, 
                pretrain:bool=False, strategy='CICD'):
        """
        Args:
            inp_k: Tensor in shape [B, P, N, patch_len], input token embedding
            inp_k_trend: Tensor in shape [B, P, N, patch_len]
            seg_id: Tensor in shape [B, qlen], segment ID of the input token embedding
            target_mapping: Tensor in shape [B, num_mask, qlen]
            target_mapping_uni : [B, N, num_mask, qlen]
            target_masked_idx : [B, N, num_mask, D]
        """
        P, N, D= self.configs.data.patch_num, self.configs.data.n_vars, self.d_model
        G = self.group_token_num if self.efficient else None
        B = inp_k.shape[0]
        qlen = P*N
        
        qlen_uni = P
        if self.mem_len==0: mlen = 0
        else: mlen = mems[0].size(0) if mems is not None else 0
        if self.mem_len_uni==0: mlen_uni = 0
        else:
            mlen_uni = mems_uni[0].size(2) if (mems_uni is not None)and(len(mems_uni)>0) else 0
        klen = mlen + qlen
        klen_uni = qlen_uni + mlen_uni

        PM = int(klen/N)
        ## number of masked tokens
        num_mask = target_mapping.shape[1] if (target_mapping is not None) else None

        #### Attention mask ####
        """
        Returns of Function `attention_mask`:
            attn_mask : [bsz, 1, qlen, klen]
            attn_mask_uni : [bsz, N, 1, P, PM]
            non_tgt_mask(not-efficient) : [bsz, 1, qlen, klen]
            non_tgt_mask(efficient) : [P*G, PM*G]
            non_tgt_mask_uni : [bsz, N, 1, P, PM]
        """
        with torch.no_grad():
            if self.klen != klen:
                self.attn_mask_cache = None
            if self.klen_uni != klen_uni:
                self.attn_mask_uni_cache = None
            attn_mask, attn_mask_uni, non_tgt_mask, non_tgt_mask_uni, attn_mask_cache, attn_mask_uni_cache = \
                attention_mask(batch_size=B, num_vars=N,
                               qlen=P*N, qlen_uni=P, mlen=mlen, mlen_uni=mlen_uni,
                               input_mask=input_mask, input_mask_uni=input_mask_uni, 
                               perm_mask=perm_mask, perm_mask_uni=perm_mask_uni, 
                               attn_mask_cache=self.attn_mask_cache, attn_mask_uni_cache=self.attn_mask_uni_cache,
                               n_cls=G, efficient=self.efficient, device=inp_k.device,
                               attn_direction=self.attn_direction, same_length=self.same_length)
            if (self.attn_mask_cache is None):
                self.attn_mask_cache = attn_mask_cache 
            if (self.attn_mask_uni_cache is None):
                self.attn_mask_uni_cache = attn_mask_uni_cache
            
            if pretrain:
                if self.efficient:
                    non_tgt_mask = einops.repeat(non_tgt_mask[None,None,:,:], '1 1 i j -> b 1 i j', b=B).to(inp_k.device)
                else:
                    attn_mask = attn_mask.to(inp_k.device)
                    non_tgt_mask = non_tgt_mask.to(inp_k.device)
                attn_mask_uni = einops.rearrange(attn_mask_uni, 'b k n i j -> (b k) n i j').to(inp_k.device)
                non_tgt_mask_uni = einops.rearrange(non_tgt_mask_uni, 'b k n i j-> (b k) n i j').to(inp_k.device)
        
        #### Group token Embedding ####
        if self.efficient:
            self.group_cls = einops.repeat(self.cls, '1 1 1 d -> b l g d', b=B, l=P, g=G).view(B*P, G, D)

        #### Word embedding ####
        """
        Args:
            inp_k_mul : [bsz*patch_num, n_vars, patch_len]
            inp_k_uni : [bsz, n_vars, patch_num, patch_len]
            emb_k_mul : [bsz*patch_num, n_vars, d_model]
            emb_k_uni : [bsz, n_vars, patch_num, d_model]
            output_h_mul : [n_vars + n_cls, PM*bsz, d_model]
            output_h_uni : [n_vars, patch_num, bsz, d_model]
        """
        # inp_k = inp_k.view(B ,P, N, self.configs.data.patch_len)
        # inp_k_trend = inp_k_trend.view(B ,P, N, self.configs.data.patch_len)
        inp_k_uni = inp_k_trend.permute(0,2,1,3).contiguous().view(B, N, P, self.configs.data.patch_len)
        if self.efficient:
            inp_k_mul = inp_k.contiguous().view(B*P, N, self.configs.data.patch_len)
        else:
            inp_k_mul = inp_k.contiguous().view(B, P*N, self.configs.data.patch_len)

        #### Content Stream(Original Attention), h^(0)_t = e(x_i) ####
        emb_k_mul = self.inp_projection_mul(inp_k_mul) # [bsz*patch_num, n_vars, d_model]
        emb_k_uni = self.inp_projection_uni(inp_k_uni) # [bsz*n_vars, patch_num, d_model]

        if self.efficient:
            with torch.no_grad():
                zero_mem_pad4mul = torch.zeros((B*(PM-P), N+G, D)).cuda() # [bsz*(PM - P), n_vars + n_cls, d_model]
            emb_k_mul = torch.cat((emb_k_mul, self.group_cls), dim=1) # [bsz*P, n_vars + n_cls, d_model]
            emb_k_mul = torch.cat((zero_mem_pad4mul, emb_k_mul), dim=0) # [bsz*PM, n_vars + n_cls, d_model]
        
        output_h_mul = self.Dropout_h(emb_k_mul) # [bsz*PM, n_vars + n_cls, d_model]
        output_h_uni = self.Dropout_h(emb_k_uni) # [bsz, n_vars, patch_num, d_model]

        #### Query Stream, g^(0)_t = w
        #### the first layer query stream is initialized with a trainable vector
        """
            word_emb_q.shape : [bsz, num_mask, d_model]
            output_g_mul : [bsz, num_mask, d_model]
            output_g_uni : [bsz, n_vars, num_mask, d_model]
        """
        word_emb_q, output_g_mul, output_g_uni = None, None, None
        if pretrain:
            if target_mapping is not None:
                ## target_mapping.shape = [bsz, num_mask, sqe_len]
                word_emb_q = einops.repeat(self.mask_emb, '1 1 d -> b m d', m=num_mask, b=B) # [bsz, num_mask, d_model]
            
            output_g_mul = self.Dropout_g(word_emb_q) # [bsz, num_mask, d_model]
            output_g_uni = torch.einsum('bmh,bkm->bkmh', output_g_mul, target_masked_idx) # [bsz, n_vars, num_mask, d_model]

        #### Positional encoding ####
        """
            pos_emb_mul(efficient) : [bsz, PM*n_cls, d_model]
            pos_emb_mul(not-efficient) : [bsz, klen, d_model]
            pos_emb_uni : [bsz*N, klen_uni, d_model]
        """
        if self.efficient and (self.pos_emb is None):
            self.pos_emb = relative_positional_encoding(
                qlen=qlen, klen=PM, k_dim=N, d_model=D, clamp_len=self.clamp_len, attn_direction=self.attn_direction, 
                bi_data=False, h_number=None, batch_first=True, dtype=torch.float32) # [1, PM, d_model]
        if (self.pos_emb_mul is None) or (self.klen != klen):
            self.pos_emb_mul = relative_positional_encoding(
                qlen=qlen, klen=klen, k_dim=N, d_model=self.d_model, clamp_len=self.clamp_len, attn_direction=self.attn_direction, 
                bi_data=False, h_number=P, batch_first=True, dtype=torch.float32) # [1, klen, d_model]
        if (self.pos_emb_uni is None) or (self.klen_uni != klen_uni):
            self.pos_emb_uni = relative_positional_encoding(
                qlen=qlen, klen=klen_uni, k_dim=N, d_model=D, clamp_len=self.clamp_len, attn_direction=self.attn_direction, 
                bi_data=False, h_number=None, batch_first=True, dtype=torch.float32) # [1, klen_uni, d_model]

        if self.efficient: # pos_emb_mul_h : [bsz, PM*n_cls, d_model]
            pos_emb_mul_h = self.Dropout_pos_mul(einops.repeat(self.pos_emb, '1 p d -> b (p g) d', b=B, g=G)).to(inp_k.device) 
        else: # pos_emb_mul_h : [bsz, klen, d_model]
            pos_emb_mul_h = self.Dropout_pos_mul(einops.repeat(self.pos_emb_mul, '1 p d -> b p d', b=B)).to(inp_k.device)
        
        pos_emb_mul_g = self.Dropout_pos_mul(einops.repeat(self.pos_emb_mul, '1 p d -> b p d', b=B)).to(inp_k.device)
        pos_emb_uni = self.Dropout_pos_uni(einops.repeat(self.pos_emb_uni, '1 p d -> b p d', b=B)).to(inp_k.device)

        self.klen_uni = klen_uni
        self.klen = klen

        new_mems = []
        new_mems_uni = []
        attn_prob_lst_M = []
        attn_prob_lst_U = []
        attn_score_lst_M = []
        attn_score_lst_U = []
        prev_mul = None
        prev_uni = None

        ##### Attention layers #####
        with torch.no_grad():
            if mems is None:
                mems = [None] * self.n_layer
            if mems_uni is None:
                mems_uni = [None] * self.n_layer_uni

        for i in range(self.n_layer):
            # cache new mems
            if self.mem_len > 0:
                # TODO : need to rearrange the shape !!!
                new_mems.append(self._cache_mem(output_h_mul, mems[i], self.mem_len, self.reuse_len))
            else: new_mems.append(None)

            PosFFN_uni = None

            # segment bias
            r_s_bias_i_mul = None
            r_s_bias_i_uni = None
            seg_embed_i_mul = None
            seg_embed_i_uni = None
            if self.use_seg_emb and (seg_id is not None):
                r_s_bias_i_mul = self.r_s_bias[i]
                seg_embed_i_mul = self.seg_embed[i]
            ## Sparse attention is only used in Multi-Module
            if mems[i] is not None:
                sparse_attn_mem = self.sparse_attn_mask_mem
            else:
                sparse_attn_mem = None
            ## computation
            if 'CD' in strategy:
                output_h_mul, output_g_mul, attn_prob_lst_mul, attn_score_lst_mul = self.TS_Attn_mul[i](
                    h=output_h_mul, g=output_g_mul, r_h=pos_emb_mul_h, r_g=pos_emb_mul_g, mems=mems[i], # r=None,
                    r_w_bias=self.r_w_bias[i], r_r_bias=self.r_r_bias[i], r_s_bias=r_s_bias_i_mul, 
                    seg_mat=None, seg_embed=seg_embed_i_mul, 
                    attn_mask_h=non_tgt_mask, attn_mask_g=attn_mask, 
                    target_mapping=target_mapping,
                    sparse_attn=self.sparse_attn_mask, sparse_attn_mem=sparse_attn_mem, 
                    pretrain=pretrain, prev=prev_mul)
                prev_mul = attn_score_lst_mul
            if 'CI' in strategy:
                for j in range(self.ratio):
                    k = i*self.ratio+j
                    ## cache new mems
                    if self.mem_len_uni > 0:
                        new_mems_uni.append(self._cache_mem(output_h_uni, mems_uni[k], self.mem_len_uni, self.reuse_len_uni, attn_module='Uni'))
                    else: new_mems_uni.append(None)

                    if self.use_seg_emb and (seg_id is not None):
                        r_s_bias_i_uni = self.r_s_bias_uni[k]
                        seg_embed_i_uni = self.seg_embed_uni[k]

                    output_h_uni, output_g_uni, attn_prob_lst_uni, attn_score_lst_uni = self.TS_Attn_uni[k](
                        h=output_h_uni, g=output_g_uni, r_h=pos_emb_uni, r_g=pos_emb_uni, mems=mems_uni[i],
                        r_w_bias=self.r_w_bias_uni[k], r_r_bias=self.r_r_bias_uni[k], r_s_bias=r_s_bias_i_uni, 
                        seg_mat=None, seg_embed=seg_embed_i_uni,
                        attn_mask_h=non_tgt_mask_uni, attn_mask_g=attn_mask_uni, 
                        target_mapping=target_mapping_uni,
                        sparse_attn=None, sparse_attn_mem=None,
                        pretrain=pretrain, prev=prev_uni)
                    prev_uni = attn_score_lst_uni

                    PosFFN_uni = self.PW_FFN_uni[k]
                    if pretrain:
                        output_g_uni = PosFFN_uni(inp=output_g_uni, stream='g')
                    output_h_uni = PosFFN_uni(inp=output_h_uni, stream='h')
            
            if 'CD' in strategy:
                PosFFN_mul = self.PW_FFN_mul[i]
                if pretrain:
                    output_g_mul = PosFFN_mul(inp=output_g_mul, stream='g')
                output_h_mul = PosFFN_mul(inp=output_h_mul, stream='h')
                attn_prob_lst_M.append(attn_prob_lst_mul)
                attn_score_lst_M.append(attn_score_lst_mul)
            
            if 'CI' in strategy:
                attn_prob_lst_U.append(attn_prob_lst_uni)
                attn_score_lst_U.append(attn_score_lst_uni)

            if strategy == 'CICD':
                #### Uni-to-Mul ####
                """
                output_h_mul : [bsz*PM, n_vars + n_cls, d_model]
                output_h_uni : [bsz, n_vars, patch_num, d_model]
                """
                if self.efficient:
                    output_h_uni2mul = einops.rearrange(output_h_uni, 'b k i d -> b i k d') # output_h_uni2mul : [B, P, N, D]
                    group_cls = output_h_mul[-B*P:,-G:] # group_cls : [B*P, G, D]
                    output_h_mul = einops.rearrange(output_h_mul, '(b p) k d -> b p k d', b=B)[:,-P:,:N] # output_h_mul : [B, P, N, D]
                    output_h_mul += output_h_uni2mul
                    output_h_mul = torch.cat((output_h_mul.view(B*P, N, D), group_cls), dim=1) # output_h_mul : [B*P, N+G, D]
                    output_h_mul = self.Norm[i](output_h_mul)
                    output_h_mul = torch.cat((zero_mem_pad4mul , output_h_mul), dim=0) # output_h_mul : [B*PM, N+G, D]
                else:
                    output_h_uni2mul = einops.rearrange(output_h_uni, 'b k i d -> b (i k) d') # output_h_uni2mul : [B, P*N, D]
                    output_h_mul += output_h_uni2mul
                    output_h_mul = self.Norm[i](output_h_mul)
                if pretrain:
                    """
                    output_g_mul : [bsz, num_mask, d_model]
                    output_g_uni : [bsz, n_vars, num_mask, d_model]
                    """
                    # output_g_uni2mul = torch.einsum('kmbh,kmbh->mbh', output_g_uni, target_masked_idx)
                    output_g_uni2mul = torch.einsum('bkmh,bkm->bmh', output_g_uni, target_masked_idx)
                    output_g_mul += output_g_uni2mul
                    output_g_mul = self.Norm[i](output_g_mul)
        
        if self.efficient:
            output_h_mul = einops.rearrange(output_h_mul, '(b p) k d -> b p k d', b=B)[:,-P:,:N] # output_h_mul : [B, P, N, D]
        else:
            output_h_mul = output_h_mul.view(B, P, N, D)

        return output_g_mul, output_h_mul, output_g_uni, output_h_uni, [new_mems, new_mems_uni], \
    [attn_prob_lst_M, attn_prob_lst_U],[attn_score_lst_M, attn_score_lst_U]

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

class JCCEncoder(nn.Module):
    def __init__(self, transformer, linear, configs, kernel_size=7):
        super(JCCEncoder, self).__init__()
        self.configs = configs
        self.n_vars = configs.data.n_vars
        self.revin = RevIN(num_features=self.n_vars)
        self.transformer = transformer
        self.linear_model_uni = linear
        self.norm_layer = nn.LayerNorm(configs.model.d_model)

    def forward(self, inp_k:Tensor, inp_decomp:Tensor=None, seg_id:Tensor=None, 
                input_mask:Tensor=None, input_mask_uni:Tensor=None, mems:list=None, mems_uni:list=None,
                perm_mask:Tensor=None, perm_mask_uni:Tensor=None,
                target_mapping:Tensor=None, target_mapping_uni:Tensor=None, 
                target_masked_idx:Tensor=None,
                pretrain:bool=False, strategy='CICD'):
        """
        B : batch size
        P : number of patches in each channel
        L : length of patches (patch_len)
        N : number of variables/channels
        """
        # B, P, N, L = inp_k.shape
        # Normalization from Non-stationary Transformer
        inp_k = data_revin(x=inp_k, revin=self.revin, mode='norm', stream='h')

        if inp_decomp is None:
            inp_decomp = inp_k
        else:
            inp_decomp = data_revin(x=inp_decomp, revin=self.revin, mode='norm', stream='h')
        
        output_g_mul, output_h_mul, output_g_uni, output_h_uni, mem_lst, attn_prob_lst, attn_score_lst = \
            self.transformer(inp_k, inp_decomp, seg_id, input_mask, input_mask_uni, mems, mems_uni, 
                             perm_mask, perm_mask_uni, 
                             target_mapping, target_mapping_uni, target_masked_idx,
                             pretrain, strategy)
        return output_g_mul, output_h_mul, output_g_uni, output_h_uni, mem_lst, attn_prob_lst, attn_score_lst
    
