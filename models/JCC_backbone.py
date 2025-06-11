__all__ = ['JCC_backbone']

from typing import Optional
import argparse

import einops
import torch
import torch.nn as nn
from torch import Tensor

from layers.attention import MultiheadRelAttention, positionwise_ffn
from layers.embed import relative_positional_encoding
from utils.masking import attention_mask


class AttrProxy(object):
    """Translates index lookups into attribute lookups."""
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))

class JCC_backbone(nn.Module):
    def __init__(self, configs:argparse.Namespace=None, attn_direction='uni', clamp_len=-1, 
                 same_length:bool=False, use_seg_emb:bool=False, learn_pe:bool=True, 
                 reuse_len_mul:int=0, reuse_len_uni:int=0, mem_len_mul:int=0, mem_len_uni:int=0,
                 mul_uni_ratio:int=1, group_token_num=10, 
                 efficient:bool=False, res_attention:bool=True, verbose:bool=False):
        super(JCC_backbone, self).__init__()
        self.configs = configs
        self.ratio = mul_uni_ratio
        self.n_layer = configs.model.e_layers
        self.n_layer_uni = int(self.n_layer*mul_uni_ratio)
        self.n_head = configs.model.n_heads
        self.d_head = configs.model.d_head
        self.d_inner = configs.model.d_ff
        self.d_model = configs.model.d_model
        self.dropout = configs.model.dropout

        self.mem_len_mul = mem_len_mul
        self.mem_len_uni = mem_len_uni

        self.clamp_len = clamp_len
        self.attn_direction = attn_direction
        self.same_length = same_length
        self.use_seg_emb = use_seg_emb
        self.efficient = efficient
        self.group_token_num = group_token_num

        self.verbose = verbose

        self.klen_mul = configs.data.patch_num*configs.data.n_vars
        self.klen_uni = configs.data.patch_num

        self.pos_emb_mul = None
        self.pos_emb_uni = None
        self.pos_emb = None
        self.attn_mask_mul_cache = None
        self.attn_mask_uni_cache = None

        self.mask_emb = nn.Parameter(nn.init.xavier_normal_(torch.randn(1, 1, self.d_model)))
        self.cls = nn.Parameter(nn.init.xavier_normal_(torch.randn(1, 1, 1, self.d_model)))

        self.backbone =JCCEncoder(mem_len_mul=mem_len_mul, mem_len_uni=mem_len_uni, 
                                  reuse_len_mul=reuse_len_mul, reuse_len_uni=reuse_len_uni,
                                  configs=configs, mul_uni_ratio=mul_uni_ratio, 
                                  group_token_num=group_token_num, 
                                  efficient=efficient, res_attention=res_attention, use_seg_emb=use_seg_emb)

        self.Dropout_h_mul = nn.Dropout(p=self.dropout)
        self.Dropout_g_mul = nn.Dropout(p=self.dropout)
        self.Dropout_pos_mul = nn.Dropout(p=self.dropout)
        self.Dropout_pos_uni = nn.Dropout(p=self.dropout)

        self.projection_mul = nn.Linear(configs.data.patch_len, self.d_model, bias=True)
        self.projection_uni = nn.Linear(configs.data.patch_len, self.d_model, bias=True)

    def forward(self, inp_k:Tensor, inp_k_decomp:Tensor, seg_id:Tensor=None,  
                sparse_attn_mask:Optional[Tensor]=None, sparse_attn_mem_mask:Optional[Tensor]=None,
                mems_mul:list=None, mems_uni:list=None,
                input_mask_mul:Optional[Tensor]=None, input_mask_uni:Optional[Tensor]=None,
                perm_mask_mul:Optional[Tensor]=None, perm_mask_uni:Optional[Tensor]=None,
                target_mapping_mul:Optional[Tensor]=None, target_mapping_uni:Optional[Tensor]=None, 
                target_masked_idx:Optional[Tensor]=None, 
                pretrain:bool=False, strategy:str='CICD'):
        """
        Args:
            inp_k : Tensor in shape [B, P, N, patch_len], input token embedding
            inp_k_decomp : Tensor in shape [B, P, N, patch_len]
            seg_id : Tensor in shape [B, qlen_mul], segment ID of the input token embedding
            target_mapping_mul : [B, num_mask, qlen_mul]
            target_mapping_uni : [B, N, num_mask, qlen_mul]
            target_masked_idx  : [B, N, num_mask, D]
        """
        device = inp_k.device
        P, N, D= self.configs.data.patch_num, self.configs.data.n_vars, self.d_model
        G = self.group_token_num if self.efficient else None
        B = inp_k.shape[0]
        
        qlen_mul = P*N
        qlen_uni = P

        if self.mem_len_mul==0: mlen_mul = 0
        else: mlen_mul = mems_mul[0].size(0) if mems_mul is not None else 0
        if self.mem_len_uni==0: mlen_uni = 0
        else: 
            # mlen_uni = 0
            # if (mems_uni is not None):
            #     print('mems_uni = ', mems_uni)
            #     if len(mems_uni)>0 and print(len(mems_uni[0]))>0:
            #         mlen_uni = mems_uni[0][0].size(2)
            mlen_uni = mems_uni[0][0].size(2) if (mems_uni is not None)and(len(mems_uni[0])>0) else 0
        klen_mul = qlen_mul + mlen_mul
        klen_uni = qlen_uni + mlen_uni

        K = int(klen_mul/N)

        ## number of masked tokens
        num_mask = target_mapping_mul.shape[1] if pretrain else None

        #### Attention mask ####
        """
        Returns of Function `attention_mask`:
            attn_mask_mul : [bsz, 1, qlen_mul, klen_mul]
            attn_mask_uni : [bsz, N, 1, P, K]
            non_tgt_mask_mul(not-efficient) : [bsz, 1, qlen_mul, klen_mul]
            non_tgt_mask_mul(efficient) : [P*G, K*G]
            non_tgt_mask_uni : [bsz, N, 1, P, K]
        """
        with torch.no_grad():
            if self.klen_mul != klen_mul:
                self.attn_mask_mul_cache = None
            if self.klen_uni != klen_uni:
                self.attn_mask_uni_cache = None
            attn_mask_mul, attn_mask_uni, non_tgt_mask_mul, non_tgt_mask_uni, attn_mask_mul_cache, attn_mask_uni_cache = \
                attention_mask(batch_size=B, num_vars=N,
                               qlen_mul=qlen_mul, qlen_uni=qlen_uni, mlen_mul=mlen_mul, mlen_uni=mlen_uni,
                               input_mask_mul=input_mask_mul, input_mask_uni=input_mask_uni, 
                               perm_mask_mul=perm_mask_mul, perm_mask_uni=perm_mask_uni, 
                               attn_mask_mul_cache=self.attn_mask_mul_cache, 
                               attn_mask_uni_cache=self.attn_mask_uni_cache,
                               n_cls=G, efficient=self.efficient, device=device,
                               attn_direction=self.attn_direction, same_length=self.same_length)
            if (self.attn_mask_mul_cache is None):
                self.attn_mask_mul_cache = attn_mask_mul_cache 
            if (self.attn_mask_uni_cache is None):
                self.attn_mask_uni_cache = attn_mask_uni_cache

            if pretrain:
                if self.efficient:
                    non_tgt_mask_mul = einops.repeat(non_tgt_mask_mul[None,None,:,:], '1 1 i j -> b 1 i j', b=B).to(device)
                else:
                    attn_mask_mul = attn_mask_mul.to(device)
                    non_tgt_mask_mul = non_tgt_mask_mul.to(device)
                attn_mask_uni = einops.rearrange(attn_mask_uni, 'b k n i j -> (b k) n i j').to(device)
                non_tgt_mask_uni = einops.rearrange(non_tgt_mask_uni, 'b k n i j-> (b k) n i j').to(device)

        #### Group token Embedding ####
        if self.efficient:
            self.group_cls = einops.repeat(self.cls, '1 1 1 d -> b l g d', b=B, l=P, g=G).view(B*P, G, D)

        #### Patches embedding ####
        """
        Args:
            inp_k_mul : [bsz*patch_num, n_vars, patch_len]
            inp_k_uni : [bsz, n_vars, patch_num, patch_len]
            emb_k_mul : [bsz*patch_num, n_vars, d_model]
            emb_k_uni : [bsz, n_vars, patch_num, d_model]
            output_h_mul : [n_vars + n_cls, PM*bsz, d_model]
            output_h_uni : [n_vars, patch_num, bsz, d_model]
        """
        inp_k_uni = inp_k_decomp.permute(0,2,1,3).contiguous().view(B, N, P, self.configs.data.patch_len)
        if self.efficient:
            inp_k_mul = inp_k.contiguous().view(B*P, N, self.configs.data.patch_len)
        else:
            inp_k_mul = inp_k.contiguous().view(B, P*N, self.configs.data.patch_len)

        #### Content Stream(Original Attention), h^(0)_t = e(x_i) ####
        emb_k_mul = self.projection_mul(inp_k_mul) # [bsz*patch_num, n_vars, d_model]
        emb_k_uni = self.projection_uni(inp_k_uni) # [bsz*n_vars, patch_num, d_model]

        zero_mem_pad4mul = None
        if self.efficient:
            with torch.no_grad():
                zero_mem_pad4mul = torch.zeros((B*(K-P), N+G, D)).cuda() # [bsz*(PM - P), n_vars + n_cls, d_model]
            emb_k_mul = torch.cat((emb_k_mul, self.group_cls), dim=1) # [bsz*P, n_vars + n_cls, d_model]
            emb_k_mul = torch.cat((zero_mem_pad4mul, emb_k_mul), dim=0) # [bsz*PM, n_vars + n_cls, d_model]

        output_h_mul = self.Dropout_h_mul(emb_k_mul) # [bsz*PM, n_vars + n_cls, d_model]
        output_h_uni = self.Dropout_h_mul(emb_k_uni) # [bsz, n_vars, patch_num, d_model]

        #### Query Stream, g^(0)_t = w
        #### the first layer query stream is initialized with a trainable vector
        """
            word_emb_q.shape : [bsz, num_mask, d_model]
            output_g_mul : [bsz, num_mask, d_model]
            output_g_uni : [bsz, n_vars, num_mask, d_model]
        """
        word_emb_q, output_g_mul, output_g_uni = None, None, None
        if pretrain:
            if target_mapping_mul is not None: ## target_mapping_mul : [bsz, num_mask, sqe_len]
                word_emb_q = einops.repeat(self.mask_emb, '1 1 d -> b m d', m=num_mask, b=B) # [bsz, num_mask, d_model]
            
            output_g_mul = self.Dropout_g_mul(word_emb_q) # [bsz, num_mask, d_model]
            output_g_uni = torch.einsum('bmh,bkm->bkmh', output_g_mul, target_masked_idx) # [bsz, n_vars, num_mask, d_model]
        
        #### Positional encoding ####
        """
            pos_emb_mul(efficient) : [bsz, PM*n_cls, d_model]
            pos_emb_mul(not-efficient) : [bsz, klen, d_model]
            pos_emb_uni : [bsz*N, klen_uni, d_model]
        """
        if self.efficient and (self.pos_emb is None):
            self.pos_emb = relative_positional_encoding(
                qlen=qlen_mul, klen=K, k_dim=N, d_model=D, clamp_len=self.clamp_len, attn_direction=self.attn_direction, 
                bi_data=False, h_number=None, batch_first=True, dtype=torch.float32) # [1, K, d_model]
        if (self.pos_emb_mul is None) or (self.klen_mul != klen_mul):
            self.pos_emb_mul = relative_positional_encoding(
                qlen=qlen_mul, klen=klen_mul, k_dim=N, d_model=self.d_model, clamp_len=self.clamp_len, attn_direction=self.attn_direction, 
                bi_data=False, h_number=P, batch_first=True, dtype=torch.float32) # [1, klen_mul, d_model]
        if (self.pos_emb_uni is None) or (self.klen_uni != klen_uni):
            self.pos_emb_uni = relative_positional_encoding(
                qlen=qlen_mul, klen=klen_uni, k_dim=N, d_model=D, clamp_len=self.clamp_len, attn_direction=self.attn_direction, 
                bi_data=False, h_number=None, batch_first=True, dtype=torch.float32) # [1, klen_uni, d_model]
        
        if self.efficient: # pos_emb_mul_h : [bsz, PM*n_cls, d_model]
            pos_emb_mul_h = self.Dropout_pos_mul(einops.repeat(self.pos_emb, '1 p d -> b (p g) d', b=B, g=G)).to(device) 
            pos_emb_mul_g = self.Dropout_pos_mul(einops.repeat(self.pos_emb_mul, '1 p d -> b p d', b=B)).to(device)
        else: # pos_emb_mul_h : [bsz, klen, d_model]
            pos_emb_mul_h = self.Dropout_pos_mul(einops.repeat(self.pos_emb_mul, '1 p d -> b p d', b=B)).to(device)
            pos_emb_mul_g = pos_emb_mul_h
        
        pos_emb_uni = self.Dropout_pos_uni(einops.repeat(self.pos_emb_uni, '1 p d -> b p d', b=B)).to(device)

        self.klen_uni = klen_uni
        self.klen_mul = klen_mul

        output_g_mul, output_h_mul, output_g_uni, output_h_uni, mem_lst, attn_prob_lst, attn_score_lst = \
            self.backbone(B, zero_mem_pad4mul, target_masked_idx,
                          output_h_mul, output_g_mul, output_h_uni, output_g_uni,
                          pos_emb_mul_h, pos_emb_mul_g, pos_emb_uni, pos_emb_uni,
                          sparse_attn_mask, sparse_attn_mem_mask,
                          memory_mul=mems_mul, memory_uni=mems_uni,
                          non_tgt_mask=non_tgt_mask_mul, non_tgt_mask_uni=non_tgt_mask_uni,
                          attn_mask=attn_mask_mul, attn_mask_uni=attn_mask_uni,
                          target_mapping=target_mapping_mul, target_mapping_uni=target_mapping_uni,
                          seg_id=seg_id, pretrain=pretrain, strategy=strategy)

        return output_g_mul, output_h_mul, output_g_uni, output_h_uni, mem_lst, attn_prob_lst, attn_score_lst

class JCCEncoder(nn.Module):
    def __init__(self, mem_len_mul:int=0, mem_len_uni:int=0, reuse_len_mul:int=0, reuse_len_uni:int=0,
                 configs:argparse.Namespace=None, mul_uni_ratio:int=1, group_token_num=10, 
                 efficient=False, res_attention=False, use_seg_emb=False):
        super(JCCEncoder, self).__init__()
        self.configs = configs
        self.n_layer_mul = configs.model.e_layers
        self.mem_len_mul = mem_len_mul
        self.mem_len_uni = mem_len_uni
        self.reuse_len_mul = reuse_len_mul
        self.reuse_len_uni = reuse_len_uni
        self.mul_uni_ratio = mul_uni_ratio
        self.efficient = efficient

        self.layers = nn.ModuleList([JCCEncoderLayer(mem_len_mul, mem_len_uni, reuse_len_mul, reuse_len_uni,
                                                     configs, mul_uni_ratio, group_token_num,
                                                     efficient, use_seg_emb, res_attention,
                                                     ) for _ in range(configs.model.e_layers)])
        self.res_attention = res_attention

    def forward(self, batch_szie:int,
                zero_mem_pad4mul:Tensor, target_masked_idx:Tensor,
                output_h_mul:Tensor, output_g_mul:Tensor, output_h_uni:Tensor, output_g_uni:Tensor,
                pos_emb_mul_h:Tensor, pos_emb_mul_g:Tensor, pos_emb_uni_h:Tensor, pos_emb_uni_g:Tensor,
                sparse_attn_mask:torch.Tensor=None, sparse_attn_mem_mask:torch.Tensor=None,
                memory_mul:Optional[list]=None, memory_uni:Optional[list]=None,
                non_tgt_mask:Optional[Tensor]=None, non_tgt_mask_uni:Optional[Tensor]=None,
                attn_mask:Optional[Tensor]=None, attn_mask_uni:Optional[Tensor]=None,
                target_mapping:Optional[Tensor]=None, target_mapping_uni:Optional[Tensor]=None,
                seg_id:Optional[Tensor]=None, 
                pretrain:bool=False, strategy:str='CICD'):

        P, N, D= self.configs.data.patch_num, self.configs.data.n_vars, self.configs.model.d_model
        B = batch_szie
        
        new_mems_mul = []
        new_mems_uni = []
        attn_prob_lst_M = []
        attn_prob_lst_U = []
        attn_score_lst_M = []
        attn_score_lst_U = []
        prev_attn_mul = None
        prev_attn_uni = None

        with torch.no_grad():
            if memory_mul is None:
                memory_mul = [None] * self.n_layer_mul
            if memory_uni is None:
                memory_uni = [[None]*self.mul_uni_ratio ]*self.n_layer_mul
        for i in range(self.n_layer_mul):
            # if i==0:
            #     if self.mem_len_mul > 0:
            #         new_mems_mul.append(JCCEncoderLayer._cache_mem(output_h_mul, memory_mul[i], self.mem_len_mul, self.reuse_len_mul, attn_module='Mul'))
            #     else: new_mems_mul.append(None)
            #     if self.mem_len_uni > 0:
            #         print("=======3=======")
            #         new_mems_mul_temp = []
            #         for k in range(int(self.mul_uni_ratio)):
            #             new_mems_mul_temp.append(JCCEncoderLayer._cache_mem(output_h_uni, memory_uni[i][k], self.mem_len_uni, self.reuse_len_uni, attn_module='Uni'))
            #         print('new_mems_mul_temp: ',new_mems_mul_temp)
            #         new_mems_uni.append(new_mems_mul_temp)
            # else:
                
            output_g_mul, output_h_mul, output_g_uni, output_h_uni, new_mem_mul, new_mem_uni, \
                attn_prob_lst_mul, attn_prob_lst_uni, attn_score_lst_mul, attn_score_lst_uni, prev_mul, prev_uni = \
                    self.layers[i](batch_szie, zero_mem_pad4mul, target_masked_idx,
                          output_h_mul, output_g_mul, output_h_uni, output_g_uni,
                          pos_emb_mul_h, pos_emb_mul_g, pos_emb_uni_h, pos_emb_uni_g,
                          sparse_attn_mask, sparse_attn_mem_mask,
                          memory_mul[i], memory_uni[i], 
                          non_tgt_mask, non_tgt_mask_uni, 
                          attn_mask, attn_mask_uni,
                          target_mapping, target_mapping_uni,
                          seg_id, prev_mul=prev_attn_mul, prev_uni=prev_attn_uni,
                          pretrain=pretrain, strategy=strategy)
            if self.mem_len_mul > 0: new_mems_mul.append(new_mem_mul)
            else: new_mems_mul.append(None)
            if self.mem_len_uni > 0: new_mems_uni.append(new_mem_uni)
            else: new_mems_uni.append([None]*self.mul_uni_ratio)
            if 'CD' in strategy:
                attn_prob_lst_M.append(attn_prob_lst_mul)
                attn_score_lst_M.append(attn_score_lst_mul)
            if 'CI' in strategy:
                attn_prob_lst_U.append(attn_prob_lst_uni)
                attn_score_lst_U.append(attn_score_lst_uni)

            prev_attn_mul = prev_mul
            prev_attn_uni = prev_uni

        if self.efficient:
            output_h_mul = einops.rearrange(output_h_mul, '(b p) k d -> b p k d', b=B)[:,-P:,:N] # output_h_mul : [B, P, N, D]
        else:
            output_h_mul = output_h_mul.view(B, P, N, D)
            
        return output_g_mul, output_h_mul, output_g_uni, output_h_uni, [new_mems_mul, new_mems_uni], \
    [attn_prob_lst_M, attn_prob_lst_U],[attn_score_lst_M, attn_score_lst_U]
    
class JCCEncoderLayer(nn.Module):
    def __init__(self, mem_len_mul:int=0, mem_len_uni:int=0, reuse_len_mul:int=0, reuse_len_uni:int=0,
                 configs:argparse.Namespace=None, mul_uni_ratio:int=1, group_token_num=10, 
                 efficient=False, res_attention=False, use_seg_emb=False):
        super(JCCEncoderLayer, self).__init__()
        self.configs = configs
        self.mem_len_mul = mem_len_mul
        self.mem_len_uni = mem_len_uni
        self.reuse_len_mul = reuse_len_mul
        self.reuse_len_uni = reuse_len_uni
        self.mul_uni_ratio = mul_uni_ratio
        self.group_token_num = group_token_num
        self.efficient = efficient
        self.use_seg_emb = use_seg_emb

        # learnable vector `u` in relative position encoding
        self.r_w_bias_mul = nn.Parameter(nn.init.xavier_normal_(
            torch.randn(configs.model.n_heads, configs.model.d_head))) 
        self.r_w_bias_uni = nn.Parameter(nn.init.xavier_normal_(
            torch.randn(int(mul_uni_ratio), configs.model.n_heads, configs.model.d_head)))
        # learnable vector `v` in relative position encoding
        self.r_r_bias_mul = nn.Parameter(nn.init.xavier_normal_(
            torch.randn(configs.model.n_heads, configs.model.d_head)))
        self.r_r_bias_uni = nn.Parameter(nn.init.xavier_normal_(
            torch.randn(int(mul_uni_ratio), configs.model.n_heads, configs.model.d_head)))

        if use_seg_emb:
            ##### Segment embedding
            self.r_s_bias_mul = nn.Parameter(nn.init.xavier_normal_(
                torch.randn(configs.model.n_heads, configs.model.d_head)))
            self.r_s_bias_uni = nn.Parameter(nn.init.xavier_normal_(
                torch.randn(int(mul_uni_ratio), configs.model.n_heads, configs.model.d_head)))
            self.seg_embed_mul = nn.Parameter(nn.init.xavier_normal_(
                torch.randn(2, configs.model.n_heads, configs.model.d_head)))
            self.seg_embed_uni = nn.Parameter(nn.init.xavier_normal_(
                torch.randn(int(mul_uni_ratio), 2, configs.model.n_heads, configs.model.d_head)))

        self.Multi_module = MultiheadRelAttention(configs=self.configs, attn_type='Mul', 
                                                  proj_drop=0.1, out_drop=configs.model.dropout, 
                                                  attn_drop=configs.model.dropout_att, 
                                                  norm_h='layer_norm', norm_shape_h=configs.model.d_model, 
                                                  norm_g='layer_norm', norm_shape_g=configs.model.d_model, 
                                                  cos_attn=True, res_attention=True, efficient=efficient)
        self.PW_FFN_mul = positionwise_ffn(d_model=configs.model.d_model, d_ff=configs.model.d_ff, dropout=configs.model.dropout,
                                           norm_h='layer_norm', norm_shape_h=configs.model.d_model, 
                                           norm_g='layer_norm', norm_shape_g=configs.model.d_model, 
                                           activation_type='gelu')
        for j in range(int(mul_uni_ratio)):
            self.add_module('TS_Attn_uni_' + str(j), MultiheadRelAttention(configs=self.configs, attn_type='Uni', 
                                                                           proj_drop=0.1, out_drop=configs.model.dropout, 
                                                                           attn_drop=configs.model.dropout_att,
                                                                           norm_h='layer_norm', norm_shape_h=configs.model.d_model,
                                                                           norm_g='layer_norm', norm_shape_g=configs.model.d_model,
                                                                           cos_attn=False, res_attention=True))
            self.add_module('PW_FFN_uni_' + str(j), positionwise_ffn(d_model=configs.model.d_model, d_ff=configs.model.d_ff, dropout=configs.model.dropout, 
                                                                     norm_h='layer_norm', norm_shape_h=configs.model.d_model, 
                                                                     norm_g='layer_norm', norm_shape_g=configs.model.d_model, 
                                                                     activation_type='gelu'))

        self.Uni_modules = AttrProxy(self, 'TS_Attn_uni_')
        self.PW_FFN_uni = AttrProxy(self, 'PW_FFN_uni_')

        self.Norm = nn.LayerNorm(normalized_shape=configs.model.d_model)

    def _cache_mem(self, curr_out, prev_mem, mem_len, reuse_len=None, attn_module=None, efficient=False):
        """
        Cache hidden states into memory.
        cached memory does not participate in gradient calculation.
        Args:
            Mul attn_module:
                curr_out(not-efficient) : [bsz, TSaS_len, d_model]
                curr_out(efficient) : [bsz*PM, n_vars+n_cls, d_model]
                prev_mem : [bsz, TSaS_len, d_model]
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
    
    def forward(self, batch_szie:int, zero_mem_pad4mul:Tensor, target_masked_idx:Tensor,
                output_h_mul:Tensor, output_g_mul:Tensor, output_h_uni:Tensor, output_g_uni:Tensor,
                pos_emb_mul_h:Tensor, pos_emb_mul_g:Tensor, pos_emb_uni_h:Tensor, pos_emb_uni_g:Tensor,
                sparse_attn_mask:Optional[Tensor]=None, sparse_attn_mask_mem:Optional[Tensor]=None,
                memory_mul:Optional[Tensor]=None, memory_uni:Optional[list]=None,
                non_tgt_mask:Optional[Tensor]=None, non_tgt_mask_uni:Optional[Tensor]=None,
                attn_mask:Optional[Tensor]=None, attn_mask_uni:Optional[Tensor]=None,
                target_mapping:Optional[Tensor]=None, target_mapping_uni:Optional[Tensor]=None,
                seg_id:Optional[Tensor]=None, 
                prev_mul=None, prev_uni=None,
                pretrain=False, strategy='CICD'):

        """
        sparse_attn_mask : [1, n_head, qlen, qlen]
        sparse_attn_mask_mem : [1, n_head, qlen, mlen]
        """
        P, N, D= self.configs.data.patch_num, self.configs.data.n_vars, self.configs.model.d_model
        G = self.group_token_num if self.efficient else None
        B = batch_szie

        new_mems_uni = []

        # cache new mems
        if self.mem_len_mul > 0:
            new_mem_mul = self._cache_mem(output_h_mul, memory_mul, self.mem_len_mul, self.reuse_len_mul, attn_module='Mul')
        else: new_mem_mul = None

        if memory_mul is None:
            sparse_attn_mask_mem = None

        # if memory_mul is not None:
        #     sparse_attn_mem = sparse_attn_mask_mem
        # else: sparse_attn_mem = None

        # segment bias
        r_s_bias_mul = None
        r_s_bias_uni = None
        seg_embed_mul = None
        seg_embed_uni = None

        attn_prob_lst_mul = None
        attn_score_lst_mul = None
        attn_prob_lst_uni = None
        attn_score_lst_uni = None

        if self.use_seg_emb and (seg_id is not None):
            r_s_bias_mul = self.r_s_bias_mul
            seg_embed_mul = self.seg_embed_mul
        ## Multi_module
        if 'CD' in strategy:
            output_h_mul, output_g_mul, attn_prob_lst_mul, attn_score_lst_mul = \
                self.Multi_module(
                    h=output_h_mul, g=output_g_mul, r_h=pos_emb_mul_h, r_g=pos_emb_mul_g, mems=memory_mul,
                    r_w_bias=self.r_w_bias_mul, r_r_bias=self.r_r_bias_mul, r_s_bias=r_s_bias_mul, 
                    seg_mat=None, seg_embed=seg_embed_mul, 
                    attn_mask_h=non_tgt_mask, attn_mask_g=attn_mask, 
                    target_mapping=target_mapping,
                    sparse_attn=sparse_attn_mask, sparse_attn_mem=sparse_attn_mask_mem, 
                    pretrain=pretrain, prev=prev_mul)
            prev_mul = attn_score_lst_mul

            if pretrain:
                output_g_mul = self.PW_FFN_mul(inp=output_g_mul, stream='g')
            output_h_mul = self.PW_FFN_mul(inp=output_h_mul, stream='h')
        if 'CI' in strategy:
            for k in range(self.mul_uni_ratio):
                ## cache new mems
                if self.mem_len_uni > 0:
                    new_mems_uni.append(self._cache_mem(output_h_uni, memory_uni[k], self.mem_len_uni, self.reuse_len_uni, attn_module='Uni'))
                else: new_mems_uni.append(None)

                if self.use_seg_emb and (seg_id is not None):
                    r_s_bias_uni = self.r_s_bias_uni[k]
                    seg_embed_uni = self.seg_embed_uni[k]
                output_h_uni, output_g_uni, attn_prob_lst_uni, attn_score_lst_uni = \
                    self.Uni_modules[k](
                    h=output_h_uni, g=output_g_uni, r_h=pos_emb_uni_h, r_g=pos_emb_uni_g, mems=memory_uni[k],
                    r_w_bias=self.r_w_bias_uni[k], r_r_bias=self.r_r_bias_uni[k], r_s_bias=r_s_bias_uni, 
                    seg_mat=None, seg_embed=seg_embed_uni,
                    attn_mask_h=non_tgt_mask_uni, attn_mask_g=attn_mask_uni, 
                    target_mapping=target_mapping_uni,
                    sparse_attn=None, sparse_attn_mem=None,
                    pretrain=pretrain, prev=prev_uni)
                prev_uni = attn_score_lst_uni

                PosFFN_uni = self.PW_FFN_uni[k]
                if pretrain:
                    output_g_uni = PosFFN_uni(inp=output_g_uni, stream='g')
                output_h_uni = PosFFN_uni(inp=output_h_uni, stream='h')

        # if 'CD' in strategy:
        #     if pretrain:
        #         output_g_mul = self.PW_FFN_mul(inp=output_g_mul, stream='g')
        #     output_h_mul = self.PW_FFN_mul(inp=output_h_mul, stream='h')
        
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
                output_h_mul = self.Norm(output_h_mul)
                output_h_mul = torch.cat((zero_mem_pad4mul , output_h_mul), dim=0) # output_h_mul : [B*PM, N+G, D]
            else:
                output_h_uni2mul = einops.rearrange(output_h_uni, 'b k i d -> b (i k) d') # output_h_uni2mul : [B, P*N, D]
                output_h_mul += output_h_uni2mul
                output_h_mul = self.Norm(output_h_mul)
            if pretrain:
                """
                output_g_mul : [bsz, num_mask, d_model]
                output_g_uni : [bsz, n_vars, num_mask, d_model]
                """
                # output_g_uni2mul = torch.einsum('kmbh,kmbh->mbh', output_g_uni, target_masked_idx)
                output_g_uni2mul = torch.einsum('bkmh,bkm->bmh', output_g_uni, target_masked_idx)
                output_g_mul += output_g_uni2mul
                output_g_mul = self.Norm(output_g_mul)

        # if 'CD' not in strategy:
        elif strategy == 'CI':
            if pretrain:
                output_g_mul = torch.einsum('bkmh,bkm->bmh', output_g_uni, target_masked_idx)
                output_g_mul = self.Norm(output_g_mul)
            output_h_mul = einops.rearrange(output_h_uni, 'b k i d -> b (i k) d') # output_h_uni2mul : [B, P*N, D]
            output_h_mul = self.Norm(output_h_mul)

        return output_g_mul, output_h_mul, output_g_uni, output_h_uni, new_mem_mul, new_mems_uni, \
    attn_prob_lst_mul, attn_prob_lst_uni, attn_score_lst_mul, attn_score_lst_uni, prev_mul, prev_uni

