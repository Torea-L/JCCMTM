import einops
import torch.nn as nn
from torch import Tensor

from layers.basics import series_decomp
from data_provider.data_utils import creat_patch
from models.JCCMTM import data_revin, JCC_backbone, JCCEncoder


class Pretrain_Head(nn.Module):
    """Decoder layer for pretraining"""

    def __init__(self, d_model, d_out, head_dropout=0.1, activation_type='leaky_relu', details=False):
        super(Pretrain_Head, self).__init__()
        self.out_proj = nn.Linear(d_model, d_out, bias=True)
        self.dropout = nn.Dropout(head_dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if details:
                    print('In Decoder, Linear layer: ', m)
                if activation_type == 'relu':
                    nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, output_g, output_h):
        """
        output_h : [bsz, patch_num, n_vars, d_model]
        output_g : [bsz, num_mask, d_model]
        """
        output_h = self.dropout(self.out_proj(output_h))
        output_g = self.dropout(self.out_proj(output_g))

        return output_g, output_h
    
class Pretrain_Model(nn.Module):
    def __init__(self, attn_direction, clamp_len, same_length, 
                 reuse_len:int=0, reuse_len_uni:int=0, mem_len:int=0, mem_len_uni:int=0, 
                 mul_uni_ratio:int=1, kernel_size:int=7, group_token_num:int=5,
                 sparse_attn:Tensor=None, sparse_attn_mem:Tensor=None, configs=None, 
                 efficient=False, details=False, strategy='CICD'):
        super(Pretrain_Model, self).__init__()
        self.configs = configs
        self.strategy = strategy
        seq_len = configs.data.patch_num*configs.data.patch_len
        self.transformer = JCC_backbone(configs=configs,
                                              attn_direction=attn_direction, 
                                              clamp_len=clamp_len, same_length=same_length, 
                                              reuse_len=reuse_len, reuse_len_uni=reuse_len_uni,
                                              mem_len=mem_len, mem_len_uni=mem_len_uni,
                                              sparse_attn_=sparse_attn, sparse_attn_mem_=sparse_attn_mem, 
                                              mul_uni_ratio=mul_uni_ratio, group_token_num=group_token_num,
                                              efficient=efficient)
        self.linear_model_uni = None
        self.encoder = JCCEncoder(transformer=self.transformer, 
                                   linear=self.linear_model_uni, 
                                   configs=configs, kernel_size=kernel_size)
        self.decoder = Pretrain_Head(d_model=configs.model.d_model, d_out=configs.data.patch_len, 
                                     head_dropout=configs.model.head_dropout, details=details)
        
    def forward(self, inp_k:Tensor, inp_decomp:Tensor, seg_id:Tensor=None, 
                input_mask:Tensor=None, input_mask_uni:Tensor=None, 
                mems:list=None, mems_uni:list=None,
                perm_mask:Tensor=None, perm_mask_uni:Tensor=None,
                target_mapping:Tensor=None, target_mapping_uni:Tensor=None, 
                target_masked_idx:Tensor=None, 
                pretrain:bool=False):
        B, plen, patch_len = inp_k.shape
        self.configs.data.bsz_efficient = inp_k.shape[0]
        P, N = self.configs.data.patch_num, self.configs.data.n_vars
        inp_k = inp_k.view(B, P, N, patch_len)
        if inp_decomp is not None:
            inp_decomp = inp_decomp.view(B, P, N, patch_len)
        # Encoder
        output_g, output_h, output_g_uni, output_h_uni, mem_lst, attn_prob_lst, attn_score_lst = \
            self.encoder(inp_k, inp_decomp, seg_id, input_mask, input_mask_uni, 
                         mems, mems_uni, perm_mask, perm_mask_uni,
                         target_mapping, target_mapping_uni, 
                         target_masked_idx, pretrain, strategy=self.strategy)
        # Decoder
        output_g, output_h = self.decoder(output_g, output_h)
        # De-Normalization
        output_h = data_revin(output_h, revin=self.encoder.revin, mode='denorm', 
                              stream='h')
        output_g = data_revin(output_g, revin=self.encoder.revin, mode='denorm', 
                              stream='g', target_masked_idx=target_masked_idx)
        output_h = output_h.reshape(B, plen, patch_len)
            
        return output_g, output_h, mem_lst

class Prediction_Head(nn.Module):
    """Prediction layer for long-term MTS forecasting"""

    def __init__(self, configs, pred_len, head_dropout=0.1):
        super(Prediction_Head, self).__init__()
        self.num_patch = configs.data.patch_num
        # self.n_vars = configs.data.n_vars
        self.d_model = configs.model.d_model

        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(self.num_patch*self.d_model, pred_len)
        self.dropout = nn.Dropout(p=head_dropout)

    def forward(self, x:Tensor):
        """
        Input shape:
            x : [bsz, n_vars, d_model, patch_num] 
        Return:
            x : [bsz, n_vars, pred_len] 
        """
        x = self.flatten(x)
        x = self.dropout(self.linear(x))

        return x
    
class Prediction_Model(nn.Module):
    def __init__(self, pred_len:int, dropout:float, encoder:nn.Module, decomposition=True, strategy='CICD'):
        super(Prediction_Model, self).__init__()
        # encoder.efficient = efficient
        self.configs = encoder.configs
        self.decomposition = decomposition
        self.strategy = strategy

        if self.decomposition:
            self.decomp_module = series_decomp(self.configs.model.kernel_size)
        self.encoder = encoder
        self.encoder.pos_emb_mul = None
        self.encoder.pos_emb_uni = None
        self.encoder.pos_emb = None
        self.encoder.attn_mask_cahce = None
        self.encoder.attn_mask_uni_cahce = None
        self.encoder.klen_uni = self.configs.data.patch_num
        self.encoder.klen = self.configs.data.patch_num*self.configs.data.n_vars

        self.pred_head = Prediction_Head(self.configs, pred_len, dropout)

    def forward(self, x:Tensor, x_decomp:Tensor, return_att=False):
        # Decomposition
        """
        x :             [Batch, Input length, Channel]
        res_init :      [Batch, Channel, Input length]
        trend_init :    [Batch, Channel, Input length]
        """
        N = x.shape[-1]
        self.configs.data.bsz_efficient = x.shape[0]
        '''if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)'''
        # Patching
        """
        inp_k :         [Batch, Channel, patch_num, patch_len]
        inp_decomp :    [Batch, Channel, patch_num, patch_len] if not None
        """
        
        patch_len = self.configs.data.patch_len
        stride = self.configs.data.stride
        
        x = x.permute(0, 2, 1)
        inp_k, patch_num = creat_patch(x, patch_len, stride)

        if self.decomposition:
            x_decomp = x_decomp.permute(0, 2, 1)
            inp_decomp, _ = creat_patch(x_decomp, patch_len, stride)
        else:
            inp_decomp = None
        # Encoder
        """
        inp_k :         [Batch, patch_num, Channel, patch_len]
        inp_decomp :    [Batch, patch_num, Channel, patch_len] if not None
        """
        inp_k = inp_k.permute(0, 2, 1, 3)
        if inp_decomp is not None:
            inp_decomp = inp_decomp.permute(0, 2, 1, 3)

        _, output_h, _, _, _, attn_prob_lst, attn_score_lst = \
            self.encoder(inp_k, inp_decomp, pretrain=False, strategy=self.strategy)
        # Decoder
        output_h = einops.rearrange(output_h, 'b p n d -> b n d p', n=N)
        dec_out = self.pred_head(output_h) # [bsz, n_vars, pred_len]

        # De-Normalization
        dec_out = dec_out.permute(0, 2, 1)
        dec_out = self.encoder.revin(dec_out, 'denorm')
        # dec_out = dec_out.permute(0, 2, 1)

        if return_att:
            return dec_out, attn_prob_lst, attn_score_lst
        else:
            return dec_out

class Anomaly_Detection_Head(nn.Module):
    """Decoder module for MTS Anomaly Detection"""
    def __init__(self, configs, head_dropout=0.1):
        super(Anomaly_Detection_Head, self).__init__()
        self.num_patch = configs.data.patch_num
        # self.n_vars = configs.data.n_vars
        self.d_model = configs.model.d_model

        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(self.num_patch*self.d_model, configs.data.context_points)
        self.dropout = nn.Dropout(p=head_dropout)

    def forward(self, x:Tensor):
        """
        Input shape:
            x : [bsz, n_vars, d_model, patch_num] 
        Return:
            x : [bsz, n_vars, pred_len] 
        """
        x = self.flatten(x)
        x = self.dropout(self.linear(x))

        return x

class Anomaly_Detection_Model(nn.Module):
    def __init__(self, dropout:float, encoder:nn.Module, decomposition=True, strategy='CICD'):
        super(Anomaly_Detection_Model, self).__init__()
        self.configs = encoder.configs
        self.decomposition = decomposition
        self.strategy = strategy

        if self.decomposition:
            self.decomp_module = series_decomp(self.configs.model.kernel_size)
        self.encoder = encoder
        self.encoder.pos_emb_mul = None
        self.encoder.pos_emb_uni = None
        self.encoder.pos_emb = None
        self.encoder.attn_mask_cahce = None
        self.encoder.attn_mask_uni_cahce = None
        self.encoder.klen_uni = self.configs.data.patch_num
        self.encoder.klen = self.configs.data.patch_num*self.configs.data.n_vars

        self.recons_head = Anomaly_Detection_Head(self.configs, dropout)

    def forward(self, x:Tensor, x_decomp:Tensor):
        # Decomposition
        """
        x :             [Batch, Input length, Channel]
        res_init :      [Batch, Channel, Input length]
        trend_init :    [Batch, Channel, Input length]
        """
        N = x.shape[-1]
        self.configs.data.bsz_efficient = x.shape[0]
        # Patching
        """
        inp_k :         [Batch, Channel, patch_num, patch_len]
        inp_decomp :    [Batch, Channel, patch_num, patch_len] if not None
        """
        patch_len = self.configs.data.patch_len
        stride = self.configs.data.stride
        
        x = x.permute(0, 2, 1)
        inp_k, patch_num = creat_patch(x, patch_len, stride)

        if self.decomposition:
            x_decomp = x_decomp.permute(0, 2, 1)
            inp_decomp, _ = creat_patch(x_decomp, patch_len, stride)
        else:
            inp_decomp = None

        # Encoder
        """
        inp_k :         [Batch, patch_num, Channel, patch_len]
        inp_decomp :    [Batch, patch_num, Channel, patch_len] if not None
        """
        inp_k = inp_k.permute(0, 2, 1, 3)
        if inp_decomp is not None:
            inp_decomp = inp_decomp.permute(0, 2, 1, 3)

        _, output_h, _, _, _, _, _ = \
            self.encoder(inp_k, inp_decomp, pretrain=False, strategy=self.strategy)
        # Reconstructor
        output_h = einops.rearrange(output_h, 'b p n d -> b n d p', n=N)
        dec_out = self.recons_head(output_h) # [bsz, n_vars, pred_len]

        # De-Normalization
        dec_out = dec_out.permute(0, 2, 1)
        dec_out = self.encoder.revin(dec_out, 'denorm')
        # dec_out = dec_out.permute(0, 2, 1)

        return dec_out