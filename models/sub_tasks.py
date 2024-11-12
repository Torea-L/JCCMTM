import einops
import torch.nn as nn
from torch import Tensor
import argparse

from layers.basics import series_decomp
from data_provider.data_utils import creat_patch
from models.JCCMTM import data_revin, Model
from models.JCC_backbone import JCC_backbone


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
    def __init__(self, configs:argparse.Namespace=None, attn_direction='uni', clamp_len=-1, 
                 same_length:bool=False, use_seg_emb:bool=False, learn_pe:bool=True, 
                 reuse_len_mul:int=0, reuse_len_uni:int=0, mem_len_mul:int=0, mem_len_uni:int=0, 
                 mul_uni_ratio:int=1, group_token_num:int=5,
                 efficient=False, res_attention:bool=True, verbose:bool=False,
                 details=False,):
        super(Pretrain_Model, self).__init__()
        self.configs = configs
        self.JCC = JCC_backbone(configs, attn_direction, clamp_len, 
                                same_length, use_seg_emb, learn_pe,
                                reuse_len_mul, reuse_len_uni,
                                mem_len_mul, mem_len_uni, 
                                mul_uni_ratio, group_token_num,
                                efficient=efficient, res_attention=res_attention, verbose=verbose)
        self.linear_model_uni = None
        self.encoder = Model(configs=configs, encoder=self.JCC)
        self.decoder = Pretrain_Head(d_model=configs.model.d_model, d_out=configs.data.patch_len, 
                                     head_dropout=configs.model.head_dropout, details=details)
        
    def forward(self, inp_k:Tensor, inp_decomp:Tensor, seg_id:Tensor=None, 
                sparse_attn_mask:Tensor=None, sparse_attn_mem_mask:Tensor=None,
                input_mask_mul:Tensor=None, input_mask_uni:Tensor=None, 
                mems_mul:list=None, mems_uni:list=None,
                perm_mask_mul:Tensor=None, perm_mask_uni:Tensor=None,
                target_mapping_mul:Tensor=None, target_mapping_uni:Tensor=None, 
                target_masked_idx:Tensor=None, 
                pretrain:bool=False, strategy='CICD'):
        B, plen, patch_len = inp_k.shape
        self.configs.data.bsz_efficient = inp_k.shape[0]
        P, N = self.configs.data.patch_num, self.configs.data.n_vars
        inp_k = inp_k.view(B, P, N, patch_len)
        if inp_decomp is not None:
            inp_decomp = inp_decomp.view(B, P, N, patch_len)
        # Encoder
        output_g, output_h, output_g_uni, output_h_uni, mem_lst, attn_prob_lst, attn_score_lst = \
            self.encoder(inp_k, inp_decomp, seg_id, 
                         sparse_attn_mask, sparse_attn_mem_mask, mems_mul, mems_uni, 
                         input_mask_mul, input_mask_uni, perm_mask_mul, perm_mask_uni,
                         target_mapping_mul, target_mapping_uni, 
                         target_masked_idx, pretrain, strategy=strategy)
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
    def __init__(self, pred_len:int, dropout:float, encoder:nn.Module, decomposition=True, component='trend', strategy='CICD'):
        super(Prediction_Model, self).__init__()
        self.configs = encoder.configs
        self.decomposition = decomposition
        self.component = component
        self.strategy = strategy

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

        if return_att:
            return dec_out, attn_prob_lst, attn_score_lst
        else:
            return dec_out

class Anomaly_Detection_Head(nn.Module):
    """Decoder module for MTS Anomaly Detection"""
    def __init__(self, configs, head_dropout=0.1):
        super(Anomaly_Detection_Head, self).__init__()
        self.num_patch = configs.data.patch_num
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
    def __init__(self, dropout:float, encoder:nn.Module, decomposition=True, component='trend',strategy='CICD'):
        super(Anomaly_Detection_Model, self).__init__()
        self.configs = encoder.configs
        self.decomposition = decomposition
        self.component = component
        self.strategy = strategy
        
        self.encoder = encoder
        self.encoder.pos_emb_mul = None
        self.encoder.pos_emb_uni = None
        self.encoder.pos_emb = None
        self.encoder.attn_mask_cahce = None
        self.encoder.attn_mask_uni_cahce = None
        self.encoder.klen_uni = self.configs.data.patch_num
        self.encoder.klen = self.configs.data.patch_num*self.configs.data.n_vars

        self.recons_head = Anomaly_Detection_Head(self.configs, dropout)

    def forward(self, x:Tensor, x_decomp:Tensor, return_att=False):
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


