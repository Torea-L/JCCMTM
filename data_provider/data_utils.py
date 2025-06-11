
__all__ = ['generate_subseq', 'creat_patch', 'create_pred_target',
           'create_data_mask', 'make_feature']

import random
import warnings
from typing import Optional

import numpy as np
import torch

warnings.filterwarnings("ignore")

def generate_subseq(data: torch.Tensor, 
                    look_back_len: int, 
                    stride: int):
    """
    data: [-1, n_vars]
    sub_seq_len: length of a sub-sequence
    """
    data_len = data.shape[0]
    num_ = (data_len - look_back_len)// stride + 1
    inp_seqs = data.unfold(dimension=0, size=look_back_len, step=stride)    # sub_seqs: [num_, nvars, look_back_len]
    return inp_seqs, num_
    
def creat_patch(inp_seq: torch.Tensor, 
                patch_len: int, 
                stride: int):
    """
    inp_seq: [num_, n_vars, look_back_len]
    """
    
    look_back_len = inp_seq.shape[2]
    num_patch = (max(look_back_len, patch_len)-patch_len) // stride + 1
    tgt_len = patch_len  + stride*(num_patch-1)
    s_begin = look_back_len - tgt_len
    # print('In `creat_patch`, s_begin : ', s_begin)
    
    xb = inp_seq[:, :, s_begin:]                                  # xb: [num_, nvars, tgt_len]
    xb = xb.unfold(dimension=-1, size=patch_len, step=stride)      # xb: [num_, nvars, num_patch, patch_len]
    assert (num_patch==xb.shape[2])
    return xb, num_patch

def create_pred_target(data: torch.Tensor, 
                       look_back_len: int, 
                       pred_len: int, 
                       stride: int):
    """
    data: [-1, nvars]
    """
    sub_seqs_pred = data[look_back_len:,:]
    num_target = (sub_seqs_pred.shape[0] - pred_len)// stride + 1
    pred_target = sub_seqs_pred.unfold(dimension=0, size=pred_len, step=stride)  # pred_target: [num_target, n_vars, pred_len]
    print('In "create_pred_target", pred_target.shape : ', pred_target.shape)
    return pred_target, num_target

def _sample_mask(seg, data_dimensions, h_number,
                 reverse=False, goal_num_predict=None, mask_num=2):
    """
    For pre-train step 1.
    Sample  tokens for partial prediction.
    
    Args:
        seg: segment that consists of several tokens
        data_dimensions: dimensions of features
        goal_num_predict: number of tokens to predict in seg
        mask_num : Number of samples covered by potential MASK token positions
    
    Example: (h_number=10; data_dimensions = 10; random.random() < 0.5; n=[36, 47, 30, 40, 45])
    mask = array([False, False, False, False, False, False, False, False, False, False, 
                  False, False, False, False, False, False, False, False, False, False, 
                  False, False, False, False, False, False, False, False, False, False, 
                  True,  False, False, False, False, False, True,  False, False, False, 
                  True,  False, False, False, False, True,  False, True,  False, False])
    """
    example_num  = int(h_number/2)
    mask = np.array([False] * seg.shape[0], dtype=np.bool_)
    if reverse: # Flip seg in the time dim.
        seg = np.flip(seg, 0)
        
    ngrams = np.linspace(data_dimensions*(example_num-mask_num),
                         data_dimensions*example_num-1, 
                         data_dimensions*mask_num)

    #if goal_num_predict is not None and num_predict >= goal_num_predict: break
    if random.random() < 0.7:
        n = np.random.choice(ngrams, goal_num_predict, replace = False)
    else:
        n_s = np.random.choice(ngrams[0:len(ngrams)-goal_num_predict])
        n = np.linspace(n_s,n_s+goal_num_predict-1,goal_num_predict)
    for n_ in n:
        mask[int(n_)] = ~mask[int(n_)]
            
    if reverse:
        mask = np.flip(mask, 0)
    return mask

def _sample_mask_all_(seg_len:int, 
                      mask_alpha: int = 8, 
                      mask_beta: int = 1, 
                      start_: int = 0, 
                      kpi_num: int = None, 
                      reverse: bool = None, 
                      goal_num_predict:int = None):
    
    mask = np.array([False] * seg_len, dtype=np.bool_)
    ngrams = np.arange(kpi_num, dtype=np.int64)
    pvals = np.array([1./kpi_num]*kpi_num)
    pvals[0] += 0.06
    pvals[1] += 0.06
    pvals /= pvals.sum(keepdims=True)
    
    num_predict = 0
    cur_len = start_
    n = np.random.choice(ngrams, p=pvals)
    cur_len += n
    while cur_len < seg_len:
        if goal_num_predict is not None and num_predict >= goal_num_predict: break
        
        beg = cur_len
        if beg >= seg_len:
            break        
        
        mask[beg] = True
        num_predict += 1
        
        ctx_size = mask_alpha // mask_beta # context size
        cur_len = beg + ctx_size
    
    while goal_num_predict is not None and num_predict < goal_num_predict:
        i = np.random.randint(low=start_,high=seg_len)  # [low, high)
        if not mask[i]:
            mask[i] = True
            num_predict += 1
    
    if reverse:
        mask = np.flip(mask, 0)
        
    return mask
    
def create_data_mask(tokens_data: np.ndarray, 
                     num_predict: int, 
                     params, 
                     tokens_decomp:np.ndarray=None,
                     data_posID:Optional[np.ndarray]=None, 
                     mask_sample: str='all'):
    '''
    Input shape:
        tokens_data: Numpy array, data of tokens, shape = [num_subseq, nvars, num_patch, patch_len]
        num_predict: Int, number of tokens to predict.

    seq_len: int, number of tokens(patches) contained in one subsequence.    
    '''
    # if params.decomposition:
    #     print('tokens_decomp.shape = ', tokens_decomp.shape)
    # print('tokens_data.shape = ', tokens_data.shape)
    [num_subseq, nvars, num_patch, patch_len] = tokens_data.shape
    seq_len = nvars*num_patch

    sub_seq_data = tokens_data.transpose(0,2,1,3).reshape(num_subseq, seq_len, patch_len)
    if params.decomposition:
        print('tokens_decomp.shape = ', tokens_decomp.shape)
        print('tokens_data.shape = ', tokens_data.shape)
        sub_seq_decomp = tokens_decomp.transpose(0,2,1,3).reshape(num_subseq, seq_len, patch_len)
    
    features = []
    print('In _create_data: tocken_train.shape = ', sub_seq_data.shape)

    for d in range(num_subseq):
        inp = sub_seq_data[d]
        if params.use_data_posID:
            pos_id = data_posID[d]

        if params.stage == 'pretrain':
            
            ## choose tokens to MASK
            is_masked = None

            if mask_sample == 'all':
                is_masked = _sample_mask_all_(seg_len=seq_len, 
                                               mask_alpha=nvars-1, mask_beta=1, start_=int(0.25*seq_len), 
                                               kpi_num=nvars, 
                                               reverse=False, goal_num_predict=num_predict)
            else:
                if num_predict is None:
                    num_predict_0 = None
                    num_predict_1 = None
                else:
                    num_predict_1 = num_predict // 2
                    num_predict_0 = num_predict - num_predict_1
                mask_0 = _sample_mask(seg=inp[:int(0.5*seq_len)], 
                                      data_dimensions=nvars, h_number=num_patch, 
                                      reverse=False, goal_num_predict=num_predict_0)
                mask_1 = _sample_mask(seg=inp[int(0.5*seq_len):], 
                                      data_dimensions=nvars, h_number=num_patch, 
                                      reverse=False, goal_num_predict=num_predict_1)
        # """
        # seg_id : segment id
        # """
        # seg_id = np.array(([1] * (seq_len // 2) + [1] * (seq_len // 2)))

        if params.stage == 'pretrain':
            if is_masked is None:
                is_masked = np.concatenate([mask_0, mask_1], 0)
        
            if num_predict is not None:
                assert np.sum(is_masked) == num_predict
        feature = {
                    "input": inp,
                    # "seg_id": seg_id
                    }
        if params.decomposition:
            feature["input_k_decomp"] = torch.from_numpy(sub_seq_decomp[d]).float()
        if params.use_data_posID:
            feature["Position_ID"] = pos_id
        if params.stage == 'pretrain':
            feature["is_masked"] = is_masked
        features.append(feature)
    
    return features, [nvars, num_patch, patch_len]

def _sift_mask(seq_len, is_masked, func_tokens=False):
    """
    Filter out masked tokens and create an attention mask accordingly.
    
    Args:
        inputs: input ids. inp or [ dataA | dataB ].
        is_masked: bool Tensor in shape [seq_len]. True means being selected for partial prediction.
          Details of the MASK, token masked is marked as 'True', others are 'False'.
    """
    index = torch.arange(seq_len, dtype=torch.int64)
    
    if func_tokens:
        ## non_func_tokens 是指非[SEP]和[CLS]的未被MASK的 tokens; [SEP]/[CLS]位置为False
        non_func_tokens = torch.BoolTensor([1]*seq_len)
        ## 定位未 MASK 的非功能性词的位置，设置为 False 
        non_mask_tokens = (~is_masked) & non_func_tokens
        ## masked_or_func_tokens 和 non_mask_tokens相反, 包括被 MASK 的 tokens 和 [SEP]与[CLS]
        masked_or_func_tokens = ~non_mask_tokens
    else:
        non_mask_tokens = (~is_masked)
        masked_or_func_tokens = is_masked

    smallest_index = -torch.ones([seq_len], dtype=torch.int64)
    # put -1 if `non_mask_tokens(real token not cls or sep)` not permutation index
    rev_index = torch.where(non_mask_tokens, smallest_index, index)
    '''
        Create `target_mask`: non-funcional and masked tokens
        1: use mask as input and have loss 
        0: use token (or [SEP], [CLS]) as input and do not have loss
        
        target_tokens: masked tokens
    '''
    if func_tokens:
        target_tokens = masked_or_func_tokens & non_func_tokens
    else:
        target_tokens = is_masked
    target_mask = target_tokens.type(torch.float32)

    # Create `perm_mask`
    # `target_tokens` cannot see themselves
    # put `rev_index` if real mask(not cls or sep) else `rev_index + 1`
    self_rev_index = torch.where(target_tokens, rev_index, rev_index + 1)
    
    # 1: if i <= j & j is masked_or_func_tokens, then it can not be attend by other tokens
    # 0: if i > j or j is not masked, set value '0'
    perm_mask = (self_rev_index[:, None] <= rev_index[None, :]) &  masked_or_func_tokens
    perm_mask = perm_mask.type(torch.float32)
    
    ## construct inputs_q, 即 MASKed 的非功能性词的标识
    inputs_q = target_mask

    return perm_mask, target_mask, inputs_q

def make_feature(feature: dict,
                 nvars: int, 
                 num_patch: int, 
                 num_predict: int,
                 params):
    '''
    Args:
        feature: {"input": cat_data, "is_masked": is_masked, "seg_id": seg_id, "Position_ID" = pos_id}
         
    '''
    inputs = torch.from_numpy(feature.pop("input")).float()
    seq_len = nvars*num_patch
        
    if params.stage == 'pretrain':
        is_masked = torch.BoolTensor(feature.pop("is_masked"))
     
        perm_mask, target_mask, input_q = _sift_mask(seq_len, is_masked)
    
        if num_predict is not None:
            indices = torch.arange(seq_len, dtype=torch.int64)
            bool_target_mask = target_mask.bool()
            indices = indices[bool_target_mask]
            ## extra padding due to CLS/SEP introduced after prepro

            actual_num_predict = indices.shape[0]
            pad_len = num_predict - actual_num_predict
            assert seq_len >= actual_num_predict
        
            ##### target_mapping #####
            ## 用 one-hot编码 标识被 MASK 的位置, 每个 one-hot编码 的长度为子序列的长度 seq_len
            ## target_mapping 的每一行为一个 one-hot 向量, 值为 1 的位置对应被 MASK 的 token 的位置索引
            ## target_mapping.shape = [actual_num_predict, seq_len]
            target_mapping = torch.eye(seq_len, dtype=torch.float32)[indices] 
            paddings = torch.zeros([pad_len, seq_len], dtype=target_mapping.dtype)
            target_mapping = torch.cat([target_mapping, paddings], dim=0)
            
            target_mapping_uni = torch.zeros(nvars, num_predict, num_patch)
            for i in range(num_predict):
                temp = torch.where(target_mapping[i]==1)
                target_mapping_uni[temp[0].item()%nvars, i, temp[0].item()//nvars] = 1
            feature["target_mapping"] = torch.reshape(target_mapping, [num_predict, seq_len])
            feature["target_mapping_uni"] = target_mapping_uni
            
            target_mapping_uni_sum = target_mapping_uni.sum(dim=1, keepdim=True)
            perm_mask_uni = torch.zeros(nvars, num_patch, num_patch)
            for i in range(nvars):
                index = torch.arange(num_patch, dtype=torch.int64)
                is_masked_ = target_mapping_uni_sum[i].squeeze().bool()
                smallest_index = -torch.ones([num_patch], dtype=torch.int64)
                rev_index = torch.where(~is_masked_, smallest_index, index)
                self_rev_index = torch.where(is_masked_, rev_index, rev_index + 1)
                perm_mask_ = (self_rev_index[:, None] <= rev_index[None, :]) &  is_masked_
                perm_mask_uni[i] = perm_mask_.type(torch.float32)
            # perm_mask_uni = perm_mask_uni[:, :, :, None]
            feature["perm_mask_uni"] = perm_mask_uni

            ##### target #####
            target = torch.arange(seq_len)
            target = target[bool_target_mask]
            paddings = torch.zeros([pad_len], dtype=target.dtype)
            target = torch.cat([target, paddings], dim=0)
            target = torch.reshape(target, [num_predict])
            # feature["target_partial_pred"] = torch.reshape(target, [num_predict])
            feature["target_partial_pred"] = inputs[target.long()]

            ##### target_masked_idx #####
            target_0 = (target%seq_len).int()
            target_masked_idx = torch.zeros(nvars, num_predict)
            for m in range(num_predict):
                temp = target_0[m].item()
                target_masked_idx[temp%nvars, m] = 1
            feature["target_masked_idx"] = target_masked_idx

            # feature["target_masked_tokens"] =  inputs[feature["target_partial_pred"].long()]
        
            ##### target mask #####
            # target_mask = torch.cat(
            #     [torch.ones([actual_num_predict], dtype=torch.float32),
            #      torch.zeros([pad_len], dtype=torch.float32)],
            #     dim=0)
            # feature["target_mask"] = torch.reshape(target_mask, [num_predict])
        # else:
        #     feature["target_mask"] = torch.reshape(target_mask, [seq_len])
        
    # if params.use_seg_id:
    #     feature["seg_id"] = torch.IntTensor(feature["seg_id"])
    feature["input_k"] = inputs
    if params.stage == 'pretrain':
        feature["perm_mask"] = torch.reshape(perm_mask, [seq_len, seq_len])
        # feature["input_q"] = torch.reshape(input_q, [seq_len])
    
    return feature
