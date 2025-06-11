# -- coding: utf-8 --

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import warnings

import numpy as np
import pandas as pd
import torch
import yaml
from data_utils import (creat_patch, create_data_mask, create_pred_target, 
                        generate_subseq, make_feature)
from layers.basics import raw_series_decomp
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils.tools import namespace2dict

warnings.filterwarnings("ignore")


type_map = {'train': 0, 'vali': 1, 'test': 2}

def pretrain_data_generate(args_parse:argparse.Namespace, configs:argparse.Namespace, 
                           border1s:list=None, border2s:list=None,
                           drop=False, batch_stride:int=None):
    output_folder = args_parse.output_folder
    # dataset_folder = args_parse.dataset_folder
    drop_list = args_parse.drop_list

    features_multi_bsz = []
    n_patch, n_pred = 0, 0
    batch_size = configs.data.batch_num_use_mem

    border1, border2 = None, None

    if drop_list is not None: drop = True

    if args_parse.task=='LTSF':
        data, border1, border2, data_decomp = _read_data(
            args_parse=args_parse, configs=configs, 
            border1s=border1s, border2s=border2s, 
            drop=drop, drop_lst=drop_list)
    else:
        data, border1, border2, data_decomp = _read_ADdata(
            args_parse=args_parse, configs=configs, 
            border1s=border1s, border2s=border2s, 
            drop=drop, drop_lst=drop_list)

    data_tensor = torch.from_numpy(data)
    if data_decomp is not None:
        data_decomp_tensor = torch.from_numpy(data_decomp)
    print(data_tensor.shape)
    if batch_size > 1:
        if batch_stride is None:
            batch_stride = int(configs.data.context_points/batch_size)
            border2 -= batch_size*batch_stride
    else:
        batch_size = 1
        if batch_stride is None:
            batch_stride = 0
    for b in range(batch_size):
        print('#'*15,' '*5, 'Batch %d'%(b), ' '*5,'#'*15)
        data_ = data_tensor[border1:border2]
        if data_decomp is not None:
            data_decomp_ = data_decomp_tensor[border1:border2]
            print('Shape of '+args_parse.component+' : ', data_decomp_.shape)
        else:
            data_decomp_=None
        print('Shape of input time series: ', data_.shape)
        print('begin: ', border1)
        print('end: ', border2)
        tokens, tokens_decomp, num_subseq, num_patch, pred_target, num_target, num_vars = \
            get_tokens(data=data_, data_decomp=data_decomp_, configs=configs, args_parse=args_parse)
        print('!!!Generating Pre-train Data!!!')
        print('Shape of tokens: ', tokens.shape)
        print('Number of sub-sequence: ', num_subseq)
        print('Number of patch in one sub-sequence: ', num_patch)

        num_predict, features = get_features(token=tokens,
                                             tokens_decomp=tokens_decomp,
                                             configs=configs,
                                             args_parse=args_parse,
                                             n_vars=num_vars,
                                             num_patch=num_patch)
        
        if batch_size==1:
            features_multi_bsz = features
        else:
            features_multi_bsz.extend(features)
        border1 += batch_stride
        border2 += batch_stride
        n_patch = num_patch
        n_pred = num_predict

    features_multi_bsz = np.array(features_multi_bsz).reshape(batch_size,-1)
    data = features_multi_bsz.transpose(1,0).reshape(-1,)

    conj = "_%d_P=%d-S=%d-Ss=%d-MR=%.2f-PN=%d-MN=%d-BSZ=%d"%(configs.data.context_points, configs.data.patch_len, 
                                                             configs.data.stride, configs.data.stride_subseq,
                                                             configs.data.mask_ratio, 
                                                             n_patch, n_pred,
                                                             configs.data.batch_num_use_mem)
    configs.data.num_predict = n_pred
    configs.data.patch_num = n_patch
    
    ## Update configs
    with open(args_parse.configs_path, 'w', encoding='utf-8') as file:
        configs = namespace2dict(configs)
        yaml.dump(configs, file)
    if args_parse.task=='AD':
        if not os.path.exists(output_folder): os.makedirs(output_folder)
        np.save(os.path.join(output_folder, "%s%s.npy"%(args_parse.dset, conj)), data)
    else:
        save_path = os.path.join(output_folder, args_parse.dset)
        if not os.path.exists(save_path): os.makedirs(save_path)
        np.save(os.path.join(save_path, "%s%s.npy"%(args_parse.dset, conj)), data)


def _read_data(args_parse:argparse.Namespace, configs:argparse.Namespace, 
               border1s:list=None, border2s:list=None,
               drop=False, drop_lst:list=None, scale=True):
    print('Loading original data...')
    data_path = args_parse.data_path
    print('Dataset path : ', data_path)
    if 'PEMS' in args_parse.dset:
        data_df_raw = np.load(data_path, allow_pickle=True)
        data_df_raw = pd.DataFrame(data_df_raw['data'][:, :, 0])
    elif args_parse.dset=='Solar':
        df_raw = []
        with open(data_path, "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)
        df_raw = np.stack(df_raw, 0)
        data_df_raw = pd.DataFrame(df_raw)
    else:
        data_df_raw = pd.read_csv(data_path)
    
    if (args_parse.features == 'M' or args_parse.features == 'MS') and drop:
        df_data = data_df_raw.drop(labels=drop_lst, axis=1)
    elif args_parse.features == 'S':
        df_data = data_df_raw[[args_parse.target]]
    else: df_data = data_df_raw
    
    print('Dataset Category : ', args_parse.category)
    if (border1s is None) or (border2s is None):
        num_train = int(len(df_data) * configs.data.train_ratio)
        num_test = int(len(df_data) * configs.data.test_ratio)
        num_vali = len(df_data) - num_train - num_test
        border1s = [0, num_train - configs.data.context_points, len(df_data) - num_test - configs.data.context_points]
        border2s = [num_train, num_train + num_vali, len(df_data)]
    
    if scale:
        if args_parse.scaler_type=='Standard':
            scaler = StandardScaler()
        elif args_parse.scaler_type=='MinMax':
            scaler = MinMaxScaler()
        else: raise Exception(
            'Invalid "scaler" type: {}, which should be "Standard" or "MinMax".'.format(args_parse.scaler_type))
        print('Scaler type : ', args_parse.scaler_type)
        
        '''## For low data distribution drift:
        train_data = df_data'''

        print('border1s : ', border1s)
        print('border2s : ', border2s)
        train_data = df_data[border1s[0]:border2s[0]]
        scaler.fit(train_data.values)

        data = scaler.transform(df_data.values)
    else:
        data = df_data.values
    
    ## Series Decomposition
    data_decomp = None
    if args_parse.decomposition:
        print('Decompose the series...')
        decomp_ = raw_series_decomp(kernel_size=configs.model.kernel_size)
        data_res_, data_trend_ = decomp_(torch.from_numpy(data))
        print(data_trend_.shape)
        if args_parse.component=='trend':
            print('Trend')
            data_decomp = data_trend_
        else:
            print('Residual')
            data_decomp = data_res_
    border1 = border1s[type_map[args_parse.category]]
    border2 = border2s[type_map[args_parse.category]]
    print('Start time step : ', border1)
    print('End time step : ', border2)

    data = data[border1:border2]
    if data_decomp is not None:
        data_decomp = data_decomp[border1:border2]
    return data, border1, border2, data_decomp

def _read_ADdata(args_parse:argparse.Namespace, configs:argparse.Namespace, 
                 border1s:int=None, border2s:int=None,
                 drop=False, drop_lst:list=None, scale=True):
    print('Loading original dataset...')
    data_path = args_parse.data_path
    print('Dataset path : ', data_path)
    if data_path.endswith('.csv'):
        data = pd.read_csv(data_path)
    else: ## endswith('.npy')
        data = np.load(data_path)

    if drop_lst is not None:
        if isinstance(data, pd.DataFrame) and isinstance(drop_lst, list):
            data = data.drop(labels=drop_lst, axis=1)
        if isinstance(drop_lst, int):
            if isinstance(data, pd.DataFrame):
                data = data.values
            data = data[:,:drop_lst]
    
    if args_parse.has_nan:
        if isinstance(data, pd.DataFrame):
            data = data.values
        data = np.nan_to_num(data)

    print('Data length: ', len(data))
    if args_parse.split:
        if (border1s is None) or (border2s is None):
            if args_parse.category == 'train':
                num_train = int(len(data) * configs.data.train_ratio)
                num_test = int(len(data) * configs.data.test_ratio)
                num_vali = len(data) - num_train - num_test
                border1s = [0, num_train - configs.data.context_points, len(data) - num_test - configs.data.context_points]
                border2s = [num_train, num_train + num_vali, len(data)]
    else:
        border1s = [0, 0, 0]
        border2s = [len(data), len(data), len(data)]
    
    if scale:
        if args_parse.scaler_type=='Standard':
            scaler = StandardScaler()
        elif args_parse.scaler_type=='MinMax':
            scaler = MinMaxScaler()
        else:
            raise Exception(
                'Invalid "scaler" type: {}, which should be "Standard" or "MinMax".'.format(args_parse.scaler_type))
        print('Scaler type : ', args_parse.scaler_type)
        
        if isinstance(data, pd.DataFrame):
            data = scaler.fit_transform(data.values)
        else:
            data = scaler.fit_transform(data)
    
    ## Series Decomposition
    data_decomp = None
    if args_parse.decomposition:
        print('Decompose the series...')
        decomp_ = raw_series_decomp(kernel_size=configs.model.kernel_size)
        data_res_, data_trend_ = decomp_(torch.from_numpy(data))
        print(data_trend_.shape)
        if args_parse.component=='trend':
            print('Trend')
            data_decomp = data_trend_
        else:
            print('Residual')
            data_decomp= data_res_
    
    border1 = border1s[type_map[args_parse.category]]
    border2 = border2s[type_map[args_parse.category]]
    print('Start time step : ', border1)
    print('End time step : ', border2)
    data = data[border1:border2]
    if data_decomp is not None:
        data_decomp = data_decomp[border1:border2]

    return data, border1, border2, data_decomp

def get_features(token:torch.Tensor, 
                 tokens_decomp:torch.Tensor,
                 args_parse:argparse.Namespace,
                 configs:argparse.Namespace,
                 n_vars:int,
                 num_patch:int):
    """
    token: [num_subseq, nvars, num_patch, patch_len]
    """
    token_np = token.numpy()
    if tokens_decomp is not None:
        token_decomp_np = tokens_decomp.numpy()
    else:
        token_decomp_np = None
    seq_len = n_vars*num_patch
    num_predict=int(seq_len*configs.data.mask_ratio)
    print('num_predict : ', num_predict)
    features, _ = create_data_mask(tokens_data=token_np,
                                   tokens_decomp=token_decomp_np,
                                   num_predict=num_predict,
                                   params=args_parse,
                                   mask_sample='all')
    features_final = []
    for feature in features:
        feature_ = make_feature(feature,
                                n_vars, num_patch,
                                num_predict=num_predict, 
                                params=args_parse)
        features_final.append(feature_)
    return num_predict, features_final

def get_tokens(data:torch.Tensor, data_decomp:torch.Tensor, 
               configs:argparse.Namespace, args_parse:argparse.Namespace):
    """
    Input:
        data: [-1, num_vars]
        data_decomp: [-1, num_vars] or None
    Return:
        num_subseq : Number of subsequences generated
        num_patch : Number of patches generated from one look-back window
        num_vars : Number of channels in the MTS data
        tokens : [num_subseq, nvars, num_patch, patch_len]
        num_target : Number of target in prediction task
        pred_target : [num_target, num_vars, pred_len]]
    """
    
    num_vars = data.shape[1]
    
    if args_parse.stage == 'finetune' and args_parse.task == 'LTSF':
        pred_target, num_target = create_pred_target(data=data,
                                                     look_back_len=configs.data.context_points,
                                                     pred_len=configs.data.target_points,
                                                     stride=configs.data.stride_subseq)
        # if args_parse.category == 'train':
        data = data[:-configs.data.target_points,:]
        if args_parse.decomposition:
            data_decomp = data_decomp[:-configs.data.target_points,:]
    else:
        pred_target = None
        num_target = None

    inp_seq, num_subseq = generate_subseq(data=data, 
                                           look_back_len=configs.data.context_points, 
                                           stride=configs.data.stride_subseq)
    tokens, num_patch = creat_patch(inp_seq=inp_seq,
                                    patch_len=configs.data.patch_len,
                                    stride=configs.data.stride)
    tokens_decomp = None
    if args_parse.decomposition:
        inp_seq_decomp, _ = generate_subseq(data=data_decomp, 
                                           look_back_len=configs.data.context_points, 
                                           stride=configs.data.stride_subseq)
        tokens_decomp, _ = creat_patch(inp_seq=inp_seq_decomp,
                                    patch_len=configs.data.patch_len,
                                    stride=configs.data.stride)
    
    return tokens, tokens_decomp, num_subseq, num_patch, pred_target, num_target, num_vars

## not used
def LTF_data_generation(args_parse:argparse.Namespace, configs:argparse.Namespace, 
                        border1s:list, border2s:list):
    '''
    Data generation for Long-term MTS Forecasting (LTSF)
    '''
    drop_list = args_parse.drop_list
    border1, border2, drop = None, None, False
    if drop_list is not None: drop = True
    data, border1, border2, data_decomp = _read_data(
        args_parse=args_parse, configs=configs, 
        border1s=border1s, border2s=border2s,
        drop=drop, drop_lst=drop_list)

    data_tensor = torch.from_numpy(data)
    if data_decomp is not None:
        data_decomp_tensor = torch.from_numpy(data_decomp)
        print('Shape of '+args_parse.component+' : ', data_decomp_tensor.shape)
    else:
        data_decomp_tensor = None
    print('Shape of input time series: ', data_tensor.shape)
    print('begin: ', border1)
    print('end: ', border2)
    tokens, tokens_decomp, num_subseq, num_patch, pred_target, num_target, num_vars = \
        get_tokens(data=data_tensor, data_decomp=data_decomp_tensor, configs=configs, args_parse=args_parse)
    print('!!!Generating Prediction Data!!!')
    print('Shape of tokens: ', tokens.shape)
    print('Number of sub-sequence: ', num_subseq)
    print('Number of patch in one sub-sequence: ', num_patch)
    print('Number of target in prediction: ', num_target)

    num_predict, features = get_features(token=tokens,
                                         tokens_decomp=tokens_decomp,
                                         configs=configs,
                                         args_parse=args_parse,
                                         n_vars=num_vars,
                                         num_patch=num_patch)
    '''
    MR: mask_ratio
    PN: number of patches in one subsequence
    MN: number of masked tokens in one subsequence
    TR: train data ratio
    PL: prediction length
    '''
    conj = "_%d_P=%d-S=%d-Ss=%d-MR=%.2f-PN=%d-MN=%d-PL=%d"%(configs.data.context_points, configs.data.patch_len, 
                                                                  configs.data.stride, configs.data.stride_subseq, 
                                                                  configs.data.mask_ratio, 
                                                                  num_patch, num_predict,
                                                                  configs.data.target_points)
    save_path = os.path.join(args_parse.output_folder)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(os.path.join(save_path, "%s%s.npy"%(args_parse.dset, conj)), features)
    np.save(os.path.join(save_path, "%s%s_target.npy"%(args_parse.dset, conj)), pred_target)

## not used
def AD_data_generate(args_parse:argparse.Namespace, configs:argparse.Namespace, 
                     border1s:list, border2s:list):
    drop_list = args_parse.drop_list
    border1, border2, drop = None, None, False
    if drop_list is not None: drop = True
    data, border1, border2, data_decomp = _read_ADdata(
        args_parse=args_parse, configs=configs, 
        border1s=border1s, border2s=border2s, 
        drop=drop, drop_lst=drop_list)
    
    data_tensor = torch.from_numpy(data)
    if data_decomp is not None:
        data_decomp_tensor = torch.from_numpy(data_decomp)
        print('Shape of '+args_parse.component+' : ', data_decomp_tensor.shape)
    else:
        data_decomp_tensor = None
    print('Shape of input time series: ', data_tensor.shape)
    print('begin: ', border1)
    print('end: ', border2)
    tokens, tokens_decomp, num_subseq, num_patch, pred_target, num_target, num_vars = \
            get_tokens(data=data_tensor, data_decomp=data_decomp_tensor, configs=configs, args_parse=args_parse)
    '''tokens, num_subseq, num_patch, pred_target, num_target, num_vars = \
        get_tokens(configs=configs, 
                   args_parse=args_parse, 
                   filename=data_filename, 
                   dset_path=dataset_folder, 
                   drop=drop, drop_list=drop_list)'''
    print('!!!Generating Pre-train Data!!!')
    print('Shape of tokens: ', tokens.shape)
    print('Number of sub-sequence: ', num_subseq)
    print('Number of patch in one sub-sequence: ', num_patch)

    if data_filename.endswith('.csv'):
        data_filename = data_filename.strip('.csv')
    elif data_filename.endswith('.txt'):
        data_filename = data_filename.strip('.txt')
    
    num_predict, features = get_features(token=tokens,
                                         tokens_decomp=tokens_decomp,
                                         configs=configs,
                                         args_parse=args_parse,
                                         n_vars=num_vars,
                                         num_patch=num_patch)
    
    conj = "_%d_P=%d-S=%d-Ss=%d-MR=%.2f-PN=%d-MN=%d"%(configs.data.context_points, configs.data.patch_len, 
                                                      configs.data.stride, configs.data.stride_subseq, 
                                                      configs.data.mask_ratio, 
                                                      num_patch, num_predict)
    
    if not os.path.exists(args_parse.output_folder): os.makedirs(args_parse.output_folder)
    np.save(os.path.join(args_parse.output_folder, "%s%s.npy"%(data_filename, conj)), features)
