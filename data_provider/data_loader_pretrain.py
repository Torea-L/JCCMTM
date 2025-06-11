
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

class preprocessed_data(Dataset):
    def __init__(self, input_features, 
                 args):
        
        self.args =args
        self.data = input_features
        self.stage = args.stage
        self.category = args.category
        self.features = args.features

    def __getitem__(self, item):
        inp_k = self.data[item]['input_k'].float()
        inp_k = torch.Tensor(inp_k)

        if self.args.decomposition:
            inp_k_decomp = self.data[item]['input_k_decomp'].float()
            inp_k_decomp = torch.Tensor(inp_k_decomp)
        else:
            inp_k_decomp = torch.zeros_like(inp_k)

        if self.stage == 'pretrain':
            target = self.data[item]['target_partial_pred'].float()
            perm_mask = self.data[item]['perm_mask'].float()
            perm_mask_uni = self.data[item]['perm_mask_uni'].float()
            target_mapping = self.data[item]['target_mapping'].float()
            target_mapping_uni = self.data[item]['target_mapping_uni'].float()
            target_masked_idx = self.data[item]['target_masked_idx'].float()

            target = torch.Tensor(target)
            perm_mask = torch.Tensor(perm_mask)
            perm_mask_uni = torch.Tensor(perm_mask_uni)
            target_mapping = torch.Tensor(target_mapping)
            target_mapping_uni = torch.Tensor(target_mapping_uni)
            target_masked_idx = torch.Tensor(target_masked_idx)
            
            return inp_k, inp_k_decomp, target_mapping, target_mapping_uni, target_masked_idx, \
                target, perm_mask, perm_mask_uni
        else:
            return inp_k, inp_k_decomp

    def __len__(self):
        return len(self.data)
    
class prediction_data(Dataset):
    def __init__(self, pred_data ):
        self.pred_data = pred_data     # pred_data [num_target, n_vars, pred_len]
    
    def __getitem__(self, index):
        target_pred = self.pred_data[index]
        
        return target_pred
    def __len__(self):
        return len(self.pred_data)

def get_dataset(root_path, args, category, config, file):
    conj = None
    if 'ETT' in args.dset:
        dset = 'ETT'
    else:
        dset = args.dset
    if args.stage == 'pretrain':
        data_path = 'prepare_input/prepare_input_' + dset + '_' + args.stage
        conj = "_%d_P=%d-S=%d-Ss=%d-MR=%.2f-PN=%d-MN=%d-BSZ=%d"%(config.data.context_points, config.data.patch_len, 
                                                                 config.data.stride, config.data.stride_subseq, 
                                                                 config.data.mask_ratio, 
                                                                 config.data.patch_num, config.data.num_predict, 
                                                                 config.data.batch_num_use_mem)
        if args.task=='LTSF':
            data_file_path = os.path.join(root_path, data_path, 
                                      category+'_data', 
                                      file, file+conj+'.npy')
        else:
            data_file_path = os.path.join(root_path, data_path, 
                                      category+'_data', file+conj+'.npy')
        print('data_file_path : ', data_file_path)
        data = np.load(data_file_path, allow_pickle=True)
        print('data shape : ', data.shape)
        dataset = preprocessed_data(data, args)
        
        return dataset
    else:
        data_path = 'prepare_input/prepare_input_' + dset + '_' + args.task
        conj = "_%d_P=%d-S=%d-Ss=%d-MR=%.2f-PN=%d-MN=%d"%(config.data.context_points, config.data.patch_len, 
                                                                 config.data.stride, config.data.stride_subseq, 
                                                                 config.data.mask_ratio, 
                                                                 config.data.patch_num, config.data.num_predict)
        if args.task == 'LTSF':
            conj += '-PL=%d'%(config.data.target_points)
            pred_data_file_path = os.path.join(root_path, data_path, 
                                               category+'_data', 
                                               file, file+conj+'_target.npy')
            data_file_path = os.path.join(root_path, data_path, 
                                          category+'_data', 
                                          file, file+conj+'.npy')
            print('data_file_path : ', data_file_path)
            print('pred_data_file_path : ', pred_data_file_path)
            pred_data = np.load(pred_data_file_path, allow_pickle=True)
            data = np.load(data_file_path, allow_pickle=True)
            data = data[:pred_data.shape[0]]
            print('Data shape : ', data.shape)
            print('Prediction data shape : ', pred_data.shape[0])
            pred_dataset = prediction_data(pred_data)
            dataset = preprocessed_data(data, args)
            
            return dataset, pred_dataset
        elif args.task == 'AD':
            data_file_path = os.path.join(root_path, data_path, 
                                          category+'_data', 
                                          file+conj+'.npy')
            print('data_file_path : ', data_file_path)
            data = np.load(data_file_path, allow_pickle=True)
            print('Data shape : ', data.shape)
            dataset = preprocessed_data(data, args)
            
            return dataset
        else:
            return None
        
    
