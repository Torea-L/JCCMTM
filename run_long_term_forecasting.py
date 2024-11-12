import argparse
import datetime
import random
import warnings

import numpy as np
import torch
from data_provider.data_factory import data_provider
from exp.exp_LTSF import Exp_Main
from utils.args_init import param_init

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PyTorch Model')

# random seed
parser.add_argument('--random_seed', type=int, default=2024, help='random seed')

# basic config
parser.add_argument('--model', type=str, required=True, default='JCCMTM', help='model name')
parser.add_argument('--stage', type=str, default='finetune', help="options: ['pretrain', 'finetune']")
parser.add_argument('--category', type=str, default='train', help="status, options: ['train', 'test']")
parser.add_argument('--task', type=str, default='LTSF', help="sub-tasks, options: ['LTSF', 'AD']")

# model config file
parser.add_argument('--cfg', type=str, required=True, default='ETTm_finetune.yaml', help='exp configure file')

# Dataset and dataloader
parser.add_argument('--dset', type=str, required=True, default='ETTm1', help='dataset name')
parser.add_argument('--data', type=str, required=True, default='custom', help='data name used in data_factory')
parser.add_argument('--root_path', type=str, default='/home/JCCMTM/', help='root path of data files')
parser.add_argument('--data_path', type=str, default='Datas/ETT-small/ETTm1.csv', help='data file')
parser.add_argument('--features', type=str, default='M', help="forecasting task, options:['M', 'S', 'MS']; \
                    M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate")
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--freq', type=str, default='h', help="freq for time features encoding, \
                    options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], \
                    you can also use more detailed freq like 15min or 3h")
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

# patching
parser.add_argument('--patch_len', type=int, default=-1, help='patch length')
parser.add_argument('--stride', type=int, default=-1, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')

# Forecasting task
parser.add_argument('--seq_len', type=int, default=-1, help='input sequence length. If set to `-1`, use the setting in `configs`.')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=-1, help='prediction sequence length. If set to `-1`, use the setting in `configs`.')

# Sparse attention
parser.add_argument('--load_sparse', action='store_true', help="Load a pre-generated sparse attention mask matrix")
parser.add_argument('--sparse_path', type=str, default='sparse_attn/', help="Loading Path of Sparse Attention Matrix")
parser.add_argument('--sparse_id', type=str, default='Null', help="ID of sparse attention matrix file, \
                    Do not use sparse attention if set to `Null`")
parser.add_argument('--block_num', type=int, default=None, help='Window size `K` in Window Attention')
parser.add_argument('--pre_group_num', type=int, default=0, help='Band size `b_s` in Band Attention')
parser.add_argument('--random_ratio', type=float, default=0., help='Random ratio `r` in Random Attention')


# model define
parser.add_argument('--enc_in', type=int, default=-1, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=-1, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')

parser.add_argument('--d_model', type=int, default=-1, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=-1, help='num of heads')
parser.add_argument('--e_layers', type=int, default=-1, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=0, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=-1, help='dimension of fcn')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--decomposition', action='store_true', 
                    help="Decompose the input time series into `trend` and `residual`")
parser.add_argument('--component', type=str, default='trend', help=" options: ['trend', 'residual']")
parser.add_argument('--kernel_size', type=int, default=-1, 
                    help="Window size of the series decomposition. If set to `-1`, use the setting in `configs`.")
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)

# JCCMTM
parser.add_argument('--TSaS_len', type=int, default=-1, help="Number of tokens in the TSaS")
parser.add_argument('--reuse_len', type=int, default=0,
                    help="Number of tokens that can be reused as memory in Multi-Module")
parser.add_argument('--reuse_len_uni', type=int, default=0,
                    help="Number of tokens that can be reused as memory in Uni-Module")
parser.add_argument('--mem_len', type=int,default=0, help="Number of tokens to cache in Multi-Module")
parser.add_argument('--mem_len_uni', type=int,default=0, help="Number of tokens to cache in Uni-Module")
parser.add_argument('--mul_uni_ratio', type=int, default=-1, 
                    help="The ratio `Lm` of Mul-modules to Uni-modules in each CICDTSM backbone. \
                        If set to `-1`, use the setting in `configs`.")
parser.add_argument('--strategy', type=str, default='CICD', help="options: ['CICD', 'CD']")
parser.add_argument('--efficient', action='store_true', help='Use the efficient mode.')
parser.add_argument('--group_token_num', type=int, default=5, help='number of global tokens in efficient mode')

# Optimization
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for DataLoader')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help="train epochs")
parser.add_argument('--batch_size', type=int, default=256, help='batch size of train input data')
parser.add_argument('--test_batch_size', type=int, default=256, help='batch size of validation set and test set')
parser.add_argument('--learning_rate', type=float, default=3e-3, help='optimizer learning rate')
parser.add_argument('--optim_type', type=str, default='Adam', help="options: ['Adam', 'SGD']")
parser.add_argument('--weight_decay', type=float, default=0, help='optimizer weight decay')
parser.add_argument('--patience', type=int, default=5, help="early stopping patience")
parser.add_argument('--lradj', type=str, default='type1', 
                    help="adjust learning rate, options: ['type1', 'type2', 'type3, 'const']")
parser.add_argument('--scale_adjustLR', type=int, default=1, 
                    help="The learning rate scaling parameter used for the `type1` learning rate adjustment strategy.")
parser.add_argument('--delta', type=float, default=1e-3, help="early stopping parameters")
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# Pre-trained model parameters
parser.add_argument('--No_Pre', action='store_true', help="Do not load the pre-trained model")
parser.add_argument('--pretrain_model_id', type=str, required=True, default='None', 
                    help="ID of pre-trained model parameters file")
parser.add_argument('--checkpoints', type=str, default='./checkpoints/',
                    help="Saving and loading paths for model parameters")
parser.add_argument('--temp_save', action='store_true', help="Save temporary model parameters")
parser.add_argument('--temp_epochs', type=int, default=50, help="Save temporary model parameters every `temp_epochs` epochs")

# Tensorboard settings
parser.add_argument('--use_tb', action='store_true', help="Using Tensorboard to record the training process")
parser.add_argument('--tb_path', type=str, default='record/tensorboard/', help="TensorBoard log path")

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

args = parser.parse_args()

# random seed
# fix_seed = args.random_seed
# random.seed(fix_seed)
# torch.manual_seed(fix_seed)
# np.random.seed(fix_seed)

if __name__ == '__main__':
    configs = param_init(args)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    print('CUDA available: ', torch.cuda.is_available())

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('#'*15)
    print('Data & Model configs: \n', configs)
    print('Args in experiment: \n ', args)
    print('#'*15)
    conj = "_sl%d_P%d_S%d_Ss%d_MR%.2f_PN=%d_MN=%d_B=%d"%(
        args.seq_len, configs.data.patch_len, 
        configs.data.stride, configs.data.pre_stride_subseq,
        configs.data.mask_ratio,
        configs.data.patch_num, configs.data.num_predict,
        configs.data.pre_batch_size)
    
    setting = '{}_{}_ft{}_{}_dm{}_nh{}_el{}_dl{}_df{}'.format(
        args.model, args.dset, args.features,
        conj,
        args.d_model, args.n_heads, 
        args.e_layers, args.d_layers, args.d_ff)
    
    setting_pred = '{}_{}_ft{}_dm{}_nh{}_el{}_dl{}_df{}'.format(
        args.model, args.dset, args.features,
        args.d_model, args.n_heads, 
        args.e_layers, args.d_layers, args.d_ff)
    
    Exp = Exp_Main

    time_start = str(datetime.datetime.now())[:19]
    time_start = time_start.replace(' ', '-')
    time_start = time_start.replace(':', '_')
    
    if args.category == 'train':
        exp = Exp(args, configs, setting, time_start)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train()
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(test=True, setting_pred=setting_pred)
        torch.cuda.empty_cache()
    elif args.category == 'val':
        exp = Exp(args, configs, setting, time_start)  # set experiments
        print('>>>>>>>start validation : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        _, vali_loader = data_provider(args, flag='val')
        exp.vali(vali_loader=vali_loader,criterion = torch.nn.MSELoss())
        torch.cuda.empty_cache()
    elif args.category=='test':
        exp = Exp(args, configs, setting, time_start)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(test=True, setting_pred=setting_pred, return_attn=True, save_result=True)
        torch.cuda.empty_cache()
