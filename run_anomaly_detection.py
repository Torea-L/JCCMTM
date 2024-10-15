import argparse
import os
import datetime
import warnings

import torch
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from utils.tools import get_configs

warnings.filterwarnings("ignore")

DSETS = ['MSL', 'PSM', 'SMD', 'SMAP', 'SWaT']
STAGE = ['pretrain', 'finetune']
TASK = ['pred', 'AD']
CATEGORY = ['train', 'test']

data_sets = dict(MSL=0, PSM=1, SMD=2, SMAP=1, SWaT=3)
variable_numbers = [55, 25, 38, 51]

parser = argparse.ArgumentParser(description='PyTorch Model')
# basic config
parser.add_argument('--model', type=str, required=True, default='CICDTSM', help="model name")
parser.add_argument('--cfg', type=str, required=True, default='data_parameters_336-96_v2.yaml', 
                    help='dataset and model hyperparameters configure file path')
parser.add_argument('--stage', type=str, default='finetune', help="options: ['pretrain', 'finetune']")
parser.add_argument('--task', type=str, default='AD', help="options: ['pred', 'AD']")
parser.add_argument('--category', type=str, default='train', help="options: ['train', 'test']")
# data loader
parser.add_argument('--dset', type=str, required=True, default='MSL', help='dataset name')
parser.add_argument('--data', type=str, required=True, default='custom', help='data name used in data_factory')
parser.add_argument('--root_path', type=str, default='/home/CICDTSM_BSZF/', help='root path of data files')
parser.add_argument('--data_path', type=str, default='Datas/ETT-small/ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M', help="forecasting task, options:['M', 'S', 'MS']; \
                    M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate")
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--freq', type=str, default='h', help="freq for time features encoding, \
                    options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], \
                    you can also use more detailed freq like 15min or 3h")
parser.add_argument('--batch_size', type=int, default=256, help='batch size of train input data')
parser.add_argument('--test_batch_size', type=int, default=256, help='batch size of validation set and test set')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for DataLoader')

# anomaly detection task
parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')
# sparse attention
parser.add_argument('--load_sparse', action='store_true', help="Load a pre-generated sparse attention mask matrix")
parser.add_argument('--sparse_path', type=str, default='sparse_attn/', help="Loading Path of Sparse Attention Matrix")
parser.add_argument('--sparse_id', type=str, default='Null', help="ID of sparse attention matrix file, \
                    Do not use sparse attention if set to `Null`")
parser.add_argument('--block_num', type=int, default=None, help='Window size `K` in Window Attention')
parser.add_argument('--pre_group_num', type=int, default=0, help='Band size `b_s` in Band Attention')
parser.add_argument('--random_ratio', type=float, default=0., help='Random ratio `r` in Random Attention')

# CICDTSM
parser.add_argument('--input_len', type=int, default=-1, help='input sequence length. \
                    If set to `-1`, use the setting in `configs`.')
# parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--seq_len', type=int, default=-1, help="Number of tokens in TSaS")
parser.add_argument('--reuse_len', type=int, default=0,
                    help="Number of tokens that can be reused as memory in Multi-Module")
parser.add_argument('--reuse_len_uni', type=int, default=0,
                    help="Number of tokens that can be reused as memory in Uni-Module")
parser.add_argument('--mem_len', type=int,default=0, help="Number of tokens to cache in Multi-Module")
parser.add_argument('--mem_len_uni', type=int,default=0, help="Number of tokens to cache in Uni-Module")
parser.add_argument('--kernel_size', type=int, default=-1, 
                    help="Window size of the series decomposition. If set to `-1`, use the setting in `configs`.")
parser.add_argument('--mul_uni_ratio', type=int, default=-1, 
                    help="The ratio `Lm` of Mul-modules to Uni-modules in each CICDTSM backbone. \
                        If set to `-1`, use the setting in `configs`.")
parser.add_argument('--strategy', type=str, default='CICD', help="options: ['CICD', 'CD']")
parser.add_argument('--decomposition', action='store_true', help="Decompose the input time series into `trend` and `residual`.")
parser.add_argument('--component', type=str, default='trend')
parser.add_argument('--efficient', action='store_true', help='Use the efficient mode.')
parser.add_argument('--group_token_num', type=int, default=5, help='number of global tokens in efficient mode')

# optimization
parser.add_argument('--optim_type', type=str, default='Adam', help="options: ['Adam', 'SGD']")
parser.add_argument('--learning_rate', type=float, default=3e-3, help='optimizer learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='optimizer weight decay')
parser.add_argument('--patience', type=int, default=5, help="early stopping patience")
parser.add_argument('--lradj', type=str, default='type1', 
                    help="adjust learning rate, options: ['type1', 'type2', 'type3, 'const']")
parser.add_argument('--scale_adjustLR', type=int, default=1, 
                    help="The learning rate scaling parameter used for the `type1` learning rate adjustment strategy.")
parser.add_argument('--delta', type=float, default=1e-3, help="early stopping parameters")
parser.add_argument('--train_epochs', type=int, default=10, help="train epochs")
parser.add_argument('--best_loss', type=float, default=0.20)
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
# pre-trained model parameters
parser.add_argument('--No_Pre', action='store_true', help="Do not load the pre-trained model")
parser.add_argument('--linear_prob', action='store_true', help="just use the pre-trained model parameters")
parser.add_argument('--pretrain_model_id', type=str, required=True, default='None', 
                    help="ID of pre-trained model parameters file")
parser.add_argument('--model_save_path', type=str, default='checkpoints/',
                    help="Saving and loading paths for model parameters")
parser.add_argument('--temp_save', action='store_true',
                    help="Save temporary model parameters")
parser.add_argument('--temp_epochs', type=int, default=50, 
                    help="Save temporary model parameters every `temp_epochs` epochs")
# tensorboard settings
parser.add_argument('--use_tb', action='store_true', 
                    help="Using Tensorboard to record the training process")
parser.add_argument('--tb_path', type=str, default='run/',  help="TensorBoard event-file save path")
# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

args = parser.parse_args()

## ==== Load data and model settings ==== 
configs_path = os.path.join(args.root_path, 'src', args.task, args.dset, args.cfg)
configs = get_configs(configs_path)
configs.data.batch_size = args.batch_size

h_number = configs.data.patch_num
kpi_number = variable_numbers[data_sets[args.dset]]
print('kpi number = ', kpi_number)
configs.data.n_vars = kpi_number
print('Data configs:', configs)

assert args.task in TASK, f"Unrecognized task (`{args.task}`). Options include: {TASK}"
assert args.stage in STAGE, f"Unrecognized stage (`{args.stage}`). Options include: {STAGE}"
assert args.dset in DSETS, f"Unrecognized dset (`{args.dset}`). Options include: {DSETS}"

if args.seq_len == -1:
    args.seq_len = kpi_number * h_number
if args.mem_len == -1:
    args.mem_len = kpi_number * h_number
if args.mem_len_uni == -1:
    args.mem_len_uni = h_number
if args.kernel_size==-1:
    args.kernel_size = configs.model.kernel_size
if args.mul_uni_ratio==-1:
    args.mul_uni_ratio = configs.model.mul_uni_ratio
if args.input_len==-1:
    args.input_len=configs.data.context_points

if __name__ == '__main__':
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    print('CUDA available: ', torch.cuda.is_available())
    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    conj = "_sl%d_P%d_S%d_Ss%d_MR%.2f_PN=%d_MN=%d_B=%d"%(configs.data.context_points, configs.data.patch_len, 
                                                         configs.data.stride, configs.data.pre_stride_subseq,
                                                         configs.data.mask_ratio,
                                                         configs.data.patch_num, configs.data.num_predict, 
                                                         configs.data.pre_batch_size)
    setting = '{}_{}_ft{}{}_dm{}_nh{}_el{}_df{}'.format(
        args.model, args.dset, args.features,
        conj,
        configs.model.d_model, configs.model.n_heads, 
        configs.model.n_layers, configs.model.d_ff)
    setting_AD = '{}_{}_ft{}_dm{}_nh{}_el{}_df{}'.format(
        args.model, args.dset, args.features,
        configs.model.d_model, configs.model.n_heads, 
        configs.model.n_layers, configs.model.d_ff)
    
    time_start = str(datetime.datetime.now())[:19]
    time_start = time_start.replace(' ', '-')
    time_start = time_start.replace(':', '_')

    Exp = Exp_Anomaly_Detection

    if args.category == 'train':
        exp = Exp(args, configs, setting, time_start)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train()
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(load=True, in_train=True, setting_pred=setting_AD)
        torch.cuda.empty_cache()
    elif args.category=='test':
        exp = Exp(args, configs, setting, time_start)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(load=True, in_train=True, setting_pred=setting_AD)
        torch.cuda.empty_cache()