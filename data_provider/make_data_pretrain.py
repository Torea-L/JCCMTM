# -- coding: utf-8 --
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import warnings

from data_provider.data_generation_pretrain import (AD_data_generate,
                                                    LTF_data_generation,
                                                    pretrain_data_generate)
from utils.tools import get_configs

warnings.filterwarnings("ignore")

PRED_DSETS = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'ILI', 'exchange', 
              'weather', 'electricity', 'Solar']
AD_DSETS = ['SMD', 'MSL', 'SMAP', 'SWaT', 'PSM']
STAGE = ['pretrain', 'finetune']
TASK = ['LTSF', 'AD']

data_sets = dict(ETTh1=0, ETTh2=0, ETTm1=0, ETTm2=0, ILI=0, exchange=1, 
                 weather=2, electricity=3, Solar=4, 
                 MSL=5, SMAP=6, PSM=6, SMD=7, SWaT=8)
pred_data_folders = dict(ILI='illness', exchange='Exchange', weather='Weather', 
                         Solar='Solar', electricity='Electricity')
variable_numbers = [7, 8, 21, 321, 137, 55, 25, 38, 51]
type_map = {'train': 0, 'val': 1, 'test': 2}

parser = argparse.ArgumentParser(description='Data Generation')
# Dataset
parser.add_argument('--cfg', type=str, default='data_parameters_pretrain-256.yaml', help='dataset configure')
parser.add_argument('--root_path', type=str, default='/home/CICDTSM_BSZF/', help='root path of the project')
parser.add_argument('--data_path', type=str, default='/Datas/Solar/solar_AL.txt', help='file path of the dataset')
parser.add_argument('--dset', type=str, default='ETTh2', help='dataset name')
parser.add_argument('--stage', type=str, default='pretrain', help="options = ['pretrain', 'finetune']")
parser.add_argument('--task', type=str, default='LTSF', help="options = ['LTSF', 'AD']")
parser.add_argument('--category', type=str, default='train', help="options = ['train', 'vali', 'test']")
parser.add_argument('--scaler_type', type=str, default='Standard', help="scale the input data, options = ['Standard', 'MinMax']")
parser.add_argument('--features', type=str, default='M', help='for multivariate model or univariate model')
parser.add_argument('--use_data_posID', action='store_true', help="Generating position id of tokens")
parser.add_argument('--decomposition', action='store_true', help="Decompose the input time series into `trend` and `residual`")
parser.add_argument('--component', type=str, default='trend', help="options = ['trend', 'res']")
parser.add_argument('--kernel_size', type=int, default=-1, 
                    help="Window size of the series decomposition. If set to `-1`, use the setting in `configs`.")
parser.add_argument('--split', action='store_true', help="If `True`, partition the validation set from the training dataset")
parser.add_argument('--has_nan', type=int, default=0, help="`1` means Nan in the data.")
# Pretrain mask
parser.add_argument('--use_mem', action='store_true', help='If `True`, use memory in pre-training')

args = parser.parse_args()
if 'ETTh' in args.dset:
    configs_path = os.path.join(args.root_path, 'scripts/configs', args.task, 'ETTh', args.cfg)
elif 'ETTm' in args.dset:
    configs_path = os.path.join(args.root_path, 'scripts/configs', args.task, 'ETTm', args.cfg)
else:
    configs_path = os.path.join(args.root_path, 'scripts/configs', args.task, args.dset, args.cfg)

configs = get_configs(configs_path)

args.configs_path = configs_path
args.stage = 'pretrain'
args.category = 'train'

assert args.task in TASK, f"Unrecognized task (`{args.task}`). Options include: {TASK}"
if args.task=='LTSF':
    assert args.dset in PRED_DSETS, f"Unrecognized dset (`{args.dset}`). Options include: {PRED_DSETS}"
else:
    assert (args.dset in AD_DSETS) or ('machine' in args.dset), f"Unrecognized dset (`{args.dset}`). Options include: {AD_DSETS}"

if args.stage =='pretrain' and args.use_mem:
    assert configs.data.stride_subseq == configs.data.context_points, "If using memory, sub-sequences should be non-overlapped."
if args.kernel_size==-1:
    args.kernel_size = configs.model.kernel_size
else:
    configs.model.kernel_size = args.kernel_size

num_vars = variable_numbers[data_sets[args.dset]]
context_points = configs.data.context_points
print('configs:', configs)

border1s, border2s= None, None
if 'ETTh' in args.dset:
    border1s = [0, 12 * 30 * 24 - context_points, 12 * 30 * 24 + 4 * 30 * 24 - context_points]
    border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
elif 'ETTm' in args.dset:
    border1s = [0, 12 * 30 * 24 * 4 - context_points, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - context_points]
    border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
    

if __name__ == '__main__':
    drop_list = None
    data_path = None
    out_folder = 'prepare_input/'
    out = None
    if args.dset in PRED_DSETS:
        if 'ETT' in args.dset:
            dataset_folder = os.path.join(args.root_path, 'Datas/ETT-small/')
            data_path = dataset_folder + args.dset + '.csv'
            out = out_folder + 'prepare_input_ETT'
            drop_list = ['date']
        else:
            data_path = os.path.join(args.root_path, args.data_path)
            out = out_folder + 'prepare_input_' + pred_data_folders[args.dset]
            if not args.dset=='Solar': drop_list = ['date']
    
    elif args.dset in AD_DSETS:
        if args.dset=='PSM':
            drop_list = ['timestamp_(min)']
            args.has_nan = 1
            data_path = os.path.join(args.root_path, 'Datas/PSM/train.csv')
        elif args.dset in ['MSL', 'SMAP', 'SMD']:
            drop_list = None
            data_path = os.path.join(args.root_path, 'Datas', args.dset, args.dset+'_train.npy')
        else: ## SWaT
            drop_list = -1
            data_path = os.path.join(args.root_path, 'Datas/SWaT/swat_train2.csv')
        out = out_folder + 'prepare_input_' + args.dset

    args.data_path = data_path
    args.output_folder = os.path.join(args.root_path, 'Datas', out + '_pretrain', args.category + '_data')
    args.drop_list = drop_list

    if args.stage == 'pretrain':
        pretrain_data_generate(args_parse=args, configs=configs, 
                               border1s=border1s, border2s=border2s, batch_stride=configs.data.batch_stride)
    ## not used
    elif args.stage == 'finetune':
        if args.task == 'LTSF':
            LTF_data_generation(args_parse=args, configs=configs, border1s=border1s, border2s=border2s)
        else:
            AD_data_generate(args_parse=args, configs=configs, border1s=border1s, border2s=border2s)
