# -- coding: utf-8 --
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import warnings

from data_provider.data_generation_pretrain import pretrain_data_generate
from utils.args_init import param_init

warnings.filterwarnings("ignore")

LTSF_DSETS = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'illness', 
              'exchange', 'weather', 'electricity', 'Solar']
AD_DSETS = ['SMD', 'MSL', 'SMAP', 'SWaT', 'PSM']
pred_data_folders = dict(illness='illness', exchange='Exchange', weather='weather', 
                         Solar='Solar', electricity='electricity')

parser = argparse.ArgumentParser(description='Data Generation')
# Dataset
parser.add_argument('--cfg', type=str, default='data_parameters_pretrain-256.yaml', help='dataset configure')
parser.add_argument('--root_path', type=str, default='/home/CICDTSM_BSZF/', help='root path of the project')
parser.add_argument('--datasets', type=str, default='datasets', help='file path of the dataset')
parser.add_argument('--data_path', type=str, default='/datasets/Solar/solar_AL.txt', help='file path of the dataset')

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
parser.add_argument('--seq_len', type=int, default=-1, help='input sequence length. If set to `-1`, use the setting in `configs`.')
# Pretrain mask
parser.add_argument('--use_mem', action='store_true', help='If `True`, use memory in pre-training')

args = parser.parse_args()


if __name__ == '__main__':
    configs = param_init(args, make_data_pretrain=True)

    print('#'*15)
    print('Data & Model configs: \n', configs)
    print('Args in experiment: \n ', args)
    print('#'*15)

    # print(f'args.data_path: {args.data_path}')
    # print(f'args.dset: {args.dset}')

    border1s, border2s= None, None
    if 'ETTh' in args.dset:
        border1s = [0, 12 * 30 * 24 - configs.data.context_points, 12 * 30 * 24 + 4 * 30 * 24 - configs.data.context_points]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
    elif 'ETTm' in args.dset:
        border1s = [0, 12 * 30 * 24 * 4 - configs.data.context_points, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - configs.data.context_points]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]

    drop_list = None
    data_path = args.data_path
    out_folder = 'prepare_input/'
    out = None
    if args.dset in LTSF_DSETS:
        if 'ETT' in args.dset:
            dataset_folder = os.path.join(args.root_path, args.datasets, 'ETT-small/')
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
            data_path = os.path.join(args.root_path, args.datasets, 'PSM/train.csv')
        elif args.dset in ['MSL', 'SMAP', 'SMD']:
            drop_list = None
            data_path = os.path.join(args.root_path, args.datasets, args.dset, args.dset+'_train.npy')
        else: ## SWaT
            drop_list = -1
            data_path = os.path.join(args.root_path, args.datasets, 'SWaT/swat_train2.csv')
        out = out_folder + 'prepare_input_' + args.dset

    else:
        data_path = os.path.join(args.root_path, args.data_path)
        out = out_folder + 'prepare_input_' + args.dset


    args.data_path = data_path
    args.output_folder = os.path.join(args.root_path, args.datasets, out + '_pretrain', args.category + '_data')
    args.drop_list = drop_list

    pretrain_data_generate(args_parse=args, configs=configs, 
                           border1s=border1s, border2s=border2s, batch_stride=configs.data.batch_stride)
