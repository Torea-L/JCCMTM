import argparse
import os

from utils.tools import get_configs

DSETS = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'ILI',
         'traffic', 'illness', 'weather', 'exchange', 'electricity', 'Solar']
AD_DSETS = ['SMD', 'MSL', 'SMAP', 'SWaT', 'PSM']
TASK = ['LTSF', 'AD']

data_sets = dict(ETTh1=0, ETTh2=0, ETTm1=0, ETTm2=0, ILI=0, 
                 exchange=1, weather=2, electricity=3, Solar=4,
                 MSL=5, SMAP=6, PSM=6, SMD=7, SWaT=8)
variable_numbers = [7, 8, 21, 321, 137, 55, 25, 38, 51]

def param_init(args:argparse.Namespace):
    assert args.task in TASK, f"Unrecognized task (`{args.task}`). Options include: {TASK}"
    if args.task=='LTSF':
        assert args.dset in DSETS, f"Unrecognized dset (`{args.dset}`). Options include: {DSETS}"
    else:
        assert (args.dset in AD_DSETS) or ('machine' in args.dset), f"Unrecognized dset (`{args.dset}`). Options include: {AD_DSETS}"

    if 'ETTh' in args.dset:
        configs_path = os.path.join(args.root_path, 'scripts/configs', args.task, 'ETTh', args.cfg)
    elif 'ETTm' in args.dset:
        configs_path = os.path.join(args.root_path, 'scripts/configs', args.task, 'ETTm', args.cfg)
    else:
        configs_path = os.path.join(args.root_path, 'scripts/configs', args.task, args.dset, args.cfg)
    
    ## loading model configs from .yaml file
    configs = get_configs(configs_path)
    args.configs_path = configs_path

    if args.dset in data_sets:
        args.enc_in = variable_numbers[data_sets[args.dset]]
    configs.data.n_vars = args.enc_in
    print('Variable number = ', configs.data.n_vars)

    if args.stage == 'pretrain':
        if (args.mem_len==0) and (args.mem_len_uni==0):
            configs.data.batch_size = args.batch_size
        else: args.batch_size = configs.data.batch_size
        args.category = 'train'
    else: configs.data.batch_size = args.batch_size


    if args.seq_len == -1: args.seq_len = configs.data.context_points
    else: configs.data.context_points = args.seq_len
    if args.stage =='finetune' and args.task=='LTSF':
        if args.pred_len == -1:
            args.pred_len = configs.data.target_points
        else: configs.data.target_points = args.pred_len
    if args.TSaS_len == -1: args.TSaS_len = configs.data.n_vars*configs.data.patch_num
    if args.mem_len == -1: args.mem_len = args.TSaS_len
    if args.mem_len_uni == -1: args.mem_len_uni = configs.data.patch_num
    if args.patch_len == -1: args.patch_len = configs.data.patch_len
    else: configs.data.patch_len = args.patch_len
    if args.stride == -1: args.stride = configs.data.stride
    else: configs.data.stride = args.stride

    if args.kernel_size==-1: args.kernel_size = configs.model.kernel_size
    else: configs.model.kernel_size = args.kernel_size
    if args.mul_uni_ratio==-1: args.mul_uni_ratio = configs.model.mul_uni_ratio
    else: configs.model.mul_uni_ratio = args.mul_uni_ratio
    
    if args.e_layers==-1: args.e_layers = configs.model.e_layers
    else: configs.model.e_layers = args.e_layers
    if args.d_layers==-1: args.d_layers = configs.model.d_layers
    else: configs.model.d_layers = args.d_layers
    if args.d_model==-1: args.d_model = configs.model.d_model
    else: configs.model.d_model = args.d_model
    if args.n_heads==-1: args.n_heads = configs.model.n_heads
    else: configs.model.n_heads = args.n_heads
    if args.d_ff==-1: args.d_ff = configs.model.d_ff
    else: configs.model.d_ff = args.d_ff
    if args.dropout==-1: args.dropout = configs.model.dropout
    else: configs.model.dropout = args.dropout

    if configs.model.d_head==-1: configs.model.d_head = int(args.d_model/args.n_heads)
    
    return configs
