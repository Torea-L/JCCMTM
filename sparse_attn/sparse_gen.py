import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse

import yaml
from making_sparse import generate_sparse_att
from utils.tools import get_configs

parser = argparse.ArgumentParser(description='PyTorch Model')

# parser.add_argument('--model', type=str, required=True, default='CICDTSM', help="model name")
parser.add_argument('--cfg', type=str, required=True, default='data_parameters_336-96_v2.yaml', 
                    help='dataset and model hyperparameters configure file path')
parser.add_argument('--root_path', type=str, default='/home/CICDTSM_BSZF/', help='root path of data files')

parser.add_argument('--n_var', type=int, default=7, help="Number of series in MTS")
parser.add_argument('--seq_len', type=int, default=-1, help="Number of tokens in TSaS")
# parser.add_argument('--reuse_len', type=int, default=0,
#                     help="Number of tokens that can be reused as memory in Multi-Module")
# parser.add_argument('--reuse_len_uni', type=int, default=0,
#                     help="Number of tokens that can be reused as memory in Uni-Module")
parser.add_argument('--mem_len', type=int,default=0, help="Number of tokens to cache in Multi-Module")
parser.add_argument('--mem_len_uni', type=int,default=0, help="Number of tokens to cache in Uni-Module")

parser.add_argument('--sparse_path', type=str, default='sparse_attn/', help="Loading Path of Sparse Attention Matrix")
parser.add_argument('--sparse_id', type=str, default='Null', help="ID of sparse attention matrix file, \
                    Do not use sparse attention if set to `Null`")
parser.add_argument('--block_num', type=int, default=None, help='Window size `K` in Window Attention')
parser.add_argument('--pre_group_num', type=int, default=0, help='Band size `b_s` in Band Attention')
parser.add_argument('--random_ratio', type=float, default=0., help='Random ratio `r` in Random Attention')

args = parser.parse_args()
print(args)

## ==== Load data and model settings ==== 
configs = get_configs(args.cfg)

h_number = configs.data.patch_num
configs.data.n_vars = args.n_var
# args.block_num=h_number

if args.seq_len == -1:
    args.seq_len = args.n_var * h_number
if args.mem_len == -1:
    args.mem_len = args.n_var * h_number
if args.mem_len_uni == -1:
    args.mem_len_uni = h_number

sparse_id = "K{}_bs{}_r{}_ID{}".format(args.n_var, args.pre_group_num, args.random_ratio, args.sparse_id)
if args.mem_len==0:
    use_mem = False
sparse_attn_mask, sparse_attn_mask_mem = generate_sparse_att(seq_len=args.seq_len,
                                                             n_vars=configs.data.n_vars,
                                                             params=args,
                                                             conj=configs,
                                                             time_start=sparse_id,
                                                             pre_group_num=args.pre_group_num,
                                                             use_mem=use_mem,
                                                            #  num_rand_blocks=args.block_num,
                                                             num_rand_blocks_ratio=args.random_ratio)
