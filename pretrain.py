import argparse
import datetime
import faulthandler
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data_provider.data_loader import get_dataset
# from models.CICD_batchfirst import Pretrain_Model
from models.sub_tasks import Pretrain_Model
from sparse_attn.making_sparse import generate_sparse_att
from torch.optim.lr_scheduler import (CosineAnnealingLR,
                                      CosineAnnealingWarmRestarts, CyclicLR)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.tools import get_configs

faulthandler.enable()
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
warnings.filterwarnings("ignore")

DSETS = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'ILI',
         'traffic', 'illness', 'weather', 'exchange', 'electricity', 'Solar']
AD_DSETS = ['SMD', 'MSL', 'SMAP', 'SWaT', 'PSM']
STAGE = ['pretrain', 'finetune']
TASK = ['pred', 'AD']
CATEGORY = ['train', 'test']

data_sets = dict(ETTh1=0, ETTh2=0, ETTm1=0, ETTm2=0, ILI=0, exchange=1, weather=2, electricity=3, Solar=4,
                 MSL=5, SMAP=6, PSM=6, SMD=7, SWaT=8)
variable_numbers = [7, 8, 21, 321, 137, 55, 25, 38, 51]

parser = argparse.ArgumentParser(description='PyTorch Model')
# Dataset and dataloader
parser.add_argument('--cfg', type=str, required=True, default='data_parameters_336-96_v2.yaml', 
                    help='dataset configure')
parser.add_argument('--dset', type=str, default='ETTh2', help='dataset name')
parser.add_argument('--root_path', type=str, default='/home/CICDTSM_BSZF/', help='root path of data files')
parser.add_argument('--stage', type=str, default='pretrain') #'pretrain' or 'finetune'
parser.add_argument('--task', type=str, default='pred')      #'pred or 'AD'
parser.add_argument('--category', type=str, default='train') #'train or 'test'
parser.add_argument('--features', type=str, default='M', help='for multivariate model or univariate model')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for DataLoader')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--decomposition', action='store_true', help="Decompose the input time series into `trend` and `residual`")
parser.add_argument('--component', type=str, default='trend')
# Sparse attention
parser.add_argument('--Continue', action='store_true', 
                    help="Continue training from a saved model")
parser.add_argument('--load_sparse', action='store_true', 
                    help="Whether to load an pre-generated sparse attention mask")
parser.add_argument('--sparse_path', type=str, default='./sparse_attn')
parser.add_argument('--sparse_id', type=str, default='105_2023-06-01-09_48_56')
parser.add_argument('--block_num', type=int, default=None)
parser.add_argument('--pre_group_num', type=int, default=0)
parser.add_argument('--random_ratio', type=float, default=0.)
# Pre trained model parameters
parser.add_argument('--model', type=str, default='CICDTSM-efficient',
                    help="model name")
parser.add_argument('--model_save_path', type=str, default='./checkpoints/pretrain',
                    help="Saving and loading paths for model parameters")
parser.add_argument('--temp_model_save', action='store_true', help="Saving temp model parameters")
parser.add_argument('--device', default='cuda', help='cuda or cpu')
# Model args
parser.add_argument('--seq_len', type=int, default=-1, 
                    help="Sub-Sequence length.")
parser.add_argument('--reuse_len', type=int, default=0,
                    help="Number of tokens that can be reused as memory in Multi-Module.")
parser.add_argument('--reuse_len_uni', type=int, default=0,
                    help="Number of tokens that can be reused as memory in Uni-Module.")
parser.add_argument('--mem_len', type=int,default=0, 
                    help="Number of tokens to cache in Multi-Module.")
parser.add_argument('--mem_len_uni', type=int,default=-1,  
                    help="Number of tokens to cache in Uni-Module.")
parser.add_argument('--bi_data', type=bool, default=False,
                    help="whether to create bidirectional data")
parser.add_argument('--kernel_size', type=int, default=-1, 
                    help="Window size of the series decomposition. If set to `-1`, use the setting in `configs`.")
parser.add_argument('--mul_uni_ratio', type=int, default=-1, 
                    help="The ratio `Lm` of Mul-modules to Uni-modules in each CICDTSM backbone. If set to `-1`, use the setting in `configs`.")
parser.add_argument('--strategy', type=str, default='CICD', help="'CI', 'CD' and 'CICD'")
parser.add_argument('--wo_linear', action='store_false', help="Do not use the linear module")
parser.add_argument('--efficient', action='store_true', help='Use the efficient mode.')
parser.add_argument('--group_token_num', type=int, default=5, help='number of global tokens in efficient mode')
# Optimization args
parser.add_argument('--Continue_optim', action='store_true', 
                    help="Continue training with a saved optimizer")
parser.add_argument('--optim_type', type=str, default='Adam') # 'Adam' or 'SGD'
parser.add_argument('--learning_rate', type=float, default=5e-3)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--LR_mode', type=str, default='CycLR') # 'cosineAnnWarm' or 'cosineAnn' or 'CycLR'
parser.add_argument('--num_epochs', type=int, default=200, help="Number of epochs")
parser.add_argument('--cal_loss_h', type=bool, default=True,
                    help="Whether calculating the loss of the h-stream")
parser.add_argument('--w_h', type=float, default=1.0,
                    help="weight of the loss of the h-stream")
parser.add_argument('--best_loss', type=float, default=np.inf)

# Tensorboard settings
parser.add_argument('--use_tb', action='store_true', 
                    help="Using Tensorboard to record the training process")
parser.add_argument('--tb_path', type=str, default='run/', 
                    help="TensorBoard log path")
parser.add_argument('--tb_tag', type=str, default='Pretrain_ETT_two-loss_sparse-attn_', 
                    help="Number of epochs")
parser.add_argument('--pretrained_model_id', type=int, default=0, 
                    help='Model id to keep track of the number of pre-trained models saved')
# args = parser.parse_args(['--cfg', 'data_parameters_pretrain-96_electricity.yaml', '--dset', 'electricity',
#                           '--learning_rate', '5e-3', '--load_sparse', '--sparse_id', 'Null', '--num_epochs', '100', 
#                           '--decomposition', '--kernel_size', '7', '--root_path', '/home/CICDTSM_BSZF/'])
args = parser.parse_args()

args.stage = 'pretrain'

assert args.stage in STAGE, f"Unrecognized stage (`{args.stage}`). Options include: {STAGE}"
assert args.task in TASK, f"Unrecognized task (`{args.task}`). Options include: {TASK}"
if args.task=='pred':
    assert args.dset in DSETS, f"Unrecognized dset (`{args.dset}`). Options include: {DSETS}"
else:
    assert (args.dset in AD_DSETS) or ('machine' in args.dset), f"Unrecognized dset (`{args.dset}`). Options include: {AD_DSETS}"

if 'ETTh' in args.dset:
    configs_path = os.path.join(args.root_path, 'src', args.task, 'ETTh', args.cfg)
elif 'ETTm' in args.dset:
    configs_path = os.path.join(args.root_path, 'src', args.task, 'ETTm', args.cfg)
elif 'machine' in args.dset:
    configs_path = os.path.join(args.root_path, 'src', args.task, 'SMD', args.cfg)
else:
    configs_path = os.path.join(args.root_path, 'src', args.task, args.dset, args.cfg)
configs = get_configs(configs_path)
args.configs_path = configs_path

h_number = configs.data.patch_num
if 'machine' in args.dset:
    kpi_number = variable_numbers[data_sets['SMD']]
else:
    kpi_number = variable_numbers[data_sets[args.dset]]
print('kpi number = ', kpi_number)
configs.data.n_vars = kpi_number
# print('Data configs:', configs)
conj = "_sl%d_P%d_S%d_Ss%d_MR%.2f_PN=%d_MN=%d_B=%d"%(configs.data.context_points, configs.data.patch_len, 
                                                     configs.data.stride, configs.data.stride_subseq,
                                                     configs.data.mask_ratio,
                                                     configs.data.patch_num, configs.data.num_predict, 
                                                     configs.data.batch_size)
if args.seq_len == -1:
    args.seq_len = kpi_number*h_number
if args.mem_len == -1:
    args.mem_len = args.seq_len
if args.mem_len_uni == -1:
    args.mem_len_uni = h_number
if args.kernel_size==-1:
    args.kernel_size = configs.model.kernel_size
if args.mul_uni_ratio==-1:
    args.mul_uni_ratio = configs.model.mul_uni_ratio
if args.kernel_size == -1:
    args.kernel_size = configs.model.kernel_size

# configs.data.n_vars = kpi_number
print('Data configs:', configs)
print('args:', args)

if __name__ == '__main__':

    time_start = str(datetime.datetime.now())[:19]
    time_start = time_start.replace(' ', '-')
    time_start = time_start.replace(':', '_')
    print('time start : ', time_start)

    files_data = [args.dset]

    for f in files_data:

        setting = '{}_{}_ft{}{}_dm{}_nh{}_el{}_df{}'.format(
            args.model, args.dset, args.features,
            conj,
            configs.model.d_model, configs.model.n_heads, 
            configs.model.n_layers, configs.model.d_ff)
        print(setting)

        save_path_final = os.path.join(args.model_save_path, 'pretrain/final/', setting)
        save_path_bestloss = os.path.join(args.model_save_path, 'pretrain/bestloss/', setting)
        for path in [save_path_final, save_path_bestloss]:
            if not os.path.exists(path):
                print('Model save path does not exist, creating folder : ' + path)
                os.makedirs(path)

        if args.temp_model_save:
            save_path_temp = os.path.join(args.model_save_path, 'temp/', setting)
            if not os.path.exists(save_path_temp):
                print('Model save path does not exist, creating folder : ' + save_path_temp)
                os.makedirs(save_path_temp)

        dataset = get_dataset(root_path=args.root_path+'Datas', params=args, category=args.category, config=configs, file=f)
        if (args.mem_len==0) and (args.mem_len_uni==0):
            bsz = args.batch_size
            configs.data.batch_size = args.batch_size
        else:
            bsz = configs.data.batch_size

        tr_dataloader = DataLoader(dataset, batch_size=bsz, shuffle=False, num_workers=0, drop_last=True, pin_memory=True)
        print('dataset.shape = ',len(tr_dataloader))
        
        ## ==== Initialization of sparse attention ====
        if args.load_sparse:
            if args.sparse_id == 'Null':
                print('Do not use sparse attention matrix.')
                sparse_attn_mask, sparse_attn_mask_mem = None, None
            else:
                print('Loading sparse attention mask matrix...')
                sparse_attn_mask = np.load(os.path.join(args.sparse_path,
                                                'sparse_attn_mask',
                                                'sparse_attn_mask-'+ args.sparse_id+'.npy'), 
                                   allow_pickle=True)
                print('sparse_attn_mask.shape : ', sparse_attn_mask.shape)
                if args.mem_len>0:
                    sparse_attn_mask_mem = np.load(os.path.join(args.sparse_path,
                                                    'sparse_attn_mask_mem',
                                                    'sparse_attn_mask_mem-'+ args.sparse_id +'.npy'), 
                                       allow_pickle=True)
                    print('sparse_attn_mask_mem.shape : ', sparse_attn_mask_mem.shape)
                else:
                    sparse_attn_mask_mem = None
        else:
            print('Generating sparse attention mask matrix...')
            if args.mem_len==0:
                use_mem = False
            sparse_attn_mask, sparse_attn_mask_mem = generate_sparse_att(seq_len=args.seq_len,
                                                                     n_vars=kpi_number,
                                                                     params=args,
                                                                     conj = configs,
                                                                     time_start=time_start,
                                                                     pre_group_num=args.pre_group_num,
                                                                     use_mem=use_mem,
                                                                     num_rand_blocks=args.block_num,
                                                                     num_rand_blocks_ratio = args.random_ratio)
        print('Sparse attention mask matrix preprocessing')
        if sparse_attn_mask is not None:
            sparse_attn_mask = 1-torch.from_numpy(sparse_attn_mask.astype(np.int64)).float()
            sparse_attn_mask = sparse_attn_mask[None,:,:,:].to(args.device) # [1, n_head, qlen, qlen]
            print('Finish, sparse_attn_mask.shape : ', sparse_attn_mask.shape)
        if sparse_attn_mask_mem is not None:
            sparse_attn_mask_mem = 1-torch.from_numpy(sparse_attn_mask_mem.astype(np.int64)).float()
            sparse_attn_mask_mem = sparse_attn_mask_mem[None,:,:,:].to(args.device) # [1, n_head, qlen, mlen]
            print('Finish, sparse_attn_mask_mem.shape : ', sparse_attn_mask_mem.shape)

        ## ==== Initialization of Pre-trained Model ====

        model_pre = Pretrain_Model(configs=configs, attn_direction="uni",
                                    clamp_len=-1, same_length=False, 
                                    reuse_len=args.reuse_len, reuse_len_uni=args.reuse_len_uni, 
                                    mem_len=args.mem_len, mem_len_uni=args.mem_len_uni,
                                    mul_uni_ratio=args.mul_uni_ratio, kernel_size=args.kernel_size,
                                    group_token_num=args.group_token_num,
                                    sparse_attn=sparse_attn_mask, sparse_attn_mem=sparse_attn_mask_mem,
                                    efficient=args.efficient, strategy=args.strategy)
        
        trainable_num = sum(p.numel() for p in model_pre.parameters() if p.requires_grad)
        print("trainable parameters:", str(trainable_num/1e6), "M")

        if args.Continue:
            print('Loading model state dict...')
            step_state_dict = torch.load(os.path.join(args.model_save_path, 'final/', 
                                                      f + '_state_dict'+ args.pretrain_model_id+'.pkl'))
            model_pre.load_state_dict(step_state_dict['model_state_dict'])
            print('Finish.')

        model_pre.to(args.device)
        model_pre.train()

        ## ==== Initialization of Optimizer ====
        print('Learning rate : ', args.learning_rate)
        print('weight decay : ', args.weight_decay)
        if args.optim_type == 'Adam':
            optimizer = optim.Adam(model_pre.parameters(), 
                               lr=args.learning_rate, 
                               weight_decay=args.weight_decay)
        elif args.optim_type == 'SGD':
            optimizer = optim.SGD(model_pre.parameters(), 
                              lr=args.learning_rate, 
                              momentum=0.8, 
                              weight_decay=args.weight_decay)
        if args.Continue_optim:
            print('Loading optimizer state dict...')
            optimizer.load_state_dict(step_state_dict['optimizer_state_dict'])
            print('Finish.')
        
        if args.LR_mode=='cosineAnn':
            scheduler = CosineAnnealingLR(optimizer, T_max=5, eta_min=0)
        elif args.LR_mode=='cosineAnnWarm':
            scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=300,T_mult=1)
        elif args.LR_mode=='CycLR':
            scheduler = CyclicLR(optimizer, base_lr=1e-7, max_lr=args.learning_rate, 
                             step_size_up=10, step_size_down=args.num_epochs-10, 
                             mode='exp_range', gamma=0.99, 
                             cycle_momentum=False)

        criterion = nn.MSELoss()

        ## ==== Initialization of Tensorboard ====
        if args.use_tb:
            today = datetime.date.today()
            writer_path = os.path.join(args.root_path, args.tb_path, 'exp_pretrain', today.strftime('%Y-%m-%d'))
            if not os.path.exists(writer_path):
                os.makedirs(writer_path)
            writer = SummaryWriter(log_dir=writer_path)
            model_tag = args.tb_tag + str(args.pretrained_model_id) + conj + '/'

        for num_epoch in range(args.num_epochs):
            num_step = 0
            mems = None
            mems_uni = None
            epoch_loss = 0
            t1 = time.time()

            for tr_data in tr_dataloader:
                inp_k, inp_k_decomp, target_mapping, target_mapping_uni, target_masked_idx, target, perm_mask, perm_mask_uni = \
                    tr_data[0], tr_data[1], tr_data[2], tr_data[3], tr_data[4], tr_data[5], tr_data[6], tr_data[7]
                """
                inp_k: [bsz, patch_num, n_vars, patch_len]
                inp_k_decomp: [bsz, patch_num, n_vars, patch_len]
                """
                input_k_decomp = inp_k_decomp.to(args.device) if args.decomposition else None
                inp_k = inp_k.to(args.device)
                target = target.to(args.device)
                perm_mask = perm_mask.to(args.device)
                perm_mask_uni = perm_mask_uni.to(args.device)
                target_mapping = target_mapping.to(args.device)
                target_mapping_uni = target_mapping_uni.to(args.device)
                target_masked_idx = target_masked_idx.to(args.device)
                input_mask, input_mask_uni = None, None

                optimizer.zero_grad()
                mems = None
                output_g, output_h, mem_lst = \
                    model_pre(inp_k = inp_k, inp_decomp = input_k_decomp,
                            seg_id = None, 
                            input_mask = input_mask, input_mask_uni = input_mask_uni,
                            mems = mems, mems_uni=mems_uni,
                            perm_mask = perm_mask, perm_mask_uni = perm_mask_uni, 
                            target_mapping = target_mapping, target_mapping_uni = target_mapping_uni, 
                            target_masked_idx = target_masked_idx,
                            pretrain=True)
            
                lm_loss_target = criterion(output_g, target)

                if args.use_tb:
                    writer.add_scalar(tag=model_tag + "lm_loss_target", scalar_value=lm_loss_target.data.cpu(), 
                                global_step=num_epoch * len(tr_dataloader) + num_step)
                lm_loss = lm_loss_target.type(torch.float32)

                if args.cal_loss_h:
                    lm_loss_total = criterion(output_h, inp_k)
                    lm_loss += args.w_h*lm_loss_total.type(torch.float32)
                    if args.use_tb:
                        writer.add_scalar(tag=model_tag + "lm_loss_total", scalar_value=lm_loss_total.data.cpu(), 
                                    global_step=num_epoch * len(tr_dataloader) + num_step)
                # print('loss = ', lm_loss.data.cpu())    
                epoch_loss += lm_loss.data.cpu()
                lm_loss.backward()
                optimizer.step()
            
                [mems, mems_uni] = mem_lst
                num_step += 1
            
            if args.use_tb:
                writer.add_scalar(tag=model_tag + "learning_rate", 
                          scalar_value=optimizer.state_dict()['param_groups'][0]['lr'], 
                          global_step=num_epoch)
            t2 = time.time()
            scheduler.step()
        
            print('Number of Epoch: {:04d}, cost = {:.6f}, time cost = {:.6f}'.\
                format((num_epoch + 1), epoch_loss/len(tr_dataloader), (t2 - t1)))
            if args.use_tb:
                writer.add_scalar(tag=model_tag + "loss_epoch", 
                          scalar_value=epoch_loss/len(tr_dataloader), 
                          global_step=num_epoch)
            if epoch_loss/len(tr_dataloader) < args.best_loss:
                args.best_loss = epoch_loss/len(tr_dataloader)
                print('Save model of best loss...')
                torch.save({'model_state_dict':model_pre.state_dict(), 
                            'optimizer_state_dict': optimizer.state_dict()
                            }, os.path.join(save_path_bestloss, 'state_dict-bestloss-'+time_start+'.pkl'))
            if args.temp_model_save and ((num_epoch+1)%10 ==0):
                print('num_epoch = ', num_epoch+1)
                print('Saving model temp...')
                torch.save({'model_state_dict':model_pre.state_dict(), 
                            'optimizer_state_dict': optimizer.state_dict()
                            }, os.path.join(save_path_temp, 'state_dict-'+str(num_epoch+1)+'-pretrain.pkl'))
        if args.use_tb:
            writer.close()
        torch.save({'model_state_dict':model_pre.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict()
                    }, os.path.join(save_path_final, 'state_dict'+time_start+'.pkl'))
