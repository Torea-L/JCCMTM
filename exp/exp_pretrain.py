import datetime
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import (CosineAnnealingLR,
                                      CosineAnnealingWarmRestarts, CyclicLR)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_provider.data_loader_pretrain import get_dataset
from exp.exp_basic import Exp_Basic
from models.sub_tasks import Pretrain_Model
from sparse_attn.making_sparse import generate_sparse_att


class Exp_Main(Exp_Basic):
    def __init__(self, args, configs, setting, time_start):
        super(Exp_Main, self).__init__(args, configs, setting)
        ## ==== Initialization of Sparse Attention Mask ==== ##
        self.sparse_attn_mask, self.sparse_attn_mask_mem = self._get_sparse(time_start=time_start)
        self.model = self._build_model().to(self.device)

        print(f'args.dset: {args.dset}')

    def _build_model(self):
        ## ==== Initialization of Pretrain Model ==== ##
        model = Pretrain_Model(configs=self.configs, attn_direction="uni", 
                               clamp_len=-1, same_length=False, 
                               reuse_len_mul=self.args.reuse_len, reuse_len_uni=self.args.reuse_len_uni,
                               mem_len_mul=self.args.mem_len, mem_len_uni=self.args.mem_len_uni,
                               mul_uni_ratio=self.args.mul_uni_ratio, 
                               group_token_num=self.args.group_token_num,
                               efficient=self.args.efficient)
        
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("trainable parameters:", str(trainable_num/1e6), "M")

        if self.args.Continue:
            print('Loading model state dict...')
            pre_state_dict = torch.load(
                os.path.join(self.args.root_path, self.args.checkpoints, 
                             'pretrain/bestloss/', self.setting, 
                             'state_dict-' + self.args.pretrain_model_id + '.pkl'))
            model.load_state_dict(pre_state_dict)
            print('Finish.')
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self):
        dataset = get_dataset(root_path=self.args.root_path+self.args.datasets, args=self.args, 
                              category=self.args.category, config=self.configs, file=self.args.dset)
        tr_dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False, 
                                   num_workers=self.args.num_workers, drop_last=True, pin_memory=True)
        print('dataset.shape = ',len(tr_dataloader))
        return tr_dataloader

    def _select_optimizer(self):
        ## ==== Initialization of Optimizer ==== ##
        model_optim = None
        if self.args.optim_type == 'Adam':
            model_optim = optim.Adam(self.model.parameters(), 
                                     lr=self.args.learning_rate, 
                                     weight_decay=self.args.weight_decay)
        elif self.args.optim_type == 'SGD':
            model_optim = optim.SGD(self.model.parameters(), 
                              lr=self.args.learning_rate, 
                              momentum=0.8, 
                              weight_decay=self.args.weight_decay)
            
        if self.args.LR_mode=='cosineAnn':
            scheduler = CosineAnnealingLR(model_optim, T_max=5, eta_min=0)
        elif self.args.LR_mode=='cosineAnnWarm':
            scheduler = CosineAnnealingWarmRestarts(model_optim,T_0=300,T_mult=1)
        elif self.args.LR_mode=='CycLR':
            scheduler = CyclicLR(model_optim, base_lr=1e-7, max_lr=self.args.learning_rate, 
                             step_size_up=10, step_size_down=self.args.train_epochs-10, 
                             mode='exp_range', gamma=0.99, 
                             cycle_momentum=False)
        return model_optim, scheduler

    def _select_criterion(self):
        assert self.args.criterion_type in ['L1', 'L2'], f"Unrecognized criterion (`{self.args.criterion_type}`). \
            Options include: {['L1', 'L2']}"
        if self.args.criterion_type == 'L2':
            criterion = nn.MSELoss()
        else:
            criterion = nn.L1Loss()
        return criterion

    def _get_sparse(self, time_start):
        ## ==== Initialization of sparse attention ==== ##
        if self.args.load_sparse:
            if self.args.sparse_id == 'Null':
                print('Do not use sparse attention matrix.')
                sparse_attn_mask, sparse_attn_mask_mem = None, None
            else:
                print('Loading sparse attention mask matrix...')
                sparse_attn_mask = np.load(os.path.join(self.args.root_path, self.args.sparse_path, 
                                                        'sparse_attn_mask', 
                                                        'sparse_attn_mask-'+ self.args.sparse_id+'.npy'), 
                                           allow_pickle=True)
                if self.args.mem_len>0:
                    sparse_attn_mask_mem = np.load(os.path.join(self.args.root_path, self.args.sparse_path, 
                                                                'sparse_attn_mask_mem', 
                                                                'sparse_attn_mask-'+ self.args.sparse_id+'.npy'), 
                                           allow_pickle=True)
                else:
                    sparse_attn_mask_mem = None
        else:
            print('Generating new sparse attention mask matrix...')
            if self.args.mem_len==0:
                use_mem = False
            sparse_attn_mask, sparse_attn_mask_mem = generate_sparse_att(seq_len=self.args.TSaS_len,
                                                                         n_vars=self.args.enc_in,
                                                                         params=self.args,
                                                                         conj=self.configs,
                                                                         time_start=time_start,
                                                                         pre_group_num=self.args.pre_group_num,
                                                                         use_mem=use_mem,
                                                                         num_rand_blocks=self.args.block_num,
                                                                         num_rand_blocks_ratio = self.args.random_ratio)
        print('Sparse attention mask matrix preprocessing')
        if sparse_attn_mask is not None:
            sparse_attn_mask = 1-torch.from_numpy(sparse_attn_mask.astype(np.int64)).float()
            sparse_attn_mask = sparse_attn_mask[None,:,:,:].to(self.device) # [1, n_head, qlen, qlen]
            print('Finish, sparse_attn_mask.shape : ', sparse_attn_mask.shape)
        if sparse_attn_mask_mem is not None:
            sparse_attn_mask_mem = 1-torch.from_numpy(sparse_attn_mask_mem.astype(np.int64)).float()
            sparse_attn_mask_mem = sparse_attn_mask_mem[None,:,:,:].to(self.device) # [1, n_head, qlen, mlen]
            print('Finish, sparse_attn_mask_mem.shape : ', sparse_attn_mask_mem.shape)

        return sparse_attn_mask, sparse_attn_mask_mem

    def train(self, setting, exp_id):
        save_path_bestloss = os.path.join(self.args.checkpoints, 'pretrain/bestloss/', setting)
        if not os.path.exists(save_path_bestloss):
            print('Model save path does not exist, creating folder : ' + save_path_bestloss)
            os.makedirs(save_path_bestloss)
        if self.args.temp_model_save:
            save_path_temp = os.path.join(self.args.model_save_path, 'temp/', setting)
            if not os.path.exists(save_path_temp):
                print('Model save path does not exist, creating folder : ' + save_path_temp)
                os.makedirs(save_path_temp)
        
        train_loader = self._get_data()
        # model = self._build_model()
        # model.to(self.device)
        optimizer, scheduler = self._select_optimizer()
        criterion = self._select_criterion()

        ## ==== Initialization of Tensorboard ====
        if self.args.use_tb:
            today = datetime.date.today().strftime('%Y-%m-%d')
            writer_path = os.path.join(self.args.root_path, self.args.tb_path, 'exp_pretrain', today)
            if not os.path.exists(writer_path):
                os.makedirs(writer_path)
            writer = SummaryWriter(log_dir=writer_path)
            model_tag =  self.args.dset + self.args.stage + str(today) + '/'
            
        self.model.train()
        for epoch in range(self.args.train_epochs):
            num_step = 0
            mems = None
            mems_uni = None
            epoch_loss = 0.0
            t1 = time.time()
            # num_samples = 0

            for tr_data in train_loader:
                inp_k, inp_k_decomp, target_mapping, target_mapping_uni, target_masked_idx, target, perm_mask, perm_mask_uni = \
                    tr_data[0], tr_data[1], tr_data[2], tr_data[3], tr_data[4], tr_data[5], tr_data[6], tr_data[7]
                # num_samples += inp_k.shape[0]
                """
                inp_k: [bsz, patch_num, n_vars, patch_len]
                inp_k_decomp: [bsz, patch_num, n_vars, patch_len]
                """
                input_k_decomp = inp_k_decomp.to(self.device) if self.args.decomposition else None
                inp_k = inp_k.to(self.device)
                target = target.to(self.device)
                perm_mask = perm_mask.to(self.device)
                perm_mask_uni = perm_mask_uni.to(self.device)
                target_mapping = target_mapping.to(self.device)
                target_mapping_uni = target_mapping_uni.to(self.device)
                target_masked_idx = target_masked_idx.to(self.device)
                input_mask, input_mask_uni = None, None

                optimizer.zero_grad()
                mems = None
                output_g, output_h, mem_lst = \
                    self.model(inp_k=inp_k, inp_decomp=input_k_decomp, seg_id=None, 
                          sparse_attn_mask=self.sparse_attn_mask, sparse_attn_mem_mask=self.sparse_attn_mask_mem,
                          input_mask_mul=input_mask, input_mask_uni=input_mask_uni,
                          mems_mul=mems, mems_uni=mems_uni,
                          perm_mask_mul=perm_mask, perm_mask_uni=perm_mask_uni, 
                          target_mapping_mul=target_mapping, target_mapping_uni=target_mapping_uni, 
                          target_masked_idx=target_masked_idx,
                          pretrain=True, strategy=self.args.strategy)
                
                loss_query_stream = criterion(output_g, target)
                if self.args.use_tb:
                    writer.add_scalar(tag=model_tag + "loss_query_stream", scalar_value=loss_query_stream.data.cpu(), 
                                global_step=epoch * len(train_loader) + num_step)
                # loss_query_stream = loss_query_stream.type(torch.float32)
                loss_content_stream = criterion(output_h, inp_k)
                loss_query_stream += self.args.w_h*loss_content_stream
                if self.args.use_tb:
                    writer.add_scalar(tag=model_tag + "loss_content_stream", scalar_value=loss_content_stream.data.cpu(), 
                                global_step=epoch * len(train_loader) + num_step)
                    writer.add_scalar(tag=model_tag + "loss_total", scalar_value=loss_query_stream.data.cpu(), 
                                global_step=epoch * len(train_loader) + num_step)
                epoch_loss += loss_query_stream.data.cpu()
                loss_query_stream.backward()
                optimizer.step()

                [mems, mems_uni] = mem_lst
                num_step += 1

            epoch_loss = epoch_loss/len(train_loader)
            if self.args.use_tb:
                writer.add_scalar(tag=model_tag + "learning_rate", 
                          scalar_value=optimizer.state_dict()['param_groups'][0]['lr'], 
                          global_step=epoch)
            t2 = time.time()
            scheduler.step()

            print('Number of Epoch: {:04d}, cost = {:.6f}, time cost = {:.6f}'.format((epoch + 1), epoch_loss, (t2 - t1)))
            if self.args.use_tb:
                writer.add_scalar(tag=model_tag + "loss_epoch", scalar_value=epoch_loss, global_step=epoch)
            if epoch_loss < self.args.best_loss:
                self.args.best_loss = epoch_loss
                print('Save model of best loss...')
                torch.save(self.model.state_dict(), os.path.join(save_path_bestloss, 'state_dict-bestloss-'+exp_id+'.pkl'))
            if self.args.temp_model_save and ((epoch+1)%20 ==0):
                print('num_epoch = ', epoch+1)
                print('Saving model temp...')
                torch.save(self.model.state_dict(), os.path.join(save_path_temp, 'state_dict-'+str(epoch+1)+exp_id+'-pretrain.pkl'))
        
        if self.args.use_tb:
            writer.close()
