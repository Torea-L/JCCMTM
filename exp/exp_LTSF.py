import datetime
import os
import time
import warnings

# import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from sparse_attn.making_sparse import generate_sparse_att
from models.sub_tasks import Prediction_Model, Pretrain_Model
# from models import (Autoformer, DLinear, Informer, Linear, NLinear, PatchTST, Transformer)
from torch import optim
# from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, visual

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args, configs, setting, time_start):
        super(Exp_Main, self).__init__(args, configs, setting)
        ## ==== Initialization of Sparse Attention Mask ====
        self.sparse_attn_mask, self.sparse_attn_mem_mask = self._sparse(time_start=time_start)
        self.pre_model_path = args.pre_model_path
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        ## ==== Initialization of Pretrain Model ====
        model_pre = Pretrain_Model(configs=self.configs, attn_direction="uni", 
                                   clamp_len=-1, same_length=False, 
                                   reuse_len_mul=self.args.reuse_len, reuse_len_uni=self.args.reuse_len_uni,
                                   mem_len_mul=self.args.mem_len, mem_len_uni=self.args.mem_len_uni,
                                   mul_uni_ratio=self.args.mul_uni_ratio, 
                                   group_token_num=self.args.group_token_num,
                                   efficient=self.args.efficient)
        if self.args.No_Pre:
            print('No prerained model used.')
        else:
            print('Loading model state dict...')
            if self.pre_model_path is not None:
                pre_state_dict = torch.load(
                    os.path.join(self.args.root_path, self.args.checkpoints, 
                                 'pretrain/bestloss/', self.pre_model_path,
                                 'state_dict-bestloss-' + self.args.pretrain_model_id + '.pkl'))
            else:
                pre_state_dict = torch.load(
                    os.path.join(self.args.root_path, self.args.checkpoints, 
                                 'pretrain/bestloss/', self.setting, 
                                 'state_dict-bestloss-' + self.args.pretrain_model_id + '.pkl'))
                
            if self.args.cross_domain:
                pre_state_dict.pop("encoder.revin.affine_weight")
                pre_state_dict.pop("encoder.revin.affine_bias")
            
            missing_keys, unexpected_keys = model_pre.load_state_dict(pre_state_dict, strict=False)
            print("Missing keys (required by the current model but not available in the pre trained model):", missing_keys)
            print("Unexpected key (pre trained model has it, but the current model does not need it):", unexpected_keys)
            print('Finish.')

        model = Prediction_Model(pred_len=self.configs.data.target_points, 
                                 dropout=self.configs.model.dropout, 
                                 encoder=model_pre.encoder, 
                                 decomposition=self.args.decomposition, strategy=self.args.strategy,
                                 cross_domain=self.args.cross_domain, 
                                 n_vars_cross_domain=self.configs.data.n_vars)
        
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("trainable parameters:", str(trainable_num/1e6), "M")
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
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
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _sparse(self, time_start):
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
            sparse_attn_mask, sparse_attn_mask_mem = generate_sparse_att(seq_len=self.args.seq_len,
                                                                         n_vars=self.configs.data.n_vars,
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

    def vali(self, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for _, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_decomp) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_decomp = batch_x_decomp.float().to(self.device)

                outputs = self.model(x=batch_x, x_decomp=batch_x_decomp,
                                     sparse_attn_mask=self.sparse_attn_mask,
                                     sparse_attn_mem_mask=self.sparse_attn_mem_mask)
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.configs.data.target_points:, f_dim:]
                batch_y = batch_y[:, -self.configs.data.target_points:, f_dim:].to(self.device)
                
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
            
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self):
        ## ==== Initialization of Tensorboard ====
        if self.args.use_tb:
            today = datetime.date.today().strftime('%Y-%m-%d')
            writer_path = os.path.join(self.args.root_path, self.args.tb_path, 'exp_pred', today)
            if not os.path.exists(writer_path):
                os.makedirs(writer_path)
            writer = SummaryWriter(log_dir=writer_path)
            tb_tag = self.args.task + '_' + self.args.dset + '_' + self.args.stage
            model_tag = tb_tag + str(today) + '/'

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        ## ==== Training ====
        print('End-to-end fine-tuning......')
        best_model_path = os.path.join(self.args.root_path, self.args.checkpoints, self.args.task, 'bestloss/', self.setting,
                                   str(self.configs.data.context_points)+'-'+str(self.configs.data.target_points))
        if not os.path.exists(best_model_path):
            print('Model save path does not exist, creating folder : ' + best_model_path)
            os.makedirs(best_model_path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, delta=-self.args.delta)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            num_samples = 0
            train_loss = 0.0

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_decomp) in enumerate(train_loader):
                num_samples += batch_x.shape[0]
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_decomp = batch_x_decomp.float().to(self.device)

                f_dim = -1 if self.args.features == 'MS' else 0
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(x=batch_x, x_decomp=batch_x_decomp)

                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                else:
                    outputs = self.model(batch_x, x_decomp=batch_x_decomp,
                                         sparse_attn_mask=self.sparse_attn_mask,
                                         sparse_attn_mem_mask=self.sparse_attn_mem_mask)
                    
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    
                train_loss += loss.item()*batch_x.shape[0]

                if (i + 1) % 500 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
            
            print("Epoch: {} cost time: {} lr: {}".format(epoch + 1, time.time() - epoch_time, 
                                                      model_optim.state_dict()['param_groups'][0]['lr']))
            '''scheduler.step()'''
            # train_loss = train_loss/num_samples*self.args.batch_size
            train_loss = train_loss/num_samples
            vali_loss = self.vali(vali_loader, criterion)
            test_loss = self.vali(test_loader, criterion)

            if self.args.use_tb:
                writer.add_scalar(tag=model_tag+"loss_epoch", 
                                  scalar_value=train_loss, 
                                  global_step=epoch+1)
                writer.add_scalar(tag=model_tag+"learning_rate", 
                                  scalar_value=model_optim.state_dict()['param_groups'][0]['lr'], 
                                  global_step=epoch+1)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, best_model_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 2, self.args, scale=self.args.scale_adjustLR)

        self.model.load_state_dict(torch.load(best_model_path+ '/' + 'checkpoint.pkl'))

        if self.args.use_tb:
            writer.close()

        return self.model

    def test(self, save_result=False, test=False, model_path=None, 
             setting_pred=None, return_attn=False, visualization=False):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            if model_path is not None:
                pred_state_dict = torch.load(
                    os.path.join(self.args.root_path, self.args.checkpoints, 
                                 self.args.task, 'bestloss/', model_path,
                                 str(self.configs.data.context_points)+'-'+str(self.configs.data.target_points),
                                'checkpoint.pkl'
                                )
                    )
            else:
                pred_state_dict = torch.load(
                    os.path.join(self.args.root_path, self.args.checkpoints, 
                                 self.args.task, 'bestloss/', self.setting, 
                                str(self.configs.data.context_points)+'-'+str(self.configs.data.target_points),
                                'checkpoint.pkl')
                    )
            self.model.load_state_dict(pred_state_dict)
        
        test_loss_MSE = 0.0
        test_loss_MAE = 0.0
        num_samples = 0
        criterion_mse = nn.MSELoss(reduction='mean')
        criterion_mae = nn.L1Loss(reduction='mean')

        preds = []
        trues = []
        inputs = []
        result_path = os.path.join(self.args.root_path, 'results')
        result_file = os.path.join(result_path, self.args.dset+'_'+self.args.model+'.txt')
        if not os.path.exists(result_path):
            print('Creating result path')
            os.makedirs(result_path)
        if not os.path.exists(result_file):
            print('Creating result file')
            f = open(result_file, 'w')
            f.close()
        
        folder_path = None
        if save_result:
            folder_path = './record/test_results/' + self.setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_decomp) in enumerate(test_loader):
                n_samples = batch_x.shape[0]
                num_samples += n_samples
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_decomp = batch_x_decomp.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(x=batch_x, x_decomp=batch_x_decomp)
                else:
                    if return_attn:
                        outputs, attn_prob_lst, attn_score_lst = self.model(
                            x=batch_x, x_decomp=batch_x_decomp, 
                            sparse_attn_mask=self.sparse_attn_mask,
                            sparse_attn_mem_mask=self.sparse_attn_mem_mask,
                            return_att=return_attn)
                    else:
                        outputs = self.model(x=batch_x, x_decomp=batch_x_decomp,
                                             sparse_attn_mask=self.sparse_attn_mask,
                                             sparse_attn_mem_mask=self.sparse_attn_mem_mask,)

                f_dim = -1 if self.args.features == 'MS' else 0
                
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
                
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                test_loss_MSE += criterion_mse(outputs, batch_y)*n_samples
                test_loss_MAE += criterion_mae(outputs, batch_y)*n_samples
                
                input = batch_x.detach().cpu().numpy()
                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)
                inputs.append(input)

                if return_attn and (i % 200 == 0):
                    attn_prob_lst_M, attn_prob_lst_U = attn_prob_lst
                    attn_score_lst_M, attn_score_lst_U = attn_score_lst
                    layer_M, layer_U = len(attn_prob_lst_M), len(attn_prob_lst_U)
                    attn_prob_M = []
                    # attn_score_M = []
                    for l in range(layer_M):
                        attns_prob = attn_score_lst_M[l][0].cpu()
                        # attn_score = attn_score_lst_M[l][0].cpu()
                        # print('attns_prob.shape = ', attns_prob.shape)
                        attn_prob_M.append(attns_prob[0].numpy())
                        # attn_score_M.append(attn_score[0].numpy())
                    attn_prob_M = np.array(attn_prob_M)
                    # attn_score_M = np.array(attn_score_M)
                    # print('attn_prob_M.shape = ', attn_prob_M.shape)
                    np.save(self.args.root_path + 'results/Attns/' + self.args.dset + '/'+ str(i) + 'attn_probs.npy',
                            attn_prob_M)
                    # np.save(self.args.root_path + 'results/Attns/' + self.args.dset + '/'+ str(i) + 'attn_score.npy',
                    #         attn_score_M)

                    # input = batch_x.detach().cpu().numpy()
                    # gt = batch_y.detach().cpu().numpy()
                    # pd = outputs.detach().cpu().numpy() 
                    gt = np.concatenate((input[0], true[0]), axis=0)
                    pd = np.concatenate((input[0], pred[0]), axis=0)

                # visual
                if visualization and (i % 200 == 0):
                    input = batch_x.detach().cpu().numpy()
                    batch_y = batch_y.detach().cpu().numpy()
                    outputs = outputs.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], batch_y[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], outputs[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())

        if visualization:
            preds = np.array(preds)
            trues = np.array(trues)
            print('test shape:', preds.shape, trues.shape)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            print('test shape:', preds.shape, trues.shape)

            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)

        mae, mse = test_loss_MAE/num_samples, test_loss_MSE/num_samples

        # mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('History Length: {} Prediction Length: {} \n mse:{}, mae:{} \n'.format(
            self.configs.data.context_points, self.configs.data.target_points, mse, mae))

        f = open(result_file, 'a')
        f.write('History Length{}-Prediction Length{} \n  MSE:{} MAE:{}\n'.format(
            self.configs.data.context_points, self.configs.data.target_points, mse, mae))
        f.write('\n')
        f.close()

        return
