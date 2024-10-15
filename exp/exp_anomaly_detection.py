import datetime
import os
import time
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.multiprocessing
import torch.nn as nn
from torch import optim

torch.multiprocessing.set_sharing_strategy('file_system')

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models.sub_tasks import Anomaly_Detection_Model, Pretrain_Model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sparse_attn.making_sparse import generate_sparse_att
from torch.utils.tensorboard import SummaryWriter
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment


class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args, configs, setting, time_start):
        super(Exp_Anomaly_Detection, self).__init__(args, configs, setting)
        ## ==== Initialization of Sparse Attention Mask ====
        self.sparse_attn_mask, self.sparse_attn_mask_mem = self._sparse(time_start=time_start)
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        ## ==== Initialization of Pretrain Model ====
        model_pre = Pretrain_Model(configs=self.configs, attn_direction="uni", 
                               clamp_len=-1, same_length=False, 
                               reuse_len=self.args.reuse_len, reuse_len_uni=self.args.reuse_len_uni,
                               mem_len=self.args.mem_len, mem_len_uni=self.args.mem_len_uni,
                               mul_uni_ratio=self.args.mul_uni_ratio, kernel_size=self.args.kernel_size,
                               group_token_num=self.args.group_token_num,
                               sparse_attn=self.sparse_attn_mask, sparse_attn_mem=self.sparse_attn_mask_mem,
                               efficient=self.args.efficient, strategy=self.args.strategy)
        if self.args.No_Pre:
            print('No prerained model used.')
        else:
            print('Loading model state dict...')
            pre_state_dict = torch.load(
                os.path.join(self.args.root_path, self.args.model_save_path, 
                             'pretrain/bestloss/', self.setting, 
                             'state_dict-bestloss-' + self.args.pretrain_model_id + '.pkl'))
            model_pre.load_state_dict(pre_state_dict['model_state_dict'])
            print('Finish.')

        model = Anomaly_Detection_Model(dropout=self.configs.model.dropout, encoder=model_pre.encoder, 
                                        decomposition=self.args.decomposition, strategy=self.args.strategy)
        if self.args.linear_prob:
            for name, param in model.named_parameters():
                if "encoder" in name:
                    param.requires_grad = False

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        # torch.save(model.state_dict(), '/home/CICDTSM_BSZF/checkpoints/pred/bestloss/CICDTSM-efficient_weather_ftM_sl96_P12_S12_Ss96_MR0.30_PN=8_MN=50_B=20_dm256_nh8_el2_df512/96-96' + '/' + 'checkpoint.pkl')
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = None
        if self.args.linear_prob:
            params = self.model.recons_head.parameters()
        else:
            params = self.model.parameters()
        if self.args.optim_type == 'Adam':
            model_optim = optim.Adam(params, 
                                     lr=self.args.learning_rate, 
                                     weight_decay=self.args.weight_decay)
        elif self.args.optim_type == 'SGD':
            model_optim = optim.SGD(params, 
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
        if sparse_attn_mask is not None:
            print('Sparse attention mask matrix preprocessing')
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
            for i, (batch_x, batch_x_decomp, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_decomp = batch_x_decomp.float().to(self.device)

                outputs = self.model(x=batch_x, x_decomp=batch_x_decomp)
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
            
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self):
        ## ==== Initialization of Tensorboard ====
        if self.args.use_tb:
            today = datetime.date.today().strftime('%Y-%m-%d')
            writer_path = os.path.join(self.args.root_path, self.args.tb_path, 'exp_AD', today)
            if not os.path.exists(writer_path):
                os.makedirs(writer_path)
            writer = SummaryWriter(log_dir=writer_path)
            tb_tag = self.args.task + '_' + self.args.dset + '_' + self.args.stage
            model_tag = tb_tag + str(today) + '/'

        _, train_loader = self._get_data(flag='train')
        _, vali_loader = self._get_data(flag='val')
        _, test_loader = self._get_data(flag='test')

        ## ==== Training ====
        print('End-to-end fine-tuning......')
        best_model_path = os.path.join(self.args.root_path, self.args.model_save_path, self.args.task, 'bestloss/', self.setting,
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
            for i, (batch_x, batch_x_decomp, _) in enumerate(train_loader):
                num_samples += batch_x.shape[0]
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_x_decomp = batch_x_decomp.float().to(self.device)

                f_dim = -1 if self.args.features == 'MS' else 0
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(x=batch_x, x_decomp=batch_x_decomp)
                        outputs = outputs[:, :, f_dim:]
                        loss = criterion(outputs, batch_x)
                else:
                    outputs = self.model(x=batch_x, x_decomp=batch_x_decomp)
                    outputs = outputs[:, :, f_dim:]
                    loss = criterion(outputs, batch_x)
                
                train_loss += loss.item()

                if (i + 1) % 200 == 0:
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
    
    def test(self, save_result=False, load=False, in_train=False, setting_pred=None):
        _, test_loader = self._get_data(flag='test')
        _, train_loader = self._get_data(flag='train')

        if load:
            print('loading model...')
            state_dict = torch.load(
                os.path.join(self.args.root_path, self.args.model_save_path, 
                             self.args.task, 'bestloss/', self.setting, 
                             str(self.configs.data.context_points)+'-'+str(self.configs.data.target_points),
                             'checkpoint.pkl'))
            self.model.load_state_dict(state_dict)

        attens_energy = []
        folder_path = './results/AD_test_results/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        # (1) stastic on the train set
        print('stastic on the train set...')
        with torch.no_grad():
            for i, (batch_x, batch_x_decomp, _) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_decomp = batch_x_decomp.float().to(self.device)
                # reconstruction
                outputs = self.model(batch_x, batch_x_decomp)
                # criterion
                score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)
        print('Finish.')

        # (2) find the threshold
        print('find the threshold...')
        attens_energy = []
        test_labels = []
        for i, (batch_x, batch_x_decomp, batch_y) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            batch_x_decomp = batch_x_decomp.float().to(self.device)
            # reconstruction
            outputs = self.model(batch_x, batch_x_decomp)
            # criterion
            score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            test_labels.append(batch_y)
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        
        test_energy = np.array(attens_energy)
        
        # combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        combined_energy = test_energy

        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)
        print("gt:     ", gt.shape)

        # for ar in range(10*int(self.args.anomaly_ratio), 10*int(self.args.anomaly_ratio)+11):
        # for ar in range(1, 10):
        for ar in [self.args.anomaly_ratio]:
            # ar = 0.1*ar
            threshold = np.percentile(combined_energy, 100 - ar)
            print("anomaly ratio :", ar)
            print("Threshold :", threshold)

            # (3) evaluation on the test set
            print('evaluation on the test set...')
            pred = (test_energy > threshold).astype(int)
            
            print("pred:   ", pred.shape)

            # (4) detection adjustment
            print('detection adjustment...')
            gt, pred = adjustment(gt, pred)

            pred = np.array(pred)
            gt = np.array(gt)
            print("pred: ", pred.shape)
            print("gt:   ", gt.shape)

            accuracy = accuracy_score(gt, pred)
            precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
            print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                accuracy, precision,
                recall, f_score))

            f = open(folder_path+"result_anomaly_detection.txt", 'a')
            f.write(self.setting + "  \n")
            f.write("anomaly ratio = "+ str(ar) + " \n")
            f.write("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                accuracy, precision,
                recall, f_score))
            f.write('\n')
            f.write('\n')
            f.close()
        return