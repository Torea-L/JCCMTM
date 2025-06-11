import os
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from layers.basics import raw_series_decomp

warnings.filterwarnings('ignore')

class PSMSegLoader(Dataset):
    def __init__(self, root_path, data_path, win_size, 
                 decomposition=True, composition='trend', kernel_size=7, stride=1,
                 step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size

        self.root_path = root_path
        self.data_path = data_path
        self.decomposition = decomposition
        self.composition = composition
        if decomposition:
            self.decomp = raw_series_decomp(kernel_size=kernel_size, stride=stride)
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        # train data
        data = pd.read_csv(os.path.join(self.root_path, self.data_path, 'train.csv'))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        self.train = data
        # test data
        test_data = pd.read_csv(os.path.join(self.root_path, self.data_path, 'test.csv'))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        test_data = self.scaler.transform(test_data)
        self.test = test_data
        # test label
        self.test_labels = pd.read_csv(os.path.join(self.root_path, self.data_path, 'test_label.csv')).values[:, 1:]
        # series decomposition
        if self.decomposition:
            data_res, data_trend = self.decomp(torch.from_numpy(data))
            data_res_test, data_trend_test = self.decomp(torch.from_numpy(test_data))
            if self.composition=='trend':
                self.test_decomp = data_trend_test
                self.train_decomp = data_trend
            else:
                self.test_decomp = data_res_test
                self.train_decomp = data_res
        else:
            self.test_decomp = self.test
            self.train_decomp = self.train
        # self.val = self.test
        # self.val_decomp = self.test_decomp
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.val_decomp = self.train_decomp[(int)(data_len * 0.8):]
        
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.train_decomp[index:index + self.win_size]), \
                np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.val_decomp[index:index + self.win_size]), \
                np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(self.test_decomp[index:index + self.win_size]), \
                np.float32(self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), \
                np.float32(self.test_decomp[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), \
                np.float32(self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
        
class MSLSegLoader(Dataset):
    def __init__(self, root_path, data_path, win_size, 
                 decomposition=True, composition='trend', kernel_size=7, stride=1,
                 step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size

        self.root_path = root_path
        self.data_path = data_path
        self.decomposition = decomposition
        self.composition = composition
        if decomposition:
            self.decomp = raw_series_decomp(kernel_size=kernel_size, stride=stride)
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        # train data
        data = np.load(os.path.join(self.root_path, self.data_path, "MSL_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        self.train = data
        # test data
        test_data = np.load(os.path.join(self.root_path, self.data_path, "MSL_test.npy"))
        test_data = self.scaler.transform(test_data)
        self.test = test_data
        # test label
        self.test_labels = np.load(os.path.join(self.root_path, self.data_path, "MSL_test_label.npy"))
        # series decomposition
        if self.decomposition:
            data_res, data_trend = self.decomp(torch.from_numpy(data))
            data_res_test, data_trend_test = self.decomp(torch.from_numpy(test_data))
            if self.composition=='trend':
                self.test_decomp = data_trend_test
                self.train_decomp = data_trend
            else:
                self.test_decomp = data_res_test
                self.train_decomp = data_res
        else:
            self.test_decomp = self.test
            self.train_decomp = self.train
        # self.val = self.test
        # self.val_decomp = self.test_decomp
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.val_decomp = self.train_decomp[(int)(data_len * 0.8):]
        
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.train_decomp[index:index + self.win_size]), \
                np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.val_decomp[index:index + self.win_size]), \
                np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(self.test_decomp[index:index + self.win_size]), \
                np.float32(self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), \
                np.float32(self.test_decomp[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), \
                np.float32(self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
        
class SMAPSegLoader(Dataset):
    def __init__(self, root_path, data_path, win_size, 
                 decomposition=True, composition='trend', kernel_size=7, stride=1,
                 step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size

        self.root_path = root_path
        self.data_path = data_path
        self.decomposition = decomposition
        self.composition = composition
        if decomposition:
            self.decomp = raw_series_decomp(kernel_size=kernel_size, stride=stride)
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        # train data
        data = np.load(os.path.join(self.root_path, self.data_path, "SMAP_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        self.train = data
        # test data
        test_data = np.load(os.path.join(self.root_path, self.data_path, "SMAP_test.npy"))
        test_data = self.scaler.transform(test_data)
        self.test = test_data
        # test label
        self.test_labels = np.load(os.path.join(self.root_path, self.data_path, "SMAP_test_label.npy"))
        # series decomposition
        if self.decomposition:
            data_res, data_trend = self.decomp(torch.from_numpy(data))
            data_res_test, data_trend_test = self.decomp(torch.from_numpy(test_data))
            if self.composition=='trend':
                self.test_decomp = data_trend_test
                self.train_decomp = data_trend
            else:
                self.test_decomp = data_res_test
                self.train_decomp = data_res
        else:
            self.test_decomp = self.test
            self.train_decomp = self.train
        # self.val = self.test
        # self.val_decomp = self.test_decomp
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.val_decomp = self.train_decomp[(int)(data_len * 0.8):]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.train_decomp[index:index + self.win_size]), \
                np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.val_decomp[index:index + self.win_size]), \
                np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(self.test_decomp[index:index + self.win_size]), \
                np.float32(self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), \
                np.float32(self.test_decomp[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), \
                np.float32(self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
        
class SMDSegLoader(Dataset):
    def __init__(self, root_path, data_path, win_size, 
                 decomposition=True, composition='trend', kernel_size=7, stride=1,
                 step=100, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size

        self.root_path = root_path
        self.data_path = data_path
        self.decomposition = decomposition
        self.composition = composition
        if decomposition:
            self.decomp = raw_series_decomp(kernel_size=kernel_size, stride=stride)
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        # train data
        data = np.load(os.path.join(self.root_path, self.data_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        self.train = data
        # test data
        test_data = np.load(os.path.join(self.root_path, self.data_path, "SMD_test.npy"))
        test_data = self.scaler.transform(test_data)
        self.test = test_data
        # test label
        self.test_labels = np.load(os.path.join(self.root_path, self.data_path, "SMD_test_label.npy"))
        # series decomposition
        if self.decomposition:
            data_res, data_trend = self.decomp(torch.from_numpy(data))
            data_res_test, data_trend_test = self.decomp(torch.from_numpy(test_data))
            if self.composition=='trend':
                self.test_decomp = data_trend_test
                self.train_decomp = data_trend
            else:
                self.test_decomp = data_res_test
                self.train_decomp = data_res
        else:
            self.test_decomp = self.test
            self.train_decomp = self.train
        data_len = len(self.train)
        # self.val = self.test
        # self.val_decomp = self.test_decomp
        self.val = self.train[(int)(data_len * 0.8):]
        self.val_decomp = self.train_decomp[(int)(data_len * 0.8):]
        # if self.decomposition:
        #     self.val_decomp = self.train_decomp[(int)(data_len * 0.8):]
        # else:
        #     self.val_decomp = self.train[(int)(data_len * 0.8):]
        
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.train_decomp[index:index + self.win_size]), \
                np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.val_decomp[index:index + self.win_size]), \
                np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(self.test_decomp[index:index + self.win_size]), \
                np.float32(self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), \
                np.float32(self.test_decomp[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), \
                np.float32(self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

class SWATSegLoader(Dataset):
    def __init__(self, root_path, data_path, win_size, 
                 decomposition=True, composition='trend', kernel_size=7, stride=1,
                 step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size

        self.root_path = root_path
        self.data_path = data_path
        self.decomposition = decomposition
        self.composition = composition
        if decomposition:
            self.decomp = raw_series_decomp(kernel_size=kernel_size, stride=stride)
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        # train data
        train_data = pd.read_csv(os.path.join(self.root_path, self.data_path, "swat_train2.csv"))
        train_data = np.array(train_data.values[:, :-1])
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        self.train = train_data
        # test data
        test_data = pd.read_csv(os.path.join(self.root_path, self.data_path, "swat2.csv"))
        labels = np.array(test_data.values[:, -1:])
        test_data = np.array(test_data.values[:, :-1])
        test_data = self.scaler.transform(test_data)
        self.test = test_data
        # test label
        self.test_labels = labels
        # series decomposition
        if self.decomposition:
            data_res, data_trend = self.decomp(torch.from_numpy(train_data))
            data_res_test, data_trend_test = self.decomp(torch.from_numpy(test_data))
            if self.composition=='trend':
                self.test_decomp = data_trend_test
                self.train_decomp = data_trend
            else:
                self.test_decomp = data_res_test
                self.train_decomp = data_res
        else:
            self.test_decomp = self.test
            self.train_decomp = self.train
        self.val = self.test
        self.val_decomp = self.test_decomp
        
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.train_decomp[index:index + self.win_size]), \
                np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.val_decomp[index:index + self.win_size]), \
                np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(self.test_decomp[index:index + self.win_size]), \
                np.float32(self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), \
                np.float32(self.test_decomp[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), \
                np.float32(self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
