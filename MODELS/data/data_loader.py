# -*- coding: utf-8 -*-
"""
@file   : data_factory.py
@author : CHE THANH PHUOC MAI
@contact: ctpmai-cntt16@tdu.edu.vn
"""

import os
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils.IntervalScaler import IntervalScaler
from utils.timefeatures import time_features
import numpy as np
import warnings
from utils.augmentation import run_augmentation_single

warnings.filterwarnings('ignore')

class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='MS', data_path='iFile.csv',
                 target='Low,High', scale=False, timeenc=0, freq='m',
                    kfold=None,dataframe=None):
        # Default sequence length settings
        self.external_dataframe = dataframe

        self.args = args

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.flag = flag

        # Set type based on kfold and flag
        self.kfold = kfold
        if self.kfold > 0:
            assert self.flag in ['train', 'test']
            type_map = {'train': 0, 'test': 1}
        else:
            assert self.flag in ['train', 'val', 'test']
            type_map = {'train': 0, 'val': 1, 'test': 2}

        self.set_type = type_map[self.flag]

        self.features = features

        # Determine target interval
        if isinstance(target, str):
            self.target = [t.strip() for t in target.split(',')]
            # print(self.target)
        else:
            self.target = target

        # Scale the data if required
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        # Set the root and data paths
        self.root_path = root_path
        self.data_path = data_path

        self.__read_data__()

    def __read_data__(self):
        if self.external_dataframe is not None:
            df_raw = self.external_dataframe.copy()
        else:
            file_path = os.path.join(self.root_path, self.data_path)
            df_raw = pd.read_csv(file_path)


        cols = list(df_raw.columns)
        cols.remove('date')
        for t in self.target:
            if t in cols:
                cols.remove(t)
        df_raw = df_raw[['date'] + cols + self.target]
        self.df_target = df_raw[['date'] + self.target].copy()

        if self.features in ['M', 'MS']:
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # Handle kfold and set type
        if self.kfold == 0:
            n = len(df_raw)
            self.num_train = int(n * 0.7)
            self.num_test = int(n * 0.15)
            self.num_vali = n - self.num_train - self.num_test

            border1s = [0, self.num_train - self.seq_len, n - self.num_test - self.seq_len]
            border2s = [self.num_train, self.num_train + self.num_vali, n]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

            if self.scale:
                self.scaler = IntervalScaler()
                train_data = df_data[border1s[0]:border2s[0]]

                # Giả định 2 cột cuối là low và high
                low_train = train_data.iloc[:, -2].values
                high_train = train_data.iloc[:, -1].values
                self.scaler.fit(low_train, high_train)

                low_all = df_data.iloc[:, -2].values
                high_all = df_data.iloc[:, -1].values
                low_scaled, high_scaled = self.scaler.transform(low_all, high_all)

                # Gộp lại các cột khác với low_scaled, high_scaled
                data_others = df_data.iloc[:, :-2].values
                data = np.concatenate([data_others, low_scaled.reshape(-1, 1), high_scaled.reshape(-1, 1)], axis=1)
            else:
                data = df_data.values


        else:  # kfold > 0
            self.num_train = int(len(df_raw) * 0.8)
            self.num_test = len(df_raw) - self.num_train
            if self.flag == 'train':
                border1 = 0
                border2 = self.num_train
            else:  # 'test'
                border1 = len(df_raw) - self.num_test
                border2 = len(df_raw)


        # Normal cases: add data
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)

        if self.timeenc == 0:
            df_stamp['year'] = df_stamp.date.apply(lambda row: row.year, 1)
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            # df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            # df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            # df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values

        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        elif self.timeenc == 2:
            cycle_length = 5
            cycle = np.arange(len(df_raw)) % cycle_length
            df_raw['five_day_sin'] = np.sin(2 * np.pi * cycle / cycle_length)
            df_raw['five_day_cos'] = np.cos(2 * np.pi * cycle / cycle_length)
            data_stamp = df_raw[['five_day_sin', 'five_day_cos']].values[border1:border2]

        self.data_stamp = data_stamp

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)


    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    # def inverse_transform(self, data):
        # return self.scaler.inverse_transform(data)
    def inverse_transform(self, data):
        low_scaled = data[:, 0]
        high_scaled = data[:, 1]
        low, high = self.scaler.inverse_transform(low_scaled, high_scaled)
        return np.stack([low, high], axis=-1)




class Dataset_Pred(Dataset):
    def __init__(self,args, root_path, flag='pred', size=None, kfold = None,
                 features='S', data_path='ETTh1.csv',
                 target='Low,High', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        self.args = args
        self.kfold = kfold
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        # init
        assert flag in ['pred']

        self.features = features
        # Determine target interval
        if isinstance(target, str):
            self.target = [t.strip() for t in target.split(',')]
            # print(self.target)
        else:
            self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove('date')
        for t in self.target:
            if t in cols:
                cols.remove(t)
        df_raw = df_raw[['date'] + cols + self.target]

        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler = IntervalScaler()

            # Giả định 2 cột cuối là low và high
            low_train = df_data.iloc[:, -2].values
            high_train = df_data.iloc[:, -1].values
            self.scaler.fit(low_train, high_train)

            low_all = df_data.iloc[:, -2].values
            high_all = df_data.iloc[:, -1].values
            low_scaled, high_scaled = self.scaler.transform(low_all, high_all)

            # Gộp lại các cột khác với low_scaled, high_scaled
            data_others = df_data.iloc[:, :-2].values
            data = np.concatenate([data_others, low_scaled.reshape(-1, 1), high_scaled.reshape(-1, 1)], axis=1)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    # def inverse_transform(self, data):
    #     return self.scaler.inverse_transform(data)
    def inverse_transform(self, data):
        low_scaled = data[:, 0]
        high_scaled = data[:, 1]
        low, high = self.scaler.inverse_transform(low_scaled, high_scaled)
        return np.stack([low, high], axis=-1)
