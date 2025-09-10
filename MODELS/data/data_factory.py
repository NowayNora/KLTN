# -*- coding: utf-8 -*-
"""
@file   : data_factory.py
@author : CHE THANH PHUOC MAI
@contact: ctpmai-cntt16@tdu.edu.vn
"""

from data.data_loader import Dataset_Custom, Dataset_Pred
from torch.utils.data import DataLoader

data_dict = {
    'custom': Dataset_Custom, # Custom dataset for user-defined data
}


def data_provider(args, flag):
    """
    Data provider function to create a dataset and data loader based on the provided arguments and flag.
    Args:
        args: Argument parser containing dataset parameters.
        flag: A string indicating the type of data to load ('train', 'val', 'test', 'pred').
    Returns:
        data_set: An instance of the dataset class.
        data_loader: A DataLoader instance for batching and loading the dataset.
    """
    Data = data_dict[args.data]
    # Determine time encoding based on the embedding type
    if args.embed == 'monthSine':
        timeenc = 2
    elif args.embed == 'timeF':
        timeenc = 1
    else:
        timeenc = 0
    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    freq = args.freq
    
    # Set parameters based on the flag
    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else: # 'train', 'val', 'TRAIN', 'VAL'
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    # Create the dataset instance
    data_set = Data(
        args = args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        kfold=args.kfold,
        scale=args.scale,
                )

    # Create the DataLoader instance
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)

    
    return data_set, data_loader