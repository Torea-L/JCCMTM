from data_provider.data_loader_LTSF import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Solar, Dataset_PEMS
from data_provider.data_loader_AD import PSMSegLoader, MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Solar': Dataset_Solar,
    'PEMS': Dataset_PEMS,
    'custom': Dataset_Custom,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWaT': SWATSegLoader,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    # print(f'args.data: {args.data}, Data: {Data}')
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        '''if args.task == 'AD':
            drop_last = True
        else:
            drop_last = False'''
        batch_size = args.test_batch_size
        freq = args.freq
    elif flag == 'val':
        shuffle_flag = False
        drop_last = False
        batch_size = args.test_batch_size
        freq = args.freq
    # elif flag == 'pred':
    #     shuffle_flag = False
    #     drop_last = False
    #     batch_size = 1
    #     freq = args.freq
    #     Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    if args.task == 'AD':
        data_set =  Data(
            root_path=args.root_path,
            data_path=args.data_path,
            win_size=args.seq_len,
            flag=flag,
            decomposition=args.decomposition,
            composition=args.component, kernel_size=args.kernel_size
        )
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )
    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            decomposition=args.decomposition, 
            composition=args.component, kernel_size=args.kernel_size
            )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
            )

    return data_set, data_loader
