import argparse
import torch
import torch.backends
import random
import numpy as np
import pandas as pd

from exp.exp_anomaly_detection import Exp_Anomaly_Detection

if __name__ == '__main__':
    fix_seed = 2025
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='')

    # basic config
    parser.add_argument('--mode', type=str, required=True, default="train", help='status')
    parser.add_argument('--configs_path', type=str, default="./configs/", help='')
    parser.add_argument('--save_path', type=str, default='./test_results/', help='')

    # data
    parser.add_argument('--root_path', type=str, default='/workspace/datasets/anomaly_detect', help='root path of the data file')
    parser.add_argument('--data', type=str, default='MSL', help='dataset type')
    parser.add_argument('--data_origin', type=str, default='DADA', help='dataset type')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # 
    parser.add_argument('--method', type=str, default='spot', help='')
    parser.add_argument('--t', type=float, nargs='+', default=[0.1], help='')
    parser.add_argument('--metrics', type=str, nargs='+', default=['best_f1', 'auc', 'r_auc', 'vus'], help='')

    args = parser.parse_args()
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    setting = pd.read_csv(f"{args.configs_path}/UCR/UCR_setting.csv")
    datas = setting["name"].values
    model_setting_ids = setting["model_setting"].values

    for i in range(250):
        args.data = datas[i]

        # Exp
        exp = Exp_Anomaly_Detection(args, id=model_setting_ids[i])
        if args.mode == "train":
            exp.train()
            exp.evaluate_UCR_accomplish()
        elif args.mode == "test":
            exp.evaluate_UCR_accomplish()

        if args.gpu_type == 'mps':
            torch.backends.mps.empty_cache()
        elif args.gpu_type == 'cuda':
            torch.cuda.empty_cache()

   
