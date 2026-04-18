import pandas as pd
import numpy as np
import torch
import random, argparse, time, os, logging
from sklearn.preprocessing import MinMaxScaler
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank

from exp.exp_TSBAD import Exp_Anomaly_Detection

# seeding
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print("CUDA available: ", torch.cuda.is_available())
print("cuDNN version: ", torch.backends.cudnn.version())




def run_AMCAD(data_train, data_test, args, id):
    clf = Exp_Anomaly_Detection(args, id)

    # 训练模型 (使用 data_train)
    clf.train(data_train)

    # [修改] 严格切分测试集
    # 如果 data_test 包含了 data_train (即传入的是全量数据)，我们需要切除训练部分
    # 只有在 train_index 之后的数据才是真正的测试数据

    # 判断是否重叠（根据长度判断）
    if len(data_test) > len(data_train):
        # 仅对未训练部分进行测试
        real_test_data = data_test[len(data_train):]
    else:
        # 如果传入的已经是切分好的测试集
        real_test_data = data_test

    # 运行测试
    score = clf.test(real_test_data)

    # 归一化得分
    score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()

    # [注意] 返回的 score 长度变短了，需要在主循环中同步截断 label
    return score


if __name__ == '__main__':
    Start_T = time.time()
    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Generating Anomaly Score')
    parser.add_argument('--dataset_dir', type=str, default='./dataset/TSB-AD/TSB-AD-U/')
    parser.add_argument('--file_lsit', type=str, default='./dataset/TSB-AD/File_List/TSB-AD-U-Eva.csv')
    parser.add_argument('--score_dir', type=str, default='./test_results/TSB-AD-U/score/')
    parser.add_argument('--save_dir', type=str, default='./test_results/TSB-AD-U/')
    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument('--AD_Name', type=str, default='CrossAD')

    # basic config
    parser.add_argument('--configs_path', type=str, default="./configs/TSB-AD-U/", help='')

    parser.add_argument('--data', type=str, default='', help='dataset type')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints_(TSB-AD-U)', help='')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    args = parser.parse_args()

    df = pd.read_csv(f"{args.configs_path}/TSB-AD-U_setting_draft.csv")
    setting = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))

    target_dir = os.path.join(args.score_dir, args.AD_Name)
    os.makedirs(target_dir, exist_ok=True)
    logging.basicConfig(filename=f'{target_dir}/000_run_{args.AD_Name}.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    file_list = pd.read_csv(args.file_lsit)['file_name'].values

    write_csv = []
    for filename in file_list:
        if os.path.exists(target_dir + '/' + filename.split('.')[0] + '.npy'): continue
        print('Processing:{} by {}'.format(filename, args.AD_Name))

        file_path = os.path.join(args.dataset_dir, filename)
        df = pd.read_csv(file_path).dropna()
        data = df.iloc[:, 0:-1].values.astype(float)
        label = df['Label'].astype(int).to_numpy()

        feats = data.shape[1]
        slidingWindow = find_length_rank(data[:, 0].reshape(-1, 1), rank=1)
        train_index = filename.split('.')[0].split('_')[-3]
        data_train = data[:int(train_index), :]

        start_time = time.time()

        args.data = filename
        model_id = setting[filename]

        Optimal_Det_HP = {'args': args, 'id': model_id}

        # 运行 CrossAD
        # 注意：这里假设你已经修改了 run_CrossAD 函数，使其只返回测试集的 score (切除了训练部分)
        output = run_AMCAD(data_train, data, **Optimal_Det_HP)

        end_time = time.time()
        run_time = end_time - start_time

        if isinstance(output, np.ndarray):
            logging.info(
                f'Success at {filename} using {args.AD_Name} | Time cost: {run_time:.3f}s at length {len(label)}')
            np.save(target_dir + '/' + filename.split('.')[0] + '.npy', output)
        else:
            logging.error(f'At {filename}: ' + output)

        ### whether to save the evaluation result
        if args.save:
            try:
                # [修改关键点] 对齐 Label 和 Output
                # 因为 output 已经被截断（去除了训练集部分），所以 label 也要取对应的后半部分
                # 使用 -len(output): 可以自动适配
                test_label = label[-len(output):]

                # 再次检查长度，防止空输出等异常
                if len(test_label) != len(output):
                    print(f"[ERROR] Length mismatch: label {len(test_label)} vs output {len(output)}")

                evaluation_result = get_metrics(output, test_label, slidingWindow=slidingWindow)
                print('evaluation_result: ', evaluation_result)
                list_w = list(evaluation_result.values())
            except Exception as e:
                print(f"[Error in Evaluation] {e}")
                list_w = [0] * 9

            list_w.insert(0, run_time)
            list_w.insert(0, filename)
            write_csv.append(list_w)

            ## Temp Save
            try:
                col_w = list(evaluation_result.keys())
                col_w.insert(0, 'Time')
                col_w.insert(0, 'file')
                w_csv = pd.DataFrame(write_csv, columns=col_w)
                os.makedirs(args.save_dir, exist_ok=True)
                w_csv.to_csv(f'{args.save_dir}/{args.AD_Name}.csv', index=False)
            except:
                pass