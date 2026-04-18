import os
import subprocess
import pandas as pd
import numpy as np

if __name__ == '__main__':
    # ---------------- 配置区域 ----------------
    path_to_crossad = r'.'
    dataset_root = r'../datasets'

    # 将这里改成一个列表，按顺序放入你想跑的数据集
    datasets_to_run = ['MSL', 'SMAP', 'SMD']

    # ----------------------------------------

    for dataset_name in datasets_to_run:
        print(f"\n{'*' * 60}")
        print(f"🚀 开始执行数据集总循环: {dataset_name}")
        print(f"{'*' * 60}\n")

        label_csv_path = ""
        data_files_dir = ""
        sources = []

        if dataset_name == 'SMD':
            data_files_dir = os.path.join(dataset_root, 'SMD', 'test')
            sources = ['1-1', '2-3', '3-7', '1-5']
        elif dataset_name == 'MSL':
            label_csv_path = os.path.join(dataset_root, 'MSL_SMAP', 'labeled_anomalies.csv')
            data_files_dir = os.path.join(dataset_root, 'MSL', 'test')
            sources = ['F-5', 'C-1', 'D-14', 'P-10']
        elif dataset_name == 'SMAP':
            label_csv_path = os.path.join(dataset_root, 'MSL_SMAP', 'labeled_anomalies.csv')
            data_files_dir = os.path.join(dataset_root, 'SMAP', 'test')
            sources = ['A-7', 'P-2', 'E-8', 'D-7']
        elif dataset_name == 'Boiler':
            data_files_dir = os.path.join(dataset_root, 'Boiler')
            sources = ['1', '2', '3']
        else:
            print(f"[错误] 未知的数据集跳过: {dataset_name}")
            continue

        files = []

        # MSL / SMAP 逻辑
        if dataset_name in ['MSL', 'SMAP'] and os.path.exists(label_csv_path):
            print(f"Loading entity list from {label_csv_path}...")
            with open(label_csv_path, 'r') as file:
                csv_reader = pd.read_csv(file, delimiter=',')
            data_info = csv_reader[csv_reader['spacecraft'] == dataset_name]
            space_files = np.asarray(data_info['chan_id'])

            if os.path.exists(data_files_dir):
                all_files = os.listdir(data_files_dir)
                all_names = [name.split('.')[0] for name in all_files if name.endswith(('.npy', '.csv'))]
                files = [file for file in all_names if file in space_files]
            else:
                files = space_files.tolist()

            files.sort()

        # SMD / Boiler 逻辑
        else:
            if not os.path.exists(data_files_dir):
                print(f"[Error] Data directory not found: {data_files_dir}")
                continue
            print(f"Listing files from {data_files_dir}...")
            all_files = os.listdir(data_files_dir)
            valid_extensions = ('.npy', '.csv', '.txt')
            for name in all_files:
                if not name.endswith(valid_extensions):
                    continue
                base_name = os.path.splitext(name)[0]
                if dataset_name == 'SMD' and base_name.startswith('machine-'):
                    clean_name = base_name.replace('machine-', '')
                else:
                    clean_name = base_name
                files.append(clean_name)

            files = sorted(list(set(files)))

        if len(files) == 0:
            print(f"[Warning] Found 0 entities in {dataset_name} directory: {data_files_dir}")
            continue

        print(f"Found {len(files)} entities in {dataset_name}: {files[:5]} ...")

        # 循环执行 (Source -> Target)
        for src in sources:
            if dataset_name == 'Boiler' and src not in files:
                pass

            for trg in files:
                if src == trg:
                    continue

                print(f'\n==================================================')
                print(f'Running CrossAD UDA ({dataset_name}): Source [{src}] -> Target [{trg}]')
                print(f'==================================================\n')

                run_script = os.path.join(path_to_crossad, 'run.py')
                command = [
                    'python', run_script,
                    '--mode', 'train',
                    '--root_path', dataset_root,
                    '--data', dataset_name,
                    '--id_src', src,
                    '--id_trg', trg,
                    '--data_origin', 'DADA',
                    '--gpu', '0'
                ]
                try:
                    subprocess.run(command, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error occurred while running {src} -> {trg}")
                    continue

        print(f"✅ 数据集 {dataset_name} 跨域实验全部完成！\n")