import os
import subprocess
import pandas as pd
import numpy as np

if __name__ == '__main__':
    # ---------------- 配置区域 ----------------
    path_to_crossad = r'D:\Desktop\研究生学习\代码\CrossAD-dev_tsb-ad'
    dataset_root = r'D:\Desktop\研究生学习\代码\数据集'

    # 修改这里选择数据集
    dataset_name = 'Boiler'

    # ----------------------------------------

    label_csv_path = ""
    data_files_dir = ""
    sources = []

    if dataset_name == 'SMD':
        data_files_dir = os.path.join(dataset_root, 'SMD', 'test')
        sources = ['1-1']
    elif dataset_name == 'MSL':
        label_csv_path = os.path.join(dataset_root, 'MSL_SMAP', 'labeled_anomalies.csv')
        data_files_dir = os.path.join(dataset_root, 'MSL', 'test')
        sources = ['D-14']
    elif dataset_name == 'SMAP':
        label_csv_path = os.path.join(dataset_root, 'MSL_SMAP', 'labeled_anomalies.csv')
        data_files_dir = os.path.join(dataset_root, 'SMAP', 'test')
        sources = ['D-7']
    elif dataset_name == 'Boiler':
        data_files_dir = os.path.join(dataset_root, 'Boiler')
        sources = ['1']
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    files = []

    # 1. Boiler 逻辑 (修改：查找包含 test.csv 的目录)
    if dataset_name == 'Boiler':
        if os.path.exists(data_files_dir):
            print(f"Listing entities from {data_files_dir} (checking for test.csv)...")
            all_items = os.listdir(data_files_dir)
            files = []
            for item in all_items:
                item_path = os.path.join(data_files_dir, item)
                # 检查是否是目录，且内部包含 test.csv (对应目标域数据)
                if os.path.isdir(item_path):
                    if os.path.exists(os.path.join(item_path, 'test.csv')):
                        files.append(item)
                    elif os.path.exists(os.path.join(item_path, 'train.csv')):
                        # 兼容只有 train.csv 的情况，但优先 test
                        files.append(item)

            try:
                files.sort(key=lambda x: int(x))
            except ValueError:
                files.sort()
        else:
            print(f"[Error] Data directory not found: {data_files_dir}")
            exit(1)

    # 2. MSL / SMAP 逻辑 (保持不变)
    elif dataset_name in ['MSL', 'SMAP'] and os.path.exists(label_csv_path):
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

    # 3. SMD 逻辑 (保持不变)
    else:
        if not os.path.exists(data_files_dir):
            print(f"[Error] Data directory not found: {data_files_dir}")
            exit(1)
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
        exit(0)

    print(f"Found {len(files)} entities in {dataset_name}: {files[:5]} ...")

    # 循环执行 (Source -> Target)
    for src in sources:
        if dataset_name == 'Boiler' and src not in files:
            # 如果源域只有train没有test，可能不会在files里，这里稍微放宽或打印警告
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