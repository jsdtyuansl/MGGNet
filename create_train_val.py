import os
import shutil
import pandas as pd

def getset(source_path, dest_path, list):
    os.makedirs(dest_path, exist_ok=True)
    # 遍历文件夹列表，复制每个文件夹
    for folder_name in list:
        # 构造完整的源文件夹和目标文件夹路径
        source_folder = os.path.join(source_path, folder_name)
        dest_folder = os.path.join(dest_path, folder_name)

        if os.path.exists(source_folder):
            shutil.copytree(source_folder, dest_folder, dirs_exist_ok=True)

        else:
            print(f'The folder {source_folder} does not exist, skipping...')

if __name__ == '__main__':
    # # 原始CSV文件路径
    # original_csv_path = './data/total_set.csv'
    # df = pd.read_csv(original_csv_path)
    #
    # # 打乱数据索引
    # shuffled_indices = np.random.permutation(df.index)
    # shuffled_df = df.loc[shuffled_indices]
    #
    # split_row = 1000
    # df_new = shuffled_df[:split_row]  # 新的CSV文件，包含1000个随机数据
    # df_remain = shuffled_df[split_row:]  # 剩余数据组成的大CSV文件
    #
    # # 新CSV文件的路径remain_csv_path
    # val_csv_path = './data/train.csv'
    # train_csv_path = './data/valid.csv'
    #
    # # 写入新的CSV文件
    # df_new.to_csv(val_csv_path, index=False)
    # df_remain.to_csv(train_csv_path, index=False)

    df1 = pd.read_csv('./data/train.csv')
    res1 = {}
    for row in df1.itertuples():
        res1[row[1]] = row[2]

    df2 = pd.read_csv('./data/valid.csv')
    res2 = {}
    for row in df2.itertuples():
        res2[row[1]] = row[2]

    list1 = list(res1.keys())
    list2 = list(res2.keys())

    # 假设源文件夹路径和目标文件夹路径
    source_path = './data/total_set'
    dest_path1 = './data/train'
    dest_path2 = './data/valid'

    getset(source_path=source_path, dest_path=dest_path1, list=list1)
    getset(source_path=source_path, dest_path=dest_path2, list=list2)
