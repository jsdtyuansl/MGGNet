import os

import pandas as pd


def generate_score(path):
    # 生成v2016的所有csv
    with open(path, 'rb') as f:
        lines = f.read().decode().strip().split('\n')
    k = []
    v = []
    for line in lines:
        if "//" in line:
            temp = line.split()
            k.append(temp[0])
            v.append(float(temp[3]))  # 0为蛋白code  3为亲和数据
    data_dict = {'PDB_code': k, '-logKd/Ki': v}
    df = pd.DataFrame(data_dict)
    df.to_csv('./data/score_total.csv', index=False)


def dir_to_csv(dataset_path, set_name):
    df = pd.read_csv('./data/score_total.csv')
    res = {}
    for row in df.itertuples():
        res[row[1]] = row[2]
    k = []
    v = []
    id_list = [x for x in os.listdir(dataset_path) if len(x) == 4]
    for item in id_list:
        k.append(item)
        v.append(res[item])
    data_dict = {'PDB_code': k, '-logKd/Ki': v}
    df = pd.DataFrame(data_dict)
    df.to_csv(f'./data/{set_name}.csv', index=False)


if __name__ == '__main__':
    score_path = './data/INDEX_general_PL_data.2016'
    generate_score(score_path)

    total_dir = './data/total_set'
    dir_to_csv(total_dir, 'total_set')

    train_dir = './data/train'
    dir_to_csv(train_dir, 'train')

    val_dir = './data/valid'
    dir_to_csv(val_dir, 'valid')

    # test2013_dir = './data/test2013'
    # dir_to_csv(test2013_dir, 'test2013')
    #
    # test2016_dir = './data/test2016'
    # dir_to_csv(test2016_dir, 'test2016')


