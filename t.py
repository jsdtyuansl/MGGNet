import torch
import pandas as pd
import os
import pickle

if __name__ == '__main__':

    # df1 = pd.read_csv('./data/train.csv')
    # res1 = {}
    # for row in df1.itertuples():
    #     res1[row[1]] = row[2]
    #
    # df2 = pd.read_csv('./data/valid.csv')
    # res2 = {}
    # for row in df2.itertuples():
    #     res2[row[1]] = row[2]
    #
    # dataset_path = './data/total_set'
    # id_list = [x for x in os.listdir(dataset_path) if len(x) == 4]
    # all_list = []
    #
    # keys_union = set(res1.keys()) | set(res2.keys())
    #
    # # 将集合转换为列表
    # keys_list = list(keys_union)
    # id_set = set(id_list)
    #
    # # {'4xkc', '1k2v', '2zjw', '1bjr', '4m3b', '2foy', '1ai6', '1esz', '4ezr'} id_list有
    # # {'4zwy', '5ai1', '4zwz', '1yw2', '1sqi', '1oxg', '2zb0', '4qf7', '4jjg', '5ah2', '1f8a', '2pg2', '3hwn', '2avq'} GIGN的total有
    #
    # difference = keys_union.difference(id_set)
    #
    #
    with open("./data/processed/test2016/1a30/1a30_5A.pkl","rb") as f:
        data = pickle.load(f)


    print(data[2])


