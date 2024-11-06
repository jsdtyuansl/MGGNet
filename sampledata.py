import os
import random
import shutil
import pandas as pd


def remove(folder_a, folder_b):
    subfolders_b = {f for f in os.listdir(folder_b) if os.path.isdir(os.path.join(folder_b, f))}
    same_name_subfolders = {f for f in os.listdir(folder_a) if
                            f in subfolders_b and os.path.isdir(os.path.join(folder_a, f))}
    print(f"将要删除 {len(same_name_subfolders)} 个重复的子文件夹")

    # 删除a文件夹中的重复子文件夹
    for folder in same_name_subfolders:
        folder_path = os.path.join(folder_a, folder)
        try:
            shutil.rmtree(folder_path)
        except OSError as e:
            print(f"remove {folder} 时出错: {e}")
    print("删除操作完成---")


if __name__ == '__main__':
    # total中去除与test2016重复的文件
    folder_a = './data/total_set'
    folder_b = './data/test2016'
    remove(folder_a, folder_b)

    folder_a = './data/test2013'
    folder_b = './data/total_set'
    remove(folder_a, folder_b)

    # # 设置源文件夹路径
    # source_folder = './data/total_set/'
    #
    # # 设置目标文件夹路径
    # selected_target_folder = './data/val_set/'
    # remaining_target_folder = './data/train_set/'
    #
    # os.makedirs(selected_target_folder, exist_ok=True)
    # os.makedirs(remaining_target_folder, exist_ok=True)
    #
    # sub_folders = [f for f in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, f))]
    #
    # if len(sub_folders) < 1000:
    #     print("子文件夹数量不足1000个，无法完成抽取。")
    # else:
    #     # 随机选择1000个子文件夹
    #     selected_folders = random.sample(sub_folders, 1000)
    #     remaining_folders = [f for f in sub_folders if f not in selected_folders]
    #
    #     # 定义复制子文件夹的函数
    #     def copy_subfolder(subfolder_name, source_path, target_path):
    #         # 构建源子文件夹和目标子文件夹的完整路径
    #         source_subfolder_path = os.path.join(source_path, subfolder_name)
    #         target_subfolder_path = os.path.join(target_path, subfolder_name)
    #         # 如果目标子文件夹不存在，则创建它
    #         if not os.path.exists(target_subfolder_path):
    #             os.makedirs(target_subfolder_path)
    #         # 复制子文件夹中的所有内容到目标子文件夹
    #         for item in os.listdir(source_subfolder_path):
    #             s = os.path.join(source_subfolder_path, item)
    #             d = os.path.join(target_subfolder_path, item)
    #             if os.path.isdir(s):
    #                 shutil.copytree(s, d, dirs_exist_ok=True)
    #             else:
    #                 shutil.copy2(s, d)
    #
    #     # 复制选中的子文件夹到目标文件夹
    #     for folder in selected_folders:
    #         copy_subfolder(folder, source_folder, selected_target_folder)
    #
    #     # 复制剩下的子文件夹到另一个目标文件夹
    #     for folder in remaining_folders:
    #         copy_subfolder(folder, source_folder, remaining_target_folder)
    # print("抽取完成---")