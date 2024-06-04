# -*- coding: utf-8 -*- 
# @Time : 2024/3/15 21:12 
# @Author : DirtyBoy 
# @File : get_sha256.py
import hashlib, os
import pandas as pd


def calculate_apk_sha256(file_path):
    # 创建 SHA-256 散列对象
    sha256_hash = hashlib.sha256()

    # 以二进制只读模式打开文件
    with open(file_path, "rb") as f:
        # 逐块读取文件内容并更新散列值
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)

    # 返回计算得到的 SHA-256 散列值的十六进制表示
    return sha256_hash.hexdigest()


# APK 文件路径
apk_folder_path = "/home/public/rmt/heren/experiment/cl-exp/LHD_apk/CICDataset/source_apk/benign/benign/"
for item in ['Benign_2016', 'Benign_2015', 'Benign_2017']:
    data_df = pd.DataFrame()
    path = os.path.join(apk_folder_path, item)
    benign_list = os.listdir(path)
    hash_256 = []
    data_df['source_name'] = benign_list
    num = 0
    for demo in benign_list:
        apk_sha256 = calculate_apk_sha256(os.path.join(path, demo))
        num = num + 1
        hash_256.append(apk_sha256)
        print(item, num, '---', apk_sha256)
    data_df['sha256'] = hash_256
    data_df.to_csv(item + '.csv')
