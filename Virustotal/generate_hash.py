# -*- coding: utf-8 -*- 
# @Time : 2024/7/22 9:24 
# @Author : DirtyBoy 
# @File : generate_hash.py
import hashlib, os
import pandas as pd
from tqdm import tqdm

malware_dir_path = '/home/public/rmt/heren/MalRadar-all'
variation_dir_path = '/home/public/rmt/heren/experiment/cl-exp/LHD_apk/Lable_noise/MalRadar_variation'

def calculate_hashes(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


if __name__ == '__main__':
    source_apk_name = os.listdir(malware_dir_path)
    source_apk_name = [item for item in source_apk_name if os.path.splitext(item)[1] == '.apk']
    source_apk_name = [os.path.splitext(item)[0] for item in source_apk_name]

    data = pd.DataFrame()
    data['source_malware'] = source_apk_name

    for i in range(20):
        variation = 'v' + str(i + 1)
        variation_path = os.path.join(variation_dir_path, variation)
        if os.path.isdir(variation_path):
            variation_apk_list = os.listdir(variation_path)
            variation_apk_list = [file for file in variation_apk_list if file.endswith('.apk')]
            variation_apk_list = [os.path.splitext(item)[0] for item in variation_apk_list]
            hash_list = [None] * len(source_apk_name)
            for variation_apk in tqdm(variation_apk_list, desc="Calculate Hashes " + variation):
                variation_apk_path = os.path.join(variation_path, variation_apk + '.apk')
                sha256 = calculate_hashes(variation_apk_path)
                try:
                    index = source_apk_name.index(variation_apk)
                    hash_list[index] = sha256
                except ValueError:
                    pass
            hash_name = variation
            data[hash_name] = hash_list
    data.to_csv('variation_hashes.csv')
