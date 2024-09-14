# -*- coding: utf-8 -*- 
# @Time : 2024/8/12 21:44 
# @Author : DirtyBoy 
# @File : variation2hash.py
import os, shutil, hashlib

DST = '/home/lhd/APK'
if not os.path.isdir(DST):
    os.makedirs(DST)


def calculate_hashes(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


path = '/home/lhd/ADV_MVML/AVPASS/MalRadar_variation'
##'/mnt/local_sdc1/lhd/MalRadar_variation/   /home/lhd/ADV_MVML/AVPASS/MalRadar_variation'
for i in range(20):
    v = 'v' + str(i + 1)
    try:

        v_list = os.listdir(os.path.join(path, v))
        print(v, '  ', len(v_list))
        for item in v_list:
            sha256 = calculate_hashes(os.path.join(os.path.join(path, v), item))
            shutil.copy(src=os.path.join(os.path.join(path, v), item),
                        dst=os.path.join(DST, sha256 + '.apk'))
            print(sha256)
    except Exception:
        print(v, ' folder not found')
