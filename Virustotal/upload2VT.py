# -*- coding: utf-8 -*- 
# @Time : 2024/8/9 9:03 
# @Author : DirtyBoy 
# @File : upload2VT.py
import requests, hashlib
import json, os
from tqdm import tqdm
logger_file = 'Upload_log/log1.txt'

url = 'https://www.virustotal.com/api/v3/files'
headers = {
    'x-apikey': '',
}


def save_to_txt(goal, txt_path):
    f = open(txt_path, "w")
    for line in goal:
        f.write(line + '\n')
    f.close()


def txt_to_list(txt_path):
    f = open(txt_path, "r")
    return f.read().splitlines()


def upload_file(file_path):
    files = {'file': (file_path, open(file_path, 'rb'))}
    response = requests.post(url, files=files, headers=headers)
    print(response)
    if response.status_code == 200:
        return 'success'
    else:
        return 'fail'


def calculate_hashes(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


if __name__ == '__main__':

    for j in range(19,20):
        variation = 'v' + str(j + 1)
        variation_path = os.path.join('/home/lhd/ADV_MVML/AVPASS/MalRadar_variation/',
                                      variation)  # /home/lhd/ADV_MVML/AVPASS/MalRadar_variation   /mnt/local_sdc1/lhd/MalRadar_variation/
        API_list = txt_to_list('vt_keys.txt')
        if not os.path.isfile(logger_file):
            save_to_txt([], logger_file)
        uploaded = txt_to_list(logger_file)
        try:
            sha256 = os.listdir(variation_path)
            var_sha256 = [calculate_hashes(os.path.join(variation_path, item)) + '.apk' for item in sha256]
            need_sha256 = []
            for i, file_sha256 in enumerate(var_sha256):
                if file_sha256.split('.')[0] not in uploaded:
                    need_sha256.append(sha256[i])
            for i, file_sha256 in tqdm(enumerate(need_sha256), total=len(need_sha256),
                                       desc='VirusTotal report ' + variation):
                headers['x-apikey'] = API_list[int(i % len(API_list))]
                ending = upload_file(os.path.join(variation_path, file_sha256))
                if ending == 'fail':
                    API_list.pop(int(i % len(API_list)))
                    print('api keys pop', API_list[int(i % len(API_list))])
                    print('there are ', len(API_list), ' available')
                else:
                    uploaded.append(calculate_hashes(
                        os.path.join(variation_path, file_sha256)))
                    save_to_txt(uploaded, logger_file)
                if len(API_list) == 0:
                    print('upload ', i, ' apk files')
                    break
        except Exception as e:
            print(variation, '---', e)
