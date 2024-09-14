# -*- coding: utf-8 -*- 
# @Time : 2024/8/6 20:31 
# @Author : DirtyBoy 
# @File : reanalyze.py
import requests, os, time
from datetime import datetime
from tqdm import tqdm

today = datetime.now().strftime('%Y%m%d')


def save_to_txt(goal, txt_path):
    f = open(txt_path, "w")
    for line in goal:
        f.write(line + '\n')
    f.close()


def txt_to_list(txt_path):
    f = open(txt_path, "r")
    return f.read().splitlines()


if __name__ == '__main__':
    api_keys = txt_to_list('vt_keys.txt')
    print('There are ', len(api_keys), ' available')
    log_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Analyze_log')
    if not os.path.isdir(log_folder):
        os.makedirs(log_folder)
    if not os.path.isfile(os.path.join(log_folder, 'drebin_malradar_' + today + '.txt')):
        with open(os.path.join(log_folder, 'drebin_malradar_' + today + '.txt'), 'w') as file:
            pass
    sha256 = txt_to_list('Datasets/tmp22.txt')
    scand_sha256 = txt_to_list(os.path.join(log_folder, 'drebin_malradar_' + today + '.txt'))
    sha256 = [item for item in sha256 if item not in scand_sha256]
    for scan_file_num, file_hash in tqdm(enumerate(sha256), total=len(sha256), desc='VirusTotal reanalyze'):
        url = f'https://www.virustotal.com/api/v3/files/{file_hash}/analyse'
        headers = {
            'x-apikey': api_keys[int(scan_file_num % len(api_keys))]
        }
        response = requests.post(url, headers=headers)

        if response.status_code == 200:
            scand_sha256.append(file_hash)
            save_to_txt(scand_sha256, os.path.join(log_folder, 'drebin_malradar_' + today + '.txt'))
            time.sleep(3)
        elif len(api_keys) == 0:
            print('reanalyze ', scan_file_num, ' apk files')
            break
        else:
            print('api keys pop', api_keys[int(scan_file_num % len(api_keys))])
            api_keys.pop(int(scan_file_num % len(api_keys)))
            print('there are ', len(api_keys), ' available')
