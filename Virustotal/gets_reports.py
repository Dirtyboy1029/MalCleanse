# -*- coding: utf-8 -*- 
# @Time : 2023/4/24 8:30 
# @Author : DirtyBoy 
# @File : gets_reports.py
import requests
import json, os, argparse
from datetime import datetime
from tqdm import tqdm

today = datetime.now().strftime('%Y%m%d')
# today = '20240806'

headers = {
    'x-apikey': '',
    'Host': 'www.virustotal.com',
    'range': 'bytes=equest',
    'user-agent': 'curl/7.68.0',
    'accept': '*/*'
}


def save_to_txt(goal, txt_path):
    f = open(txt_path, "w")
    for line in goal:
        f.write(line + '\n')
    f.close()


def txt_to_list(txt_path):
    f = open(txt_path, "r")
    return f.read().splitlines()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_type', "-dt", type=str, default="malgenome",
                        choices=['drebin_malradar', 'Malradar_obfs', 'malgenome'])

    args = parser.parse_args()
    software_type = args.data_type
    API_list = txt_to_list('vt_keys.txt')
    json_folder = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'JSON'), software_type)
    if not os.path.isdir(json_folder):
        os.makedirs(json_folder)
    if software_type == 'Malradar_obfs':
        sha256 = txt_to_list('Upload_log/log.txt')
    elif software_type == 'drebin_malradar':
        sha256 = txt_to_list('Datasets/drebin.txt') + txt_to_list('Datasets/malradar.txt')
    elif software_type == 'malgenome':
        sha256 = txt_to_list('Datasets/malgenome.txt')
    else:
        sha256 = ['24d823fa8cd20c0e1d96107813c7ffddc671ac6535848f80fc636c17ebc57950',
                  '09d6f71c34296daceb09f5a962f306d815f76019500e53b613a4c43e503cd774',
                  '4f6d124bb6d0ce04cd4a8af29d6baee9a984322523c4a19df8f2002699812bab',
                  '8971bc9175a0c6837006b35a0fb3d114ce59306213d805fadaf7c580c7d175d5',
                  'f459f37d236e5ba5dcae8dd5be7f5edd98374761dc3af4ccc6812c6997c2b5a0',
                  'ea36de02a420c194e9877f66b5682b6b2e29e9762117aa8fabfe594e2d24491d']  #
    need_sha256 = []
    for file_sha256 in sha256:
        if not os.path.isfile((os.path.join(json_folder, file_sha256 + '.json'))):
            need_sha256.append(file_sha256)
    # save_to_txt(need_sha256,'tmp11.txt')
    for i, file_sha256 in tqdm(enumerate(need_sha256), total=len(need_sha256), desc='VirusTotal report'):
        headers['x-apikey'] = API_list[int(i % len(API_list))]
        url = "https://www.virustotal.com/api/v3/files/" + file_sha256 + ""
        res = requests.get(url=url, headers=headers)
        content = res.json()
        content = json.dumps(content)
        if len(content) <= 100:
            API_list.pop(int(i % len(API_list)))
            print('api keys pop', API_list[int(i % len(API_list))])
            print('there are ', len(API_list), ' available')
        else:
            json_path = os.path.join(json_folder, file_sha256 + '.json')
            with open(json_path, 'w') as f:
                f.write(content)
            f.close()
        if len(API_list) == 0:
            print('get reports ', i, ' apk files')
            break
