# -*- coding: utf-8 -*- 
# @Time : 2024/8/6 22:09 
# @Author : DirtyBoy 
# @File : update_scan_list.py
import os
from datetime import datetime

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
    all = txt_to_list('experiment_sha256.txt')
    print(len(all))
    have = os.listdir('JSON/20240808')
    print(len(have))
    have = [item.split('.')[0] for item in have]
    need = []
    for item in all:
        if item not in have:
            need.append(item)
    save_to_txt(need,'tmp.txt')

