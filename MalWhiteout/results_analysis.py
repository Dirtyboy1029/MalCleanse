# -*- coding: utf-8 -*- 
# @Time : 2024/5/31 16:19 
# @Author : DirtyBoy 
# @File : results_analysis.py
import os
import numpy as np
from sklearn.metrics import accuracy_score


def txt_to_list(txt_path):
    import joblib, json
    with open(txt_path, 'r') as file:
        data = joblib.load(file)
    return json.loads(data)


def read_joblib(path):
    import joblib
    if os.path.isfile(path):
        with open(path, 'rb') as fr:
            return joblib.load(fr)
    else:
        raise IOError("The {0} is not a file.".format(path))


if __name__ == '__main__':
    enigne = 'Alibaba'
    filenames, gt_labels, noise_labels = read_joblib('../Training/config/malradar_database_' + enigne + '_1.drebin')
    label_errors_mask = np.load('output/drebin_labelerrorsmask_' + enigne + '.npy')
    print(accuracy_score(gt_labels, noise_labels))
    for i, is_noise in enumerate(label_errors_mask):
        if is_noise:
            noise_labels[i] = (noise_labels[i] + 1) % 2
    print(accuracy_score(gt_labels, noise_labels))
