# -*- coding: utf-8 -*- 
# @Time : 2024/9/9 13:26 
# @Author : DirtyBoy 
# @File : generate_random_conf.py
import numpy as np
import random, time, os
from sklearn.metrics import accuracy_score


def save_to_txt(goal, txt_path):
    f = open(txt_path, "w")
    for line in goal:
        f.write(line + '\n')
    f.close()


def txt_to_list(txt_path):
    f = open(txt_path, "r")
    return f.read().splitlines()


def read_joblib(path):
    import joblib
    if os.path.isfile(path):
        with open(path, 'rb') as fr:
            return joblib.load(fr)
    else:
        raise IOError("The {0} is not a file.".format(path))


def dump_joblib(data, path):
    try:
        import joblib
        with open(path, 'wb') as wr:
            joblib.dump(data, wr)
        return
    except IOError:
        raise IOError("Dump data failed.")


def main_func(noise_ratio=20,
              malware_type='malradar'):
    my_benign = txt_to_list('config/my_benign.drebin')
    if malware_type == 'malradar':
        malware = txt_to_list('config/malradar_genome_sha256.drebin')
    else:
        malware = txt_to_list('config/drebin_sha256.drebin')
    benign = random.sample(my_benign, k=len(malware))
    for i in range(1, 4):
        time.sleep(1)
        seed = int(time.time())
        random.seed(seed)
        malware_noise_index = random.sample(list(range(len(malware))), k=int(len(malware) * noise_ratio / 100))
        random.seed(seed + 1)
        benign_noise_index = random.sample(list(range(len(malware))), k=int(len(malware) * noise_ratio / 100))

        malware_gt_label = [1] * len(malware)
        malware_noise_label = []
        for j in range(len(malware)):
            if j in malware_noise_index:
                malware_noise_label.append(0)
            else:
                malware_noise_label.append(1)

        benign_gt_label = [0] * len(benign)
        benign_noise_label = []
        for j in range(len(benign)):
            if j in benign_noise_index:
                benign_noise_label.append(1)
            else:
                benign_noise_label.append(0)
        data_filenames = malware + benign
        gt_labels = np.array(malware_gt_label + benign_gt_label)
        noise_labels = np.array(malware_noise_label + benign_noise_label)
        print(accuracy_score(gt_labels, noise_labels))
        dump_joblib((data_filenames, gt_labels, noise_labels),
                    'config/databases_random_' + str(i) + '_' + malware_type + '_' + str(noise_ratio) + '.conf')
        print(
            'generate conf file to config/databases_random_' + str(i) + '_' + malware_type + '_' + str(
                noise_ratio) + '.conf')


if __name__ == '__main__':
    for noise_ratio in [5, 10, 15, 20, 25, 30, 35, 40]:
        for malware_type in ['drebin', 'malradar']:
            main_func(noise_ratio=noise_ratio, malware_type=malware_type)
