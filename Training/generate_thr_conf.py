# -*- coding: utf-8 -*- 
# @Time : 2024/9/8 14:45 
# @Author : DirtyBoy 
# @File : generate_thr_conf.py
import pandas as pd
import numpy as np
import random
from sklearn.metrics import accuracy_score


def save_to_txt(goal, txt_path):
    f = open(txt_path, "w")
    for line in goal:
        f.write(line + '\n')
    f.close()


def txt_to_list(txt_path):
    f = open(txt_path, "r")
    return f.read().splitlines()


def dump_joblib(data, path):
    try:
        import joblib
        with open(path, 'wb') as wr:
            joblib.dump(data, wr)
        return
    except IOError:
        raise IOError("Dump data failed.")


def shuffle_data(list1, list2, list3):
    combined = list(zip(list1, list2, list3))
    random.shuffle(combined)
    list1_shuffled, list2_shuffled, list3_shuffled = zip(*combined)
    list1_shuffled = list(list1_shuffled)
    list2_shuffled = list(list2_shuffled)
    list3_shuffled = list(list3_shuffled)

    return list1_shuffled, list2_shuffled, list3_shuffled


TOTAL_SAMPLES_NUM = 10000

if __name__ == '__main__':
    Threshold = 26
    malware_type = 'first_2024'
    if malware_type == 'first_rescan':
        truth_columns = 'rescan_2024'
        old_columns = 'vt_2022'
    elif malware_type == 'first_2024':
        truth_columns = 'vt_2024'
        old_columns = 'vt_2022'
    else:
        truth_columns = ''
        old_columns = ''
    vt_thr = pd.read_csv(
        '../APK_Collection/2.csv')  # (['sha256', 'malware_type', 'vt_2022', 'vt_2024', 'rescan_2024']
    del vt_thr['Unnamed: 0']
    vt_thr = vt_thr.drop_duplicates(subset='sha256', keep='first')

    vt_thr_nan = vt_thr[vt_thr[old_columns].isna()]
    vt_thr_nan_benign = vt_thr_nan[(vt_thr[truth_columns] <= Threshold)]

    vt_thr_nan_malware = vt_thr_nan[(vt_thr[truth_columns] > Threshold)]
    vt_thr = vt_thr[vt_thr[truth_columns].notna() & vt_thr[old_columns].notna()]

    mis_benign = pd.concat([vt_thr[(vt_thr[old_columns] <= Threshold) & (vt_thr[truth_columns] <= Threshold)],
                            vt_thr[(vt_thr[old_columns] < Threshold) & (vt_thr[truth_columns] > Threshold)]])

    mis_benign = pd.concat([mis_benign, vt_thr_nan_benign])['sha256'].tolist()
    truth_malware = pd.concat(
        [vt_thr[(vt_thr[old_columns] > Threshold) & (vt_thr[truth_columns] > Threshold)], vt_thr_nan_malware])[
        'sha256'].tolist()

    malware = txt_to_list('config/drebin_sha256.mwo') + txt_to_list('config/malradar_sha256.mwo') + txt_to_list(
        'config/malDroid_2020_Adware.mwo') + txt_to_list('config/malDroid_2020_Banking.mwo') + txt_to_list(
        'config/malDroid_2020_SMS.mwo')

    truth_malware = [item.lower() for item in truth_malware if item.lower() in malware]
    mis_benign = [item.lower() for item in mis_benign if item.lower() in malware]

    benign = txt_to_list('config/my_benign.mwo')

    print(len(mis_benign))
    import time

    for i in range(1, 4):
        time.sleep(1)
        seed = int(time.time())
        random.seed(seed)
        truth_malware = random.sample(truth_malware, k=5000)

        my_benign = random.sample(benign, k=5000 - len(mis_benign))

        data_filenames = truth_malware + mis_benign + my_benign
        gt_labels = [1] * len(truth_malware + mis_benign) + [0] * len(my_benign)
        noise_labels = [1] * len(truth_malware) + [0] * len(mis_benign + my_benign)
        print(accuracy_score(gt_labels, noise_labels))

        data_filenames, gt_labels, noise_labels = shuffle_data(data_filenames, gt_labels, noise_labels)
        gt_labels = np.array(gt_labels)
        noise_labels = np.array(noise_labels)
        dump_joblib((data_filenames, gt_labels, noise_labels),
                    'config/databases_thr_' + str(i) + '_' + str(Threshold) + '.conf')
        print(
            'generate conf file to ' + 'config/databases_thr_' + str(i) + '_' + str(Threshold) + '.conf')
