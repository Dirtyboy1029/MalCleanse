# -*- coding: utf-8 -*- 
# @Time : 2024/3/29 14:16 
# @Author : DirtyBoy 
# @File : add_noise.py
import os


def mkdir(target):
    try:
        if os.path.isfile(target):
            target = os.path.dirname(target)

        if not os.path.exists(target):
            os.makedirs(target)
        return 0
    except IOError as e:
        raise Exception("Fail to create directory! Error:" + str(e))


def dump_joblib(data, path):
    if not os.path.exists(os.path.dirname(path)):
        mkdir(os.path.dirname(path))

    try:
        import joblib
        with open(path, 'wb') as wr:
            joblib.dump(data, wr)
        return
    except IOError:
        raise IOError("Dump data failed.")


def save_to_txt(goal, txt_path):
    f = open(txt_path, "w")
    for line in goal:
        f.write(line + '\n')
    f.close()


def txt_to_list(txt_path):
    f = open(txt_path, "r")
    return f.read().splitlines()


model_type = 'drebin'
benign_type = 'MalDroid_benign'

naive_pool = '/home/public/rmt/heren/experiment/cl-exp/LHD_apk/naive_pool'

benign_tmp = txt_to_list('output/benign_10178_noise_threshold_1.txt')
benign_tmp = [item for item in benign_tmp if os.path.isfile(os.path.join(naive_pool, item + '.drebin'))]
benign_tmp = [item.replace('.apk', '') for item in benign_tmp]

benign_source = txt_to_list('software_hash/MalDroid_benign.txt')
benign_source = [item for item in benign_source if os.path.isfile(os.path.join(naive_pool, item + '.drebin'))]
benign_correct = txt_to_list('output/MalDroid_benign_noise_threshold_1.txt')
benign_correct = [item for item in benign_correct if os.path.isfile(os.path.join(naive_pool, item + '.drebin'))]
benign_error = [item for item in benign_source if item not in benign_correct]

ad_source = txt_to_list('software_hash/MalDroid_2020_Adware.txt')
ad_source = [item for item in ad_source if os.path.isfile(os.path.join(naive_pool, item + '.drebin'))]
ad_error = txt_to_list('output/MalDroid_2020_ad_noise_threshold_1.txt')
ad_error = [item for item in ad_error if os.path.isfile(os.path.join(naive_pool, item + '.drebin'))]
ad_correct = [item for item in ad_source if item not in ad_error]

bank_source = txt_to_list('software_hash/MalDroid_2020_Banking.txt')
bank_source = [item for item in bank_source if os.path.isfile(os.path.join(naive_pool, item + '.drebin'))]
bank_error = txt_to_list('output/MalDroid_2020_bank_noise_threshold_1.txt')
bank_error = [item for item in bank_error if os.path.isfile(os.path.join(naive_pool, item + '.drebin'))]
bank_correct = [item for item in bank_source if item not in bank_error]

risk_source = txt_to_list('software_hash/MalDroid_2020_Riskware.txt')
risk_source = [item for item in risk_source if os.path.isfile(os.path.join(naive_pool, item + '.drebin'))]
risk_error = txt_to_list('output/MalDroid_2020_risk_noise_threshold_1.txt')
risk_error = [item for item in risk_error if os.path.isfile(os.path.join(naive_pool, item + '.drebin'))]
risk_correct = [item for item in risk_source if item not in risk_error]

sms_source = txt_to_list('software_hash/MalDroid_2020_SMS.txt')
sms_source = [item for item in sms_source if os.path.isfile(os.path.join(naive_pool, item + '.drebin'))]
sms_error = txt_to_list('output/MalDroid_2020_sms_noise_threshold_1.txt')
sms_error = [item for item in sms_error if os.path.isfile(os.path.join(naive_pool, item + '.drebin'))]
sms_correct = [item for item in sms_source if item not in sms_error]

import random

all_malware = len(ad_source) + len(bank_source) + len(risk_source) + len(sms_source)
malware_ratio = 5100 / all_malware

ad_source = random.choices(ad_source, k=int(malware_ratio * len(ad_source)))
sms_source = random.choices(sms_source, k=int(malware_ratio * len(sms_source)))

risk_source = random.choices(risk_source, k=int(malware_ratio * len(risk_source)))
bank_source = random.choices(bank_source, k=int(malware_ratio * len(bank_source)))

malware = ad_source + sms_source + risk_source + bank_source + benign_error
malware_noise_label = []
malware_gt_label = (len(ad_source) + len(bank_source) + len(risk_source) + len(sms_source)) * [1] + len(
    benign_error) * [0]
for item in ad_source:
    if item in ad_error:
        malware_noise_label.append(0)
    else:
        malware_noise_label.append(1)

for item in sms_source:
    if item in sms_error:
        malware_noise_label.append(0)
    else:
        malware_noise_label.append(1)

for item in risk_source:
    if item in risk_error:
        malware_noise_label.append(0)
    else:
        malware_noise_label.append(1)

for item in bank_source:
    if item in bank_error:
        malware_noise_label.append(0)
    else:
        malware_noise_label.append(1)
print(len(malware))
print(len(malware_noise_label))
print(sum(malware_noise_label))
malware_noise_label = malware_noise_label + [1] * len(benign_error)
print(sum(malware_noise_label))
print(len(benign_tmp))
benign = benign_correct + list(
    random.choices(benign_tmp, k=int(len(malware) - len(benign_correct))))
benign_gt_label = [0] * len(benign)

data_filenames = malware + benign
data_filenames = [item + '.drebin' for item in data_filenames]
gt_label = malware_gt_label + benign_gt_label
noise_label = malware_noise_label + benign_gt_label
import numpy as np

print(len(data_filenames), len(gt_label), len(noise_label))
print(np.sum(gt_label))
print(np.sum(noise_label))
dump_joblib((data_filenames, gt_label, noise_label),
            './MalDroid_2020_threshold_1_database.drebin')
