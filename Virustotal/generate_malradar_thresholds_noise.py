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


naive_pool = '/home/public/rmt/heren/experiment/cl-exp/LHD_apk/naive_pool'

benign_tmp = txt_to_list('output/benign_10178_noise_threshold_1.txt')
benign_tmp = [item for item in benign_tmp if os.path.isfile(os.path.join(naive_pool, item + '.drebin'))]
benign_tmp = [item.replace('.apk', '') for item in benign_tmp]

malware_type = 'malradar'
benign_type = 'MalDroid_benign'

all_malware = txt_to_list('./software_hash/malradar.txt')
all_malware = [item.split('.')[0] for item in all_malware]

all_benign = txt_to_list('./software_hash/MalDroid_benign.txt')
clean_benign = txt_to_list('./output/MalDroid_benign_noise_threshold_1.txt')
error_benign = [item for item in all_benign if item not in clean_benign]

error_mal = txt_to_list('./output/malradar_noise_threshold_1.txt')
clean_mal = [item for item in all_malware if item not in error_mal]


def a(aa):
    return [item for item in aa if os.path.isfile(os.path.join(naive_pool, item + '.drebin'))]


error_benign = a(error_benign)
clean_mal = a(clean_mal)
error_mal = a(error_mal)
clean_benign = a(clean_benign)

malware = error_benign + clean_mal
mal_gt_label = [0] * len(error_benign) + [1] * len(clean_mal)
mal_noise_label = [1] * len(malware)
print(len(mal_gt_label))
benign = error_mal + clean_benign
ben_gt_label = [1] * len(error_mal) + [0] * len(clean_benign)
ben_noise_label = [0] * len(benign)
print(len(ben_gt_label))
print(len(malware))
print(len(benign))
import random
ccc = int(len(malware) - len(benign))
b = random.choices(benign_tmp, k=ccc)
print(len(malware) - len(benign))
print(len(b))
print(len(benign))
benign = benign + b
print(len(benign))
ben_gt_label = ben_gt_label + [0] * ccc
print(len(ben_gt_label))
ben_noise_label = ben_noise_label + [0] * ccc

data_filenames = malware + benign
data_filenames = [item + '.drebin' for item in data_filenames]
gt_label = mal_gt_label + ben_gt_label
noise_label = mal_noise_label + ben_noise_label
print(len(data_filenames))
print(len(gt_label))
print(len(noise_label))
print(sum(gt_label))
print(sum(noise_label))
from sklearn.metrics import accuracy_score

print(accuracy_score(gt_label, noise_label))
dump_joblib((data_filenames, gt_label, noise_label),
            './malradar_database_threshold_1.drebin')