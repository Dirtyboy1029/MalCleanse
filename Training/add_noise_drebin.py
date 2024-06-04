# -*- coding: utf-8 -*- 
# @Time : 2024/4/10 8:29 
# @Author : DirtyBoy 
# @File : add_noise.py
import os, argparse, random
import numpy as np
from sklearn.metrics import accuracy_score


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


def exist_feature_file(source_list):
    naive_pool = '/home/public/rmt/heren/experiment/cl-exp/LHD_apk/naive_pool'
    return_list = []
    for item in source_list:
        if os.path.isfile(os.path.join(naive_pool, item + '.drebin')):
            return_list.append(item)
    return return_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-malware_type', '-dt', type=str, default="malradar")
    parser.add_argument('-engine_name', '-en', type=str,
                        default="Alibaba")  # ['AVG', 'F-Secure', 'Ikarus', 'Sophos', 'Kaspersky','threshold']
    args = parser.parse_args()
    malware_type = args.malware_type
    engine_name = args.engine_name
    benign_type = 'MalDroid_benign'
    if malware_type=='malradar':

        print('**************************Enigne : ' + engine_name + '*********************************')

        all_malware = txt_to_list('../Virustotal/software_hash/' + malware_type + '.txt')
        all_malware = [item.split('.')[0] for item in all_malware]
        all_benign = txt_to_list('../Virustotal/software_hash/' + benign_type + '.txt')
        print(malware_type + ' exist source malware apk: ', len(all_malware))
        print(benign_type + ' exist source benign apk: ', len(all_benign))
        all_malware = exist_feature_file(all_malware)
        all_benign = exist_feature_file(all_benign)
        print('naive pool exist malware feature file: ', len(all_malware))
        print('naive pool exist benign feature file: ', len(all_benign))

        error_mal = txt_to_list('../Virustotal/output/' + malware_type + '_error_' + engine_name + '_.txt')
        error_mal = exist_feature_file(error_mal)
        print('malware exist error sample:', len(error_mal))
        unknown_mal = txt_to_list('../Virustotal/output/' + malware_type + '_unknown_' + engine_name + '_.txt')
        clean_mal = [item for item in all_malware if item not in error_mal]
        print('malware exist clean sample:', len(clean_mal))

        clean_benign = txt_to_list('../Virustotal/output/' + benign_type + '_error_' + engine_name + '_.txt')
        clean_benign = exist_feature_file(clean_benign)
        unknown_benign = txt_to_list('../Virustotal/output/' + benign_type + '_unknown_' + engine_name + '_.txt')
        print('benign exist clean sample:', len(clean_benign))
        error_benign = [item for item in all_benign if item not in clean_benign and item not in unknown_benign]
        print('benign exist error sample:', len(error_benign))

        print('************************First time evaluate**************************')
        malware = clean_mal + error_benign
        malware_gt_label = [1] * len(clean_mal) + [0] * len(error_benign)
        benign = clean_benign + error_mal
        benign_gt_label = [0] * len(clean_benign) + [1] * len(error_mal)
        print('malware: ', len(malware))
        print('benign: ', len(benign))

        if len(malware) < len(benign) and len(benign) >= 6000:
            ad_source = txt_to_list('../Virustotal/software_hash/MalDroid_2020_Adware.txt')
            ad_source = exist_feature_file(ad_source)
            bank_source = txt_to_list('../Virustotal/software_hash/MalDroid_2020_Banking.txt')
            bank_source = exist_feature_file(bank_source)
            # risk_source = txt_to_list('../Virustotal/software_hash/MalDroid_2020_Riskware.txt')
            # risk_source = exist_feature_file(risk_source)
            sms_source = txt_to_list('../Virustotal/software_hash/MalDroid_2020_SMS.txt')
            sms_source = exist_feature_file(sms_source)
            mal_tmp = ad_source + bank_source +  sms_source

            sms_error = txt_to_list('../Virustotal/output/MalDroid_2020_sms_error_' + engine_name + '_.txt')
            # risk_error = txt_to_list('../Virustotal/output/MalDroid_2020_risk_error_' + engine_name + '_.txt')
            bank_error = txt_to_list('../Virustotal/output/MalDroid_2020_bank_error_' + engine_name + '_.txt')
            ad_error = txt_to_list('../Virustotal/output/MalDroid_2020_ad_error_' + engine_name + '_.txt')
            mal_error = ad_error + sms_error +  bank_error
            mal_error = exist_feature_file(mal_error)

            sms_unknown = txt_to_list('../Virustotal/output/MalDroid_2020_sms_unknown_' + engine_name + '_.txt')
            #risk_unknown = txt_to_list('../Virustotal/output/MalDroid_2020_risk_unknown_' + engine_name + '_.txt')
            bank_unknown = txt_to_list('../Virustotal/output/MalDroid_2020_bank_unknown_' + engine_name + '_.txt')
            ad_unknown = txt_to_list('../Virustotal/output/MalDroid_2020_ad_unknown_' + engine_name + '_.txt')

            unknown = sms_unknown + ad_unknown + bank_unknown

            mal_clean = [item for item in mal_tmp if item not in mal_error]
            mal_clean = [item for item in mal_clean if item not in unknown]
            samples_sub = len(benign) - len(malware)
            malware = malware + random.sample(mal_clean, k=samples_sub)
            malware_gt_label = malware_gt_label + [1] * samples_sub

        elif len(malware) > len(benign) and len(malware) >= 6000:
            benign_tmp = txt_to_list('../Virustotal/output/benign_10178_error_' + engine_name + '_.txt')
            benign_tmp = exist_feature_file(benign_tmp)
            samples_sub = len(malware) - len(benign)
            benign_tmp = random.choices(benign_tmp, k=samples_sub)
            benign = benign + benign_tmp
            benign_gt_label = benign_gt_label + [0] * samples_sub
        else:
            ad_source = txt_to_list('../Virustotal/software_hash/MalDroid_2020_Adware.txt')
            ad_source = exist_feature_file(ad_source)
            bank_source = txt_to_list('../Virustotal/software_hash/MalDroid_2020_Banking.txt')
            bank_source = exist_feature_file(bank_source)
            # risk_source = txt_to_list('../Virustotal/software_hash/MalDroid_2020_Riskware.txt')
            # risk_source = exist_feature_file(risk_source)
            sms_source = txt_to_list('../Virustotal/software_hash/MalDroid_2020_SMS.txt')
            sms_source = exist_feature_file(sms_source)
            mal_tmp = ad_source + bank_source +  sms_source

            sms_error = txt_to_list('../Virustotal/output/MalDroid_2020_sms_error_' + engine_name + '_.txt')
            #risk_error = txt_to_list('../Virustotal/output/MalDroid_2020_risk_error_' + engine_name + '_.txt')
            bank_error = txt_to_list('../Virustotal/output/MalDroid_2020_bank_error_' + engine_name + '_.txt')
            ad_error = txt_to_list('../Virustotal/output/MalDroid_2020_ad_error_' + engine_name + '_.txt')
            mal_error = ad_error + sms_error +  bank_error
            mal_error = exist_feature_file(mal_error)

            sms_unknown = txt_to_list('../Virustotal/output/MalDroid_2020_sms_unknown_' + engine_name + '_.txt')
            #risk_unknown = txt_to_list('../Virustotal/output/MalDroid_2020_risk_unknown_' + engine_name + '_.txt')
            bank_unknown = txt_to_list('../Virustotal/output/MalDroid_2020_bank_unknown_' + engine_name + '_.txt')
            ad_unknown = txt_to_list('../Virustotal/output/MalDroid_2020_ad_unknown_' + engine_name + '_.txt')

            unknown = sms_unknown + ad_unknown +  bank_unknown

            mal_clean = [item for item in mal_tmp if item not in mal_error]
            mal_clean = [item for item in mal_clean if item not in unknown]
            samples_sub = 6000 - len(malware)
            malware = malware + random.sample(mal_clean, k=samples_sub)
            malware_gt_label = malware_gt_label + [1] * samples_sub

            benign_tmp = txt_to_list('../Virustotal/output/benign_10178_error_' + engine_name + '_.txt')
            benign_tmp = exist_feature_file(benign_tmp)
            samples_sub = 6000 - len(benign)
            benign_tmp = random.choices(benign_tmp, k=samples_sub)
            benign = benign + benign_tmp
            benign_gt_label = benign_gt_label + [0] * samples_sub

        data_filenames = malware + benign
        data_filenames = [item + '.drebin' for item in data_filenames]
        gt_label = malware_gt_label + benign_gt_label
        gt_label = np.array(gt_label)
        noise_label = [1] * len(malware) + [0] * len(benign)
        noise_label = np.array(noise_label)
        print('samples num:', len(data_filenames))
        print('malware samples num:', len(malware))
        print('noise ratio:', accuracy_score(gt_label, noise_label))
        print('malware noise ratio:', accuracy_score(malware_gt_label, [1] * len(malware)))
        print('benign noise ratio:', accuracy_score(benign_gt_label, [0] * len(malware)))
        dump_joblib((data_filenames, gt_label, noise_label),
                    './' + malware_type + '_database_' + engine_name + '_0.drebin')
    elif malware_type == 'MalDroid_2020':

        naive_pool = '/home/public/rmt/heren/experiment/cl-exp/LHD_apk/naive_pool'

        benign_tmp = txt_to_list('../Virustotal/output/benign_10178_error_' + engine_name + '_.txt')
        benign_tmp = [item for item in benign_tmp if os.path.isfile(os.path.join(naive_pool, item + '.drebin'))]
        benign_tmp = [item.replace('.apk', '') for item in benign_tmp]

        benign_source = txt_to_list('../Virustotal/software_hash/MalDroid_benign.txt')
        benign_source = [item for item in benign_source if os.path.isfile(os.path.join(naive_pool, item + '.drebin'))]
        benign_correct = txt_to_list('../Virustotal/output/MalDroid_benign_error_' + engine_name + '_.txt')
        benign_correct = [item for item in benign_correct if os.path.isfile(os.path.join(naive_pool, item + '.drebin'))]
        benign_unknown = txt_to_list('../Virustotal/output/MalDroid_benign_unknown_' + engine_name + '_.txt')
        benign_error = [item for item in benign_source if item not in benign_correct and item not in benign_unknown]

        ad_source = txt_to_list('../Virustotal/software_hash/MalDroid_2020_Adware.txt')
        ad_source = [item for item in ad_source if os.path.isfile(os.path.join(naive_pool, item + '.drebin'))]
        ad_error = txt_to_list('../Virustotal/output/MalDroid_2020_ad_error_' + engine_name + '_.txt')
        ad_error = [item for item in ad_error if os.path.isfile(os.path.join(naive_pool, item + '.drebin'))]
        ad_unknown = txt_to_list('../Virustotal/output/MalDroid_2020_ad_unknown_' + engine_name + '_.txt')
        ad_correct = [item for item in ad_source if item not in ad_error and item not in ad_unknown]

        bank_source = txt_to_list('../Virustotal/software_hash/MalDroid_2020_Banking.txt')
        bank_source = [item for item in bank_source if os.path.isfile(os.path.join(naive_pool, item + '.drebin'))]
        bank_error = txt_to_list('../Virustotal/output/MalDroid_2020_bank_error_' + engine_name + '_.txt')
        bank_error = [item for item in bank_error if os.path.isfile(os.path.join(naive_pool, item + '.drebin'))]
        bank_unknown = txt_to_list('../Virustotal/output/MalDroid_2020_bank_unknown_' + engine_name + '_.txt')
        bank_correct = [item for item in bank_source if item not in bank_error and item not in bank_unknown]

        # risk_source = txt_to_list('software_hash/MalDroid_2020_Riskware.txt')
        # risk_source = [item for item in risk_source if os.path.isfile(os.path.join(naive_pool, item + '.drebin'))]
        # risk_error = txt_to_list('output/MalDroid_2020_risk_error_' + engine_name + '_.txt')
        # risk_error = [item for item in risk_error if os.path.isfile(os.path.join(naive_pool, item + '.drebin'))]
        # risk_unknown = txt_to_list('output/MalDroid_2020_risk_unknown_' + engine_name + '_.txt')
        # risk_correct = [item for item in risk_source if item not in risk_error and item not in risk_unknown]

        sms_source = txt_to_list('../Virustotal/software_hash/MalDroid_2020_SMS.txt')
        sms_source = [item for item in sms_source if os.path.isfile(os.path.join(naive_pool, item + '.drebin'))]
        sms_error = txt_to_list('../Virustotal/output/MalDroid_2020_sms_error_' + engine_name + '_.txt')
        sms_error = [item for item in sms_error if os.path.isfile(os.path.join(naive_pool, item + '.drebin'))]
        sms_unknown = txt_to_list('../Virustotal/output/MalDroid_2020_sms_unknown_' + engine_name + '_.txt')
        sms_correct = [item for item in sms_source if item not in sms_error and item not in sms_unknown]

        import random

        all_malware = len(ad_source) + len(bank_source) + len(sms_source)
        malware_ratio = 5200 / all_malware

        ad_correct = random.choices(ad_correct, k=int(malware_ratio * len(ad_source)))
        sms_correct = random.choices(sms_correct, k=int(malware_ratio * len(sms_source)))

        # risk_correct = random.choices(risk_correct, k=int(malware_ratio * len(risk_source)))
        bank_correct = random.choices(bank_correct, k=int(malware_ratio * len(bank_source)))

        malware = ad_correct + sms_correct + bank_correct + benign_error
        print(len(benign_error))
        malware_gt_label = [1] * (len(ad_correct) + len(sms_correct) + len(bank_correct)) + [0] * len(
            benign_error)
        malware_noise_label = [1] * len(malware)

        benign_ = benign_correct + ad_error + sms_error + bank_error
        benign_correct = random.choices(benign_correct, k=int(5200 / len(benign_)))
        ad_error = random.choices(ad_error, k=int(5200 / len(benign_) * len(ad_error)))
        sms_error = random.choices(sms_error, k=int(5200 / len(benign_) * len(sms_error)))
        bank_error = random.choices(bank_error, k=int(5200 / len(benign_) * len(bank_error)))
        # risk_error = random.choices(risk_error, k=int(5200 / len(benign_) * len(risk_error)))

        benign = benign_correct + ad_error + sms_error + bank_error
        benign_gt_label = [0] * len(benign_correct) + [1] * (
                len(ad_error) + len(sms_error) + len(bank_error)) + [0] * (len(malware) - len(benign))
        print(len(ad_error) + len(sms_error) + len(bank_error))
        benign = benign + random.choices(benign_tmp, k=(len(malware) - len(benign)))

        benign_noise_label = [0] * len(benign)

        data_filenames = malware + benign
        data_filenames = [item + '.drebin' for item in data_filenames]
        gt_label = malware_gt_label + benign_gt_label
        gt_label = np.array(gt_label)
        noise_label = [1] * len(malware) + [0] * len(benign)
        noise_label = np.array(noise_label)
        print('samples num:', len(data_filenames))
        print('malware samples num:', len(malware))
        print('noise ratio:', accuracy_score(gt_label, noise_label))
        print('malware noise ratio:', accuracy_score(malware_gt_label, [1] * len(malware)))
        print('benign noise ratio:', accuracy_score(benign_gt_label, [0] * len(malware)))
        dump_joblib((data_filenames, gt_label, noise_label),
                    './MalDroid_2020_database_' + engine_name + '_' + '0.drebin')

