# -*- coding: utf-8 -*- 
# @Time : 2024/1/30 16:32 
# @Author : DirtyBoy 
# @File : evaluate_noise_labels.py
import json, os


def save_to_txt(goal, txt_path):
    f = open(txt_path, "w")
    for line in goal:
        f.write(line + '\n')
    f.close()


def txt_to_list(txt_path):
    f = open(txt_path, "r")
    return f.read().splitlines()


def evaluate_noise(data_type, malware, benign, unknown):
    print('Dataset is ' + data_type)
    print('The number of malicious samples that meet the criteria is ', malware)
    print("The number of malicious samples misclassified as benign is ", benign)
    print("The noise rate is ", (benign / (malware + benign)) * 100, '%')
    if unknown > 0:
        print("The number of samples with unknown attributes is ", unknown)


def main_(data_type='MalDroid_2020_risk', vt=None, engine_name_list=None):
    json_file_path = 'JSON/' + data_type
    json_file_list = os.listdir(json_file_path)
    if not engine_name_list:
        print('*********************Create noise through threshold*********************')
        print('thrsholds is ', vt)
    elif len(engine_name_list) > 1:
        print(
            '*********************Select trustworthy engine groups and create noise through a threshold.*********************')
        print('engine name set contain ' + str(engine_name_list))
        print('thrsholds=', vt)
    elif len(engine_name_list) == 1:
        print('*********************Trust only a certain reliable engine.*********************')
        print('engine name is ' + engine_name_list[0])

    malware = 0
    benign = 0
    unknown = 0
    error_list = []
    unknown_list = []

    for json_file in json_file_list:
        json_path = os.path.join(json_file_path, json_file)
        with open(json_path, 'r') as f:
            content = f.read()
        data = json.loads(content)
        if not engine_name_list:
            if data['data']['attributes']['last_analysis_stats']['malicious'] >= vt:
                malware += 1
            else:
                benign += 1
                error_list.append(os.path.basename(json_path).split('.')[0])
        elif len(engine_name_list) > 1:
            if len(engine_name_list) < vt:
                print("请给出合适的阈值")
                import sys
                sys.exit()
            else:
                count = 0
                for engine_name in engine_name_list:
                    try:
                        if data['data']['attributes']['last_analysis_results'][engine_name]['category'] == 'malicious':
                            count += 1
                    except KeyError:
                        pass
                if count >= vt:
                    malware += 1
                else:
                    benign += 1
                    error_list.append(os.path.basename(json_path).split('.')[0])
        elif len(engine_name_list) == 1:
            try:
                if data['data']['attributes']['last_analysis_results'][engine_name_list[0]]['category'] == 'malicious':
                    malware += 1
                else:
                    benign += 1
                    error_list.append(os.path.basename(json_path).split('.')[0])
            except KeyError:
                unknown += 1
                unknown_list.append(os.path.basename(json_path).split('.')[0])
    # save_to_txt(unknown_list, 'output/' + data_type + '_unknown_' + engine_name_list[0] + '_' + '.txt')
    #
    # save_to_txt(error_list, 'output/' + data_type + '_error_' + engine_name_list[0] + '_' + '.txt')
    return malware, benign, unknown


if __name__ == '__main__':
    import os

    '''
    drebin:['AntiVir', 'AVG', 'BitDefender', 'ClamAV', 'ESET', 'F-Secure', 'Kaspersky', 'McAfee', 'Panda', 'Sophos'] 2/10
    
    "Scalable Malware Clustering Through Coarse-Grained Behavior Modeling":
    ['F-Secure','Ikarus','Symantec','Kaspersky','VirusBuster'] 
    '''
    vt = 15
    ''''
        'Bkav', 'Lionic', 'Elastic', 'MicroWorld-eScan', 'FireEye', 'CAT-QuickHeal', 'McAfee', 'Malwarebytes', 
        'VIPRE', 'Sangfor', 'Trustlook', 'BitDefender', 'K7GW', 'K7AntiVirus', 'BitDefenderTheta', 'VirIT', 'Cyren', 
        'SymantecMobileInsight', 'Symantec', 'ESET-NOD32', 'Cynet', 'TrendMicro-HouseCall', 'Avast', 'ClamAV', 'Kaspersky', 
        'Alibaba', 'NANO-Antivirus', 'ViRobot', 'Rising', 'Sophos', 'Baidu', 'F-Secure', 'DrWeb', 'Zillya', 'TrendMicro', 
        'McAfee-GW-Edition', 'CMC', 'Emsisoft', 'Ikarus', 'GData', 'Jiangmin', 'Avira', 'Antiy-AVL', 'Microsoft', 'Gridinsoft', 
        'Xcitium', 'Arcabit', 'SUPERAntiSpyware', 'ZoneAlarm', 'Avast-Mobile', 'Google', 'BitDefenderFalx', 'AhnLab-V3', 
        'Acronis', 'VBA32', 'ALYac', 'MAX', 'Zoner', 'Tencent', 'Yandex', 'TACHYON', 'MaxSecure', 'Fortinet', 'AVG', 
        'Panda', 'Cybereason', 'tehtris', 'DeepInstinct', 'Webroot', 'APEX', 'Paloalto', 'Trapmine', 'Cylance', 'SentinelOne', 'CrowdStrike'
        '''

    for data_type in ['benign_10178','malradar', 'MalDroid_2020_sms', 'MalDroid_2020_ad', 'MalDroid_2020_bank', 'MalDroid_2020_risk','MalDroid_benign']: #

        # ['benign_10178', 'malradar', 'MalDroid_2020_sms', 'MalDroid_2020_ad', 'MalDroid_2020_bank',
        #   'MalDroid_benign']
        # for vt in [25,]:
        malware, benign, unknown = main_(data_type, vt, engine_name_list=None)
        evaluate_noise(data_type, malware, benign, unknown)

    # for data_type in ['benign']:
    #     for vt in [1, 2, 3, 4, 5]:
    #         malware, benign = main_(data_type, vt)
    #         print(data_type, vt, 100 - (benign / (malware + benign) * 100), malware + benign, benign)
