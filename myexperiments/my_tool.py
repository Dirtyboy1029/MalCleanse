# -*- coding: utf-8 -*- 
# @Time : 2024/5/11 21:11 
# @Author : DirtyBoy 
# @File : my_tool_difference.py
import numpy as np
from myexperiments.utils import data_process_for_bayesian, evaluate_dataset_noise, get_confident_joint_index
from sklearn.cluster import KMeans
from myexperiments.metrics_utils import *
import argparse, os
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_data_type', '-dt', type=str, default="malradar")
    parser.add_argument('-noise_type', '-nt', type=str,
                        default='Sophos')  # ['F-Secure', 'Ikarus', 'Sophos', 'Alibaba','ZoneAlarm']
    parser.add_argument('-noise_hyper', '-nh', type=int, default=0)
    args = parser.parse_args()
    data_type = args.train_data_type
    noise_type = args.noise_type
    noise_hyper = args.noise_hyper
    if not os.path.isdir('my_tool_difference/dbscan/'):
        os.makedirs('my_tool_difference/dbscan/')
    if not os.path.isdir('my_tool_difference/isolation_forest/'):
        os.makedirs('my_tool_difference/isolation_forest/')
    if not os.path.isdir('my_tool_difference/one_class_svm/'):
        os.makedirs('my_tool_difference/one_class_svm/')

   
    print('-----------------------------------------------------------------------------------')
    print(
        '-------------' + data_type + '     ' + noise_type + '--------------------------')
    print('-----------------------------------------------------------------------------------')
    data_filenames, gt_labels, noise_labels, bayesian_prob = data_process_for_bayesian(
        data_type,
        noise_type,
        noise_hyper)
    print('Source benign noise number is ',
          len(np.where(noise_labels == 0)[0]) - len(np.where(gt_labels == 0)[0]))
    print('Source benign noise ratio is {:.5f}%'.format(
        ((len(np.where(noise_labels == 0)[0]) - len(np.where(gt_labels == 0)[0])) / len(
            np.where(noise_labels == 0)[0])) * 100))

    noise_benign_index = np.where(noise_labels == 0)[0]
    noise_malware_index = np.where(noise_labels == 1)[0]

    data = []
    for epoch in range(20):
        bay_prob = bayesian_prob[epoch]
        uc_entropy = np.array([predictive_entropy(prob_) for prob_ in bay_prob])
        uc_mean = np.array([np.mean(prob_) for prob_ in bay_prob])
        data.append(uc_entropy[noise_benign_index])
        data.append(uc_mean[noise_benign_index])
    data = np.array(data).T
    bay_prob = np.mean(bayesian_prob[19], axis=1)
    confident_joint = get_confident_joint_index(noise_labels, bay_prob)
    print(len(confident_joint[0]), len(confident_joint[2]))
    eva_noise_ratio = len(confident_joint[2]) / (len(confident_joint[2]) + len(confident_joint[0]))
    eva_noise_num = int(len(np.where(noise_labels == 0)[0]) * eva_noise_ratio)
    print('evalaute noise ratio is {:.5f}%'.format(eva_noise_ratio * 100))
    print('evalaute noise number is ', eva_noise_num)

    X_diff_feature = []
    for item in data:
        X_diff_feature.append(item)
    X_diff_feature = np.array(X_diff_feature)

    isolation_forest = IsolationForest(n_estimators=100,
                                       contamination=eva_noise_ratio)
    isolation_forest.fit(X_diff_feature)
    outliers_isolation_forest = isolation_forest.predict(X_diff_feature)
    np.save('my_tool_difference/isolation_forest/' + data_type + '_' + noise_type, outliers_isolation_forest)

    dbscan = DBSCAN(eps=0.1,
                    min_samples=eva_noise_num)  # len(np.where(noise_labels == 0)[0]) - len(np.where(gt_labels == 0)[0])
    dbscan.fit(X_diff_feature)
    outliers_dbscan = dbscan.labels_
    np.save('my_tool_difference/dbscan/' + data_type + '_' + noise_type, outliers_dbscan)

    clf = OneClassSVM(kernel='rbf', nu=eva_noise_ratio)
    clf.fit(X_diff_feature)
    outliers_svm = clf.predict(X_diff_feature)
    np.save('my_tool_difference/one_class_svm/' + data_type + '_' + noise_type, outliers_svm)
