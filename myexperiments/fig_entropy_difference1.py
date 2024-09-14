# -*- coding: utf-8 -*- 
# @Time : 2024/5/22 11:24 
# @Author : DirtyBoy 
# @File : fig_entropy_difference1.py
import numpy as np
from utils import data_process, evaluate_dataset_noise, prob2psx, evaluate_cleanlab, predictive_entropy
from sklearn.metrics import accuracy_score
import cleanlab as clb
import argparse
import matplotlib.pyplot as plt



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_data_type', '-dt', type=str, default="malradar")
    parser.add_argument('-noise_type', '-nt', type=str,
                        default='F-Secure')  ##['AVG', 'F-Secure', 'Ikarus', 'Sophos', 'Kaspersky','Alibaba','ZoneAlarm']
    parser.add_argument('-noise_hyper', '-nh', type=int, default=0)
    args = parser.parse_args()
    data_type = args.train_data_type
    noise_type = args.noise_type
    noise_hyper = args.noise_hyper

    data_filenames, gt_labels, noise_labels, vanilla_prob, mcdropout_prob, bayesian_prob, deep_prob = data_process(data_type, noise_type,
                                                                                       noise_hyper)
    noise_benign_index = np.where(noise_labels == 0)[0]

    noise_benign_gt_malware_index = [i for i in noise_benign_index if gt_labels[i] != noise_labels[i]]
    noise_benign_gt_benign_index = [i for i in noise_benign_index if gt_labels[i] == noise_labels[i]]

    bayes_entropy = []
    mc_entropy = []
    deep_entropy = []
    for i in range(len(bayesian_prob)):
        if i % 2 == 0:
            bayes_entropy.append(np.array([predictive_entropy(item) for item in bayesian_prob[i]]))
            mc_entropy.append(np.array([predictive_entropy(item) for item in mcdropout_prob[i]]))
            deep_entropy.append(np.array([predictive_entropy(item) for item in deep_prob[i]]))

    bayes_benign_entropy = []
    bayes_malware_entropy = []

    mc_benign_entropy = []
    mc_malware_entropy = []

    deep_benign_entropy = []
    deep_malware_entropy = []

    for i in range(len(bayes_entropy)):
        bayes_benign_entropy.append(bayes_entropy[i][noise_benign_gt_benign_index])
        bayes_malware_entropy.append(bayes_entropy[i][noise_benign_gt_malware_index])

        mc_benign_entropy.append(mc_entropy[i][noise_benign_gt_benign_index])
        mc_malware_entropy.append(mc_entropy[i][noise_benign_gt_malware_index])

        deep_benign_entropy.append(deep_entropy[i][noise_benign_gt_benign_index])
        deep_malware_entropy.append(deep_entropy[i][noise_benign_gt_malware_index])

    fig, axs = plt.subplots(3, 1, figsize=(6, 8))

    axs[0].boxplot(bayes_benign_entropy, showfliers=False, positions=[(i + 0.8) for i in range(1,11)], widths=0.3,
                              patch_artist=True,
                              boxprops=dict(facecolor='green'))
    axs[0].boxplot(bayes_malware_entropy, showfliers=False, positions=[(i + 1.2) for i in range(1,11)], widths=0.3,
                    patch_artist=True,
                    boxprops=dict(facecolor='red'))

    axs[1].boxplot(mc_benign_entropy, showfliers=False, positions=[(i + 0.8) for i in range(1, 11)], widths=0.3,
                   patch_artist=True,
                   boxprops=dict(facecolor='green'))
    axs[1].boxplot(mc_malware_entropy, showfliers=False, positions=[(i + 1.2) for i in range(1, 11)], widths=0.3,
                   patch_artist=True,
                   boxprops=dict(facecolor='red'))

    axs[2].boxplot(deep_benign_entropy, showfliers=False, positions=[(i + 0.8) for i in range(1, 11)], widths=0.3,
                   patch_artist=True,
                   boxprops=dict(facecolor='green'))
    axs[2].boxplot(deep_malware_entropy, showfliers=False, positions=[(i + 1.2) for i in range(1, 11)], widths=0.3,
                   patch_artist=True,
                   boxprops=dict(facecolor='red'))

    plt.show()