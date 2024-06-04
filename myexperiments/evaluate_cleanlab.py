# -*- coding: utf-8 -*- 
# @Time : 2024/5/7 15:30 
# @Author : DirtyBoy 
# @File : demo1.py
import numpy as np
from utils import data_process, evaluate_dataset_noise, prob2psx, evaluate_cleanlab
from sklearn.metrics import accuracy_score
import cleanlab as clb
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_data_type', '-dt', type=str, default="malradar")
    parser.add_argument('-noise_type', '-nt', type=str, default='Microsoft') ##['AVG', 'F-Secure', 'Ikarus', 'Sophos', 'Kaspersky','Alibaba','ZoneAlarm']
    parser.add_argument('-noise_hyper', '-nh', type=int, default=0)
    args = parser.parse_args()
    data_type = args.train_data_type
    noise_type = args.noise_type
    noise_hyper = args.noise_hyper

    data_filenames, gt_labels, noise_labels, vanilla_prob, mcdropout_prob, bayesian_prob, deep_prob = data_process(
        data_type,
        noise_type,
        noise_hyper)
    evaluate_dataset_noise(data_type, noise_type, noise_hyper, data_filenames, gt_labels, noise_labels)
    print('******' + 'Cross-validation for ' + str(
        vanilla_prob.shape[0] * 5) + ' epochs, evaluating every 5 epochs.' + '*******')

    print('***************************************************************************')
    for i in range(vanilla_prob.shape[0]):
        if i%2==1:
            print('******************************epoch ' + str(5 * (i + 1)) + '***************************************')
            # print('vanilla acc', round(accuracy_score(gt_labels, (vanilla_prob[i] > 0.5).astype(int)), 6),
            #       '  mcdropout acc',
            #       round(accuracy_score(gt_labels, (np.mean(mcdropout_prob[i], axis=1) > 0.5).astype(int)), 6),
            #       '  bayesian acc',
            #       round(accuracy_score(gt_labels, (np.mean(bayesian_prob[i], axis=1) > 0.5).astype(int)), 6),
            #       )
            noise_labels = [int(item) for item in noise_labels]
            vanilla_ordered_label_errors = clb.pruning.get_noise_indices(s=noise_labels,
                                                                         psx=prob2psx(vanilla_prob[i]),
                                                                         prune_method='prune_by_noise_rate')  # sorted_index_method='prob_given_label', verbose=1
            mcdropout_ordered_label_errors = clb.pruning.get_noise_indices(s=noise_labels,
                                                                           psx=prob2psx(np.mean(mcdropout_prob[i], axis=1)),
                                                                           prune_method='prune_by_noise_rate')  # sorted_index_method='prob_given_label', verbose=1
            bayesian_ordered_label_errors = clb.pruning.get_noise_indices(s=noise_labels,
                                                                          psx=prob2psx(np.mean(bayesian_prob[i], axis=1)),
                                                                          prune_method='prune_by_noise_rate')  # sorted_index_method='prob_given_label', verbose=1
            deep_ordered_label_errors = clb.pruning.get_noise_indices(s=noise_labels,
                                                                          psx=prob2psx(
                                                                              np.mean(deep_prob[i], axis=1)),
                                                                          prune_method='prune_by_noise_rate')  # sorted_index_method='prob_given_label', verbose=1
            print('vanilla acc',
                  evaluate_cleanlab(gt_labels, noise_labels, vanilla_ordered_label_errors),
                  '  mcdropout acc',
                  evaluate_cleanlab(gt_labels, noise_labels, mcdropout_ordered_label_errors),
                  '  bayesian acc',
                  evaluate_cleanlab(gt_labels, noise_labels, bayesian_ordered_label_errors),
                  '  deepensemble acc',
                  evaluate_cleanlab(gt_labels, noise_labels, deep_ordered_label_errors),
                  )
            # if noise_type != 'random':
            #     print('vanilla acc',
            #           evaluate_cleanlab(gt_labels, noise_labels, vanilla_ordered_label_errors, clean_malware=False),
            #           '  mcdropout acc',
            #           evaluate_cleanlab(gt_labels, noise_labels, mcdropout_ordered_label_errors, clean_malware=False),
            #           '  bayesian acc',
            #           evaluate_cleanlab(gt_labels, noise_labels, bayesian_ordered_label_errors, clean_malware=False),
            #           )
