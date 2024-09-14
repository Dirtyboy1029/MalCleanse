# -*- coding: utf-8 -*- 
# @Time : 2024/6/2 11:05 
# @Author : DirtyBoy 
# @File : malwhiteout_cl.py
import os
import numpy as np
import cleanlab as clb
from collections import Counter
from cleanlab.classification import LearningWithNoisyLabels
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from cleanlab.pruning import get_noise_indices
from cleanlab.latent_estimation import estimate_cv_predicted_probabilities
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


def read_joblib(path):
    import joblib
    if os.path.isfile(path):
        with open(path, 'rb') as fr:
            return joblib.load(fr)
    else:
        raise IOError("The {0} is not a file.".format(path))


def main_(noise_type_source):
    ENDING = []
    aa = []
    bb = []
    cc= []
    dd = []
    ee = []
    ff = []
    for i in range(1, 4):
        a, b = noise_type_source.rsplit('_', 1)
        noise_type = a + '_' + str(i) + '_' + b
        ending = []
        data_filenames, gt_labels, noise_labels = read_joblib(
            '../Training/config/' + 'databases_' + str(noise_type) + '.conf')
        gt_labels = np.array(gt_labels)
        noise_labels = np.array(noise_labels)
        is_noise = []
        for i in range(len(gt_labels)):
            if gt_labels[i] == noise_labels[i]:
                is_noise.append(0)
            else:
                is_noise.append(1)
        print(np.sum(is_noise))
        # print(len(data_filenames))
        MSG = "Before Noise Benign accurce is {:.2f}%."
        print(MSG.format((1 - accuracy_score(gt_labels, noise_labels)) * 200))

        drebin_psx = np.load('output/psx_drebin_' + str(noise_type) + '.npy')
        csbd_psx = np.load('output/psx_csbdRF_' + str(noise_type) + '.npy')
        malscan_psx = np.load(
            'output/psx_malscan_katz_' + noise_type + '.npy')

        combined_psx = np.concatenate((drebin_psx, malscan_psx), axis=1)
        combined_psx = np.concatenate((combined_psx, csbd_psx), axis=1)

        # Wrap around LogisticRegression classifier.
        lnl = LearningWithNoisyLabels(clf=LogisticRegression())
        lnl.fit(X=combined_psx, s=noise_labels)

        psx = estimate_cv_predicted_probabilities(
            X=combined_psx,
            labels=noise_labels,
            clf=lnl.clf,
            cv_n_folds=lnl.cv_n_folds,
            seed=lnl.seed,
        )

        ordered_label_errors = clb.pruning.get_noise_indices(s=noise_labels,
                                                             psx=psx,
                                                             prune_method='prune_by_noise_rate')

        label_errors = []
        for item in ordered_label_errors:
            if item:
                label_errors.append(1)
            else:
                label_errors.append(0)

        np.save('mwo_output/' + 'mwo_label_mask_' + noise_type, label_errors)

        # print(np.sum(label_errors))

        def evaluate_mwo(gt_labels, noise_labels, ordered_label_errors):
            denoise_label = []
            for i, item in enumerate(ordered_label_errors):
                if item:
                    denoise_label.append((noise_labels[i] + 1) % 2)
                else:
                    denoise_label.append(noise_labels[i])

            denoise_label_benign = []
            for i, item in enumerate(ordered_label_errors):
                if noise_labels[i] == 0:
                    if item:
                        denoise_label_benign.append((noise_labels[i] + 1) % 2)
                    else:
                        denoise_label_benign.append(noise_labels[i])
                else:
                    denoise_label_benign.append(noise_labels[i])

            return round(accuracy_score(gt_labels, denoise_label_benign), 6)

        # print(evaluate_mwo(gt_labels, noise_labels, ordered_label_errors))

        denoise_label_benign = []
        for i, item in enumerate(ordered_label_errors):
            if noise_labels[i] == 0:
                if item:
                    denoise_label_benign.append((noise_labels[i] + 1) % 2)
                else:
                    denoise_label_benign.append(noise_labels[i])
            else:
                denoise_label_benign.append(noise_labels[i])
        noise_benign_index = np.where(noise_labels == 0)[0]
        denoise_label_benign_all = denoise_label_benign
        denoise_label_benign = np.array(denoise_label_benign)[noise_benign_index]

        benign_noise = 0
        for i in noise_benign_index:
            if gt_labels[i] != denoise_label_benign_all[i]:
                benign_noise = benign_noise + 1
        accuracy = accuracy_score(gt_labels, noise_labels)
        accuracy1 = accuracy_score(gt_labels[noise_benign_index], denoise_label_benign)
        recall = recall_score(gt_labels[noise_benign_index], denoise_label_benign)
        prescion = precision_score(gt_labels[noise_benign_index], denoise_label_benign)
        f1 = f1_score(gt_labels[noise_benign_index], denoise_label_benign)
        # print("The Noise benign denoise on the dataset is ", benign_noise)
        ending.append(benign_noise)
        # MSG = "The clean accuracy on the dataset is {:.2f}%, Noise Benign accurce is {:.2f}%."
        # print(MSG.format(accuracy * 100, (1-accuracy1) * 100))
        # MSG = "Noise Benign accurce is {:.2f}%."
        # print(MSG.format((1-accuracy1) * 100))
        ending.append((1 - accuracy1) * 100)

        # MSG = "Recall is {:.2f}%"
        # print(MSG.format(recall * 100))
        ending.append(recall)

        # MSG = "Prescion is {:.2f}%"
        # print(MSG.format(prescion * 100))
        ending.append(prescion)
        # MSG = "F1 score is {:.2f}%"
        # print(MSG.format(f1 * 100))
        ending.append(f1)
        ENDING.append(ending)
        print(ending)

        if 'variation' in noise_type_source:
            def txt_to_list(txt_path):
                f = open(txt_path, "r")
                return f.read().splitlines()

            obfs_list = txt_to_list('../Training/config/Malradar_variation.txt')
            malware_common = []
            malware_obfs = []
            for item in noise_benign_index:
                if data_filenames[item] in obfs_list:
                    malware_obfs.append(item)
                else:
                    if gt_labels[item] == 1:
                        malware_common.append(item)
            print('obfs is', len(malware_obfs))
            cc.append(accuracy_score(gt_labels[malware_obfs], np.array(denoise_label_benign_all)[malware_obfs]))
            print(accuracy_score(gt_labels[malware_obfs], np.array(denoise_label_benign_all)[malware_obfs]))
            dd.append(np.sum(np.array(denoise_label_benign_all)[malware_obfs]))
            print('common is ', len(malware_common))
            aa.append(accuracy_score(gt_labels[malware_common], np.array(denoise_label_benign_all)[malware_common]))
            print(accuracy_score(gt_labels[malware_common], np.array(denoise_label_benign_all)[malware_common]))
            print(np.sum(np.array(denoise_label_benign_all)[malware_common]))
            bb.append(np.sum(np.array(denoise_label_benign_all)[malware_common]))
    print('---------------------------------------')
    print(np.mean(aa))
    print(len(malware_common)-int(np.mean(bb)))
    print(len(malware_obfs)-np.mean(dd))
    print(np.mean(cc)*100)
    print('---------------------------------------')
    return ENDING


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-noise_type', '-nt', type=str,
                        default='random')
    args = parser.parse_args()
    noise_type_source = args.noise_type

    end = np.mean(main_(noise_type_source), axis=0)
    print(end)
    MSG = "Noise Benign accurce is {:.2f}%."
    print(MSG.format(end[1]))
    print("The Noise benign denoise on the dataset is ", int(end[0]))

    MSG = "Recall is {:.2f}%"
    print(MSG.format(end[2] * 100))
    MSG = "Prescion is {:.2f}%"
    print(MSG.format(end[3] * 100))
    MSG = "F1 score is {:.2f}%"
    print(MSG.format(end[4] * 100))
