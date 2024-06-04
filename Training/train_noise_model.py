# -*- coding: utf-8 -*- 
# @Time : 2024/1/28 13:21 
# @Author : DirtyBoy 
# @File : train_noise_model.py
from core.dataset_lib import build_dataset_from_numerical_data
from core.data_preprocessing import data_preprocessing
import argparse, os
import numpy as np
from sklearn.model_selection import KFold
from core.ensemble.vanilla import Vanilla
from core.ensemble.bayesian_ensemble import BayesianEnsemble
from core.ensemble.mc_dropout import MCDropout
from core.ensemble.deep_ensemble import DeepEnsemble, WeightedDeepEnsemble


def save_logger(base_path, vanilla_prob, vanilla_training_log, type, fold=0):
    np.save(os.path.join(base_path, type + '/prob/fold' + str(fold + 1)), vanilla_prob)
    np.save(os.path.join(base_path, type + '/log/fold' + str(fold + 1)), vanilla_training_log)


def main_deepdrebin(model_type, data_type, noise_type, noise_hyper, EPOCH=100):
    if model_type == 'vanilla':
        model_architecture = Vanilla
    elif model_type == 'bayesian':
        model_architecture = BayesianEnsemble
    elif model_type == 'mcdropout':
        model_architecture = MCDropout
    elif model_type == 'deepensemble':
        model_architecture = DeepEnsemble
    elif model_type == 'wdeepensemble':
        model_architecture = WeightedDeepEnsemble
    else:
        model_architecture = None
    output_path = '/home/lhd/Label_denoise_via_uncertainty/Training/output/' + data_type + '/drebin/' + noise_type + '_' + str(
        noise_hyper)

    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    if not os.path.isdir(os.path.join(output_path, model_type + '/prob')):
        os.makedirs(os.path.join(output_path, model_type + '/prob'))
        os.makedirs(os.path.join(output_path, model_type + '/log'))

    if os.path.isfile(os.path.join(output_path, model_type + '/prob/fold5.npy')):
        print('********************************************************************************')
        print('cross val training detail: ' + model_type + ' ' + data_type + ' ' + noise_type + ' ' + str(
            noise_hyper))
        print('training finish! ... file exist!!')
        print('********************************************************************************')
    else:
        dataset, gt_labels, noise_labels, input_dim, dataX_np, data_filenames = data_preprocessing(
            feature_type='drebin', data_type=data_type, noise_type=noise_type, noise_hyper=noise_hyper)
        if os.path.isfile(os.path.join(output_path, 'index/fold5.npy')):
            for fold in range(5):
                print('********************************************************************************')
                print('cross val training detail: ' + model_type + ' ' + data_type + ' ' + noise_type + ' ' + str(
                    noise_hyper))
                print('*********processing  No.' + str(fold + 1) + ' fold training********')
                print('********************************************************************************')
                print('load index from ', os.path.join(output_path, 'index/fold' + str(fold + 1) + '.npy'))
                train_index, test_index = np.load(os.path.join(output_path, 'index/fold' + str(fold + 1) + '.npy'),
                                                  allow_pickle=True)
                test_data = dataX_np[test_index]
                train_set = build_dataset_from_numerical_data((dataX_np[train_index], noise_labels[train_index]))
                validation_set = build_dataset_from_numerical_data((dataX_np[test_index], noise_labels[test_index]))

                model = model_architecture(architecture_type='dnn',
                                           model_directory='../Model/' + data_type + '/drebin/' + noise_type + '_' + str(
                                               str(noise_hyper) + '/fold' + str(fold + 1)))
                model_prob, model_log = model.fit(train_set=train_set, validation_set=validation_set,
                                                  input_dim=input_dim,
                                                  EPOCH=EPOCH,
                                                  test_data=test_data,
                                                  training_predict=True)
                save_logger(output_path, model_prob, model_log, type=model_type, fold=fold)
                del model_log
                del model_prob
        else:
            kf = KFold(n_splits=5, shuffle=True, random_state=123)

            for fold, (train_index, test_index) in enumerate(kf.split(dataX_np)):
                print('********************************************************************************')
                print('cross val training detail: ' + model_type + ' ' + data_type + ' ' + noise_type + ' ' + str(
                    noise_hyper))
                print('*********processing  No.' + str(fold + 1) + ' fold training********')
                print('********************************************************************************')
                if not os.path.isdir(os.path.join(output_path, 'index')):
                    os.makedirs(os.path.join(output_path, 'index'))
                np.save(os.path.join(output_path, 'index/fold' + str(fold + 1)), [train_index, test_index])

                test_data = dataX_np[test_index]
                train_set = build_dataset_from_numerical_data((dataX_np[train_index], noise_labels[train_index]))
                validation_set = build_dataset_from_numerical_data((dataX_np[test_index], noise_labels[test_index]))

                model = model_architecture(architecture_type='dnn',
                                           model_directory='../Model/' + data_type + '/drebin/' + noise_type + '_' + str(
                                               str(noise_hyper) + '/fold' + str(fold + 1)))
                model_prob, model_log = model.fit(train_set=train_set, validation_set=validation_set,
                                                  input_dim=input_dim,
                                                  EPOCH=EPOCH,
                                                  test_data=test_data,
                                                  training_predict=True)
                save_logger(output_path, model_prob, model_log, type=model_type, fold=fold)
                del model_log
                del model_prob


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_data_type', '-dt', type=str, default="malradar")
    parser.add_argument('-model_type', '-mt', type=str, default="vanilla")
    parser.add_argument('-noise_type', '-nt', type=str, default='random')
    parser.add_argument('-noise_hyper', '-nh', type=int, default=30)
    args = parser.parse_args()
    data_type = args.train_data_type
    model_type = args.model_type
    noise_type = args.noise_type
    noise_hyper = args.noise_hyper

    for data_type in ["malradar", ]:
        for model_type in ['deepensemble' ]:  #'deepensemble'
            for noise_type in ['Microsoft']:  # ['AVG', 'F-Secure', 'Ikarus', 'Sophos', 'Kaspersky','Alibaba','ZoneAlarm']
                for noise_hyper in [0]:
                    main_deepdrebin(model_type=model_type, data_type=data_type, noise_type=noise_type,
                                    noise_hyper=noise_hyper, EPOCH=100)
