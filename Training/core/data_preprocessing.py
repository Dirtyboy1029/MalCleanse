# conduct the group of 'out of distribution' experiments on drebin dataset
import os
from sklearn.metrics import accuracy_score
import pickle
import numpy as np
from .feature import feature_type_scope_dict
from .tools import utils
from .config import config, logging

def save_to_txt(goal, txt_path):
    f = open(txt_path, "w")
    for line in goal:
        f.write(line + '\n')
    f.close()


def txt_to_list(txt_path):
    f = open(txt_path, "r")
    return f.read().splitlines()

def load_dict_from_file(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


def data_preprocessing(feature_type='drebin', proc_numbers=8, data_type="drebin", noise_type='random', noise_hyper=20):
    assert feature_type in feature_type_scope_dict.keys(), 'Expected {}, but {} are supported.'.format(
        feature_type, feature_type_scope_dict.keys())
    android_features_saving_dir = config.get('metadata', 'naive_data_pool')
    intermediate_data_saving_dir = config.get('dataset', 'intermediate_directory')
    feature_extractor = feature_type_scope_dict[feature_type](android_features_saving_dir,
                                                              intermediate_data_saving_dir,
                                                              update=False,
                                                              proc_number=proc_numbers)

    save_path = os.path.join(intermediate_data_saving_dir,
                             data_type + '_database_' + noise_type + '_' + str(noise_hyper) + '.' + feature_type)

    print('load filename and label from ' + save_path)
    data_filenames, gt_labels, noise_labels = utils.read_joblib(save_path)
    gt_labels = np.array(gt_labels)
    noise_labels = np.array(noise_labels)
    print(accuracy_score(gt_labels, noise_labels))
    oos_features = [os.path.join(android_features_saving_dir, filename) for filename in data_filenames]
    dataset, input_dim, dataX_np = feature_extractor.feature2ipt(oos_features, noise_labels, data_type=data_type,
                                                                 noise_type=noise_type,
                                                                 noise_hyper=noise_hyper)
    return dataset, gt_labels, noise_labels, input_dim, dataX_np, data_filenames







