import networkx as nx
import time
import argparse
import csv
from multiprocessing import Pool as ThreadPool
from functools import partial
import glob
import os
import numpy as np


def read_joblib(path):
    import joblib
    if os.path.isfile(path):
        with open(path, 'rb') as fr:
            return joblib.load(fr)
    else:
        raise IOError("The {0} is not a file.".format(path))


def parseargs():
    parser = argparse.ArgumentParser(description='Malware Detection with centrality.')
    parser.add_argument('-d', '--dir', help='The path of a dir contains benign and malware.', type=str,
                        default='/home/public/rmt/heren/experiment/cl-exp/LHD_apk/malwhiteout_naive_pool/malscan')
    parser.add_argument('-o', '--output', help='The dir_path of output', type=str)
    parser.add_argument('-c', '--centrality', help='The type of centrality: degree, katz, closeness, harmonic',
                        type=str)
    parser.add_argument('-e', '--engine', type=str)
    args = parser.parse_args()
    return args


def obtain_sensitive_apis(file):
    sensitive_apis = []
    with open(file, 'r') as f:
        for line in f.readlines():
            if line.strip() == '':
                continue
            else:
                sensitive_apis.append(line.strip())
    return sensitive_apis


def callgraph_extraction(file):
    print(file)
    CG = nx.read_gexf(file)
    return CG


def degree_centrality_feature(file, sensitive_apis):
    sha256 = file.split('/')[-1].split('.')[0]
    CG = callgraph_extraction(file)
    node_centrality = nx.degree_centrality(CG)

    vector = []
    for api in sensitive_apis:
        if api in node_centrality.keys():
            vector.append(node_centrality[api])
        else:
            vector.append(0)

    return (sha256, vector)


def katz_centrality_feature(file, sensitive_apis):
    sha256 = file.split('/')[-1].split('.')[0]
    CG = callgraph_extraction(file)

    node_centrality = nx.katz_centrality(CG)

    vector = []
    for api in sensitive_apis:
        if api in node_centrality.keys():
            vector.append(node_centrality[api])
        else:
            vector.append(0)

    return (sha256, vector)


def closeness_centrality_feature(file, sensitive_apis):
    sha256 = file.split('/')[-1].split('.')[0]
    CG = callgraph_extraction(file)
    node_centrality = nx.closeness_centrality(CG)

    vector = []
    for api in sensitive_apis:
        if api in node_centrality.keys():
            vector.append(node_centrality[api])
        else:
            vector.append(0)

    return (sha256, vector)


def harmonic_centrality_feature(file, sensitive_apis):
    sha256 = file.split('/')[-1].split('.')[0]
    CG = callgraph_extraction(file)
    node_centrality = nx.harmonic_centrality(CG)

    vector = []
    for api in sensitive_apis:
        if api in node_centrality.keys():
            vector.append(node_centrality[api])
        else:
            vector.append(0)

    return (sha256, vector)


def obtain_dataset(dataset_path, centrality_type, sensitive_apis, noise_type):
    Vectors = []
    Labels = []

    data_filenames, gt_labels, noise_labels = read_joblib(
        os.path.join('/home/lhd/MalCleanse/Training/config',
                     'databases_' + str(noise_type) + '.conf'))
    print(len(data_filenames))
    gt_labels = np.array(gt_labels)
    noise_labels = np.array(noise_labels)
    data_filenames = [item + '.gexf' for item in data_filenames]
    oos_features = np.array([os.path.join(dataset_path, filename) for filename in data_filenames])

    benign_index = np.where(noise_labels == 0)[0]
    malware_index = np.where(noise_labels == 1)[0]
    apps_b = list(oos_features[benign_index])
    apps_m = list(oos_features[malware_index])

    pool_b = ThreadPool(15)
    pool_m = ThreadPool(15)
    if centrality_type == 'degree':
        vector_b = pool_b.map(partial(degree_centrality_feature, sensitive_apis=sensitive_apis), apps_b)
        vector_m = pool_m.map(partial(degree_centrality_feature, sensitive_apis=sensitive_apis), apps_m)
    elif centrality_type == 'katz':
        vector_b = pool_b.map(partial(katz_centrality_feature, sensitive_apis=sensitive_apis), apps_b)
        vector_m = pool_m.map(partial(katz_centrality_feature, sensitive_apis=sensitive_apis), apps_m)
    elif centrality_type == 'closeness':
        vector_b = pool_b.map(partial(closeness_centrality_feature, sensitive_apis=sensitive_apis), apps_b)
        vector_m = pool_m.map(partial(closeness_centrality_feature, sensitive_apis=sensitive_apis), apps_m)
    elif centrality_type == 'harmonic':
        vector_b = pool_b.map(partial(harmonic_centrality_feature, sensitive_apis=sensitive_apis), apps_b)
        vector_m = pool_m.map(partial(harmonic_centrality_feature, sensitive_apis=sensitive_apis), apps_m)
    else:
        print('Error Centrality Type!')

    Vectors.extend(vector_b)
    Labels.extend([0 for i in range(len(vector_b))])

    Vectors.extend(vector_m)
    Labels.extend([1 for i in range(len(vector_m))])

    return Vectors, Labels


def main(dataset_path, cetrality_type, output, noise_type='random'):
    if os.path.isfile(os.path.join(output, noise_type + '_' + cetrality_type + '_malscan_features.csv')):
        print(os.path.join(output, noise_type + '_' + cetrality_type + '_malscan_features.csv') + ' is exist!!')
    else:
        sensitive_apis_path = 'sensitive_apis.txt'
        sensitive_apis = obtain_sensitive_apis(sensitive_apis_path)

        Vectors, Labels = obtain_dataset(dataset_path, cetrality_type, sensitive_apis,
                                         noise_type=noise_type)
        feature_csv = [[] for i in range(len(Labels) + 1)]
        feature_csv[0].append('SHA256')
        feature_csv[0].extend(sensitive_apis)
        feature_csv[0].append('Label')

        for i in range(len(Labels)):
            (sha256, vector) = Vectors[i]
            feature_csv[i + 1].append(sha256)
            feature_csv[i + 1].extend(vector)
            feature_csv[i + 1].append(Labels[i])

        csv_path = os.path.join(output, noise_type + '_' + cetrality_type + '_malscan_features.csv')

        with open(csv_path, 'w', newline='') as f:
            csvfile = csv.writer(f)
            csvfile.writerows(feature_csv)


if __name__ == '__main__':
    # args = parseargs()
    # dataset_path = args.dir
    # cetrality_type = args.centrality
    # engine = args.engine
    # output = args.output
    dataset_path = '/home/public/rmt/heren/experiment/cl-exp/LHD_apk/malwhiteout_naive_pool/malscan'
    output = '/home/lhd/MalCleanse/MalWhiteout/MalScan_Feature1'
    if not os.path.isdir(output):
        os.makedirs(output)
    for cetrality_type in ['katz', ]:
        for i in range(3, 4):
            for noise_type in ['thr_variation_' + str(i) + '_20', ]:
                start = time.time()
                main(dataset_path, cetrality_type, output, noise_type=noise_type)
                end = time.time()
                print(str(noise_type) + ' use time:', end - start)

    # for cetrality_type in ['katz',  ]:
    #     for data_type in ['drebin', 'malradar']:
    #         for noise_type in ['thr_10_3', 'thr_15_3']:
    #             for noise_hyper in ['all']:
    #                 main(dataset_path, cetrality_type, output, data_type=data_type, noise_type=noise_type,
    #                      noise_hyper=noise_hyper)
    #
    #         for noise_type in ['thr_15_1', 'thr_15_2']:
    #             for noise_hyper in [10, 20]:
    #                 main(dataset_path, cetrality_type, output, data_type=data_type, noise_type=noise_type,
    #                      noise_hyper=noise_hyper)
    #
    #         for noise_type in ['thr_15_3']:
    #             for noise_hyper in [10]:
    #                 main(dataset_path, cetrality_type, output, data_type=data_type, noise_type=noise_type,
    #                      noise_hyper=noise_hyper)
    #
    #
    # for cetrality_type in ['katz', ]:
    #     for data_type in ['drebin', 'malradar']:
    #         for noise_type in ['thr_20_1', 'thr_20_2', 'thr_20_3']:
    #             for noise_hyper in [10, 20,]:
    #                 main(dataset_path, cetrality_type, output, data_type=data_type, noise_type=noise_type,
    #                      noise_hyper=noise_hyper)
    #
    #             for noise_type in ['thr_15_3']:
    #                 for noise_hyper in [20]:
    #                     main(dataset_path, cetrality_type, output, data_type=data_type, noise_type=noise_type,
    #                          noise_hyper=noise_hyper)
    #
    # for cetrality_type in ['katz', ]:
    # for data_type in ['malradar', 'drebin']:
    #     for noise_type in ['thr_18_1', 'thr_18_2', 'thr_18_3', ]:
    #         for noise_hyper in ['all', 10, 20, 30, 40]:
    #             main(dataset_path, cetrality_type, output, data_type=data_type, noise_type=noise_type,
    #                  noise_hyper=noise_hyper)

    # for data_type in ['malradar', 'drebin']:
    #     for noise_type in ['thr_22_1', 'thr_22_2', 'thr_22_3']:
    #         for noise_hyper in [50, 10, 20, 30, 40]:
    #             main(dataset_path, cetrality_type, output, data_type=data_type, noise_type=noise_type,
    #                  noise_hyper=noise_hyper)
    #
    # for data_type in ['drebin']:
    #     for noise_type in ['thr_22_1', 'thr_22_2', 'thr_22_3']:
    #         for noise_hyper in ['all']:
    #             main(dataset_path, cetrality_type, output, data_type=data_type, noise_type=noise_type,
    #                  noise_hyper=noise_hyper)
    # #
    # for data_type in ['malradar']:
    #     for noise_type in ['thr_22_1', 'thr_22_2', 'thr_22_3']:
    #         for noise_hyper in ['60']:
    #             main(dataset_path, cetrality_type, output, data_type=data_type, noise_type=noise_type,
    #                  noise_hyper=noise_hyper)
    #
    # for data_type in ['malradar', 'drebin']:
    #     for noise_type in ['thr_20_1', 'thr_20_2', 'thr_20_3']:  # , 'thr_22_1', 'thr_22_2', 'thr_22_3'
    #         for noise_hyper in [50, 'all']:
    #             main(dataset_path, cetrality_type, output, data_type=data_type, noise_type=noise_type,
    #                  noise_hyper=noise_hyper)
    # #
    # for data_type in ['drebin', 'malradar']:
    #     for noise_type in ['thr_25_1', 'thr_25_2', 'thr_25_3']:
    #         for noise_hyper in ['10', '20', 30, 40, 50, 60]:
    #             main(dataset_path, cetrality_type, output, data_type=data_type, noise_type=noise_type,
    #                  noise_hyper=noise_hyper)
