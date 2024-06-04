# -*- coding: utf-8 -*- 
# @Time : 2023/11/23 12:53 
# @Author : DirtyBoy 
# @File : feature_extractor.py
import os
from core.feature.feature_extraction import DrebinFeature
from core.config import config, logging

if __name__ == '__main__':
    android_features_saving_dir = '/home/lhd/apk/drebin1'
    intermediate_data_saving_dir = '/home/lhd/apk/drebin2'
    feature_extractor = DrebinFeature(android_features_saving_dir, intermediate_data_saving_dir, update=False,
                                  proc_number=24)

    dir = '/home/public/rmt/heren/experiment/cl-exp/LHD_apk/apk/Robust_model/trainset/malradar/malware'
    feature_extractor.feature_extraction(dir)
