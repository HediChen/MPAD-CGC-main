import os.path
import numpy as np
import pandas as pd
import scipy.io as sio
import h5py
# import hdf5storage
import matplotlib.pyplot as plt
import torch
from unit.feacture_extractor import extractor
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, SVMSMOTE, BorderlineSMOTE, KMeansSMOTE


def nan_mean_normalization(data):
    if (np.nanmax(data) - np.nanmin(data)) == 0:
        np.nan_to_num(data, copy=False, nan=0.0)
        data = np.zeros_like(data)
        return data
    else:
        data = (data - np.nanmean(data)) / (np.nanmax(data)-np.nanmin(data)) # 归一化到-0.5到0.5
        np.nan_to_num(data, copy=False, nan=0.0)
        return data

def convert_to_MultiLables(label_list, num_classes, normal_class=False):
    labels = np.zeros((len(label_list), num_classes))
    for i in range(len(label_list)):
        label = label_list[i]
        for j in label[:2]:
            if j.size == 0: # to check that an array is empty
                continue
            elif normal_class == False and j == 0:
                continue
            elif normal_class == False and j != 0:
                labels[i, j-1] = 1
            else:
                labels[i, j] = 1
        # print("num={}, label={}, multilabel={}".format(i,label[:2],labels[i]))

        # labels[i, label_list[i]] = 1
    return labels

def data_preprocessing(balanced = False, platform = 'pytorch', normal_class=True, method='train'):
    dict_classes = {0: '1_missing', 1: '2_trend', 2: '3_drift'}
    # dict_classes = {0: '0_normal', 1: '1_missing', 2: '2_trend', 3: '3_drift'}
    # dict_classes = {0: '2_trend', 1: '3_drift'}
    num_classes = len(dict_classes)
    # dict_classes = {1: '1_missing', 2: '2_trend', 3: '3_drift'}
    feature_length = 9
    input_data_length = 1008 + feature_length
    # datafolder = './data/anomaly/fullData_Feb2Oct_20250621/Data'
    # labelfolder = './data/anomaly/fullData_Feb2Oct_20250621/Label/Label'
    datafolder = 'D:\\ShihongChen\\1-code\\7-anomaly_classification\\data\\anomaly\\fullData_Feb2Oct_20250621\\Data'
    labelfolder = 'D:\\ShihongChen\\1-code\\7-anomaly_classification\\data\\anomaly\\fullData_Feb2Oct_20250621\\Label\\Label'
    datafiles = os.listdir(datafolder)
    datafiles.sort()
    datafiles = [f for f in datafiles if f.endswith('.mat')]
    # datafiles = ['G_01.mat', 'G_02.mat', 'G_03.mat', 'G_04.mat', 'G_05.mat',
    #              'G_06.mat', 'G_07.mat', 'G_08.mat', 'G_09.mat', 'G_10.mat',
    #              'G_11.mat']
    data_name = [f.split('.')[0] for f in datafiles]

    # load input data
    x_data = []
    for n in data_name:
        x_matData_o = sio.loadmat(os.path.join(datafolder, n+'.mat'))
        x_matdata = x_matData_o['data']
        for num in range(x_matdata.shape[0]):
            x_data_num = np.array(x_matdata[num][0]).T
            x_data_num = nan_mean_normalization(x_data_num)
            features = extractor(np.squeeze(x_data_num), length = feature_length)
            x_data_num = np.append(x_data_num, features)
            x_data.append(x_data_num)
    x_data = np.asarray(x_data).astype(np.float64)
    x_data = np.reshape(x_data, (-1, input_data_length))
    print("Whether there are infinite values in the data: {}".format(np.isinf(x_data).any()))
    print("Whether there are nan values in the data: {}".format(np.isnan(x_data).any()))
    print(x_data.shape)

    # load labels
    y_data = []
    for n in data_name:
        y_matData_o = sio.loadmat(os.path.join(labelfolder, n+'_Label.mat'))
        y_matdata = y_matData_o['label']
        y_data_temp = np.array(y_matdata)
        multilabels = convert_to_MultiLables(y_data_temp, num_classes, normal_class=normal_class)
        y_data.append(multilabels)
    y_data = np.asarray(y_data).astype(np.int64)
    y_data = np.reshape(y_data, (-1, num_classes))
    y_1149 = y_data[1149]
    # y_1116 = y_data[1116]
    # y_1128= y_data[1128]
    # y_1193 = y_data[1193]

    if method == 'test':
        x = np.reshape(x_data, [-1, input_data_length])
        y = np.reshape(y_data, [-1, num_classes])
        indices = np.arange(x_data.shape[0])
        x_temp = []
        y_temp = []
        return x_temp, y_temp, x, y, indices


    elif method == 'train':
        # Shuffle data
        np.random.seed(1)
        indices = np.arange(x_data.shape[0])
        np.random.shuffle(indices)
        x_data = x_data[indices]
        y_data = y_data[indices]

        # balance data
        if balanced == False:
            # Segment data
            train_length = int(x_data.shape[0] * 0.8)
            print("balanced = False")
            x_train = np.expand_dims(x_data[0:train_length],axis=1)  # [-1,1,1008]
            y_train = y_data[0:train_length]  # [-1,5]
            x_val = np.expand_dims(x_data[train_length:],axis=1)  # [-1,1,1008]
            y_val = y_data[train_length:]  # [-1,5]
            
            
            for c in range(num_classes):
                quality_train = np.sum(y_train[:,c])
                quality_val = np.sum(y_val[:,c])
                print("length of class {} for train is {}".format(dict_classes[c], quality_train))
                print("length of class {} for val is {}".format(dict_classes[c], quality_val))

        if platform == 'pytorch':
            # For Pytorch
            import torch
            x_train = np.reshape(x_train, [-1, input_data_length])
            x_val = np.reshape(x_val, [-1, input_data_length])
            # y_train_label = [y_train[:, key] for key in dict_classes.keys()]
            # y_val_label = [y_val[:, key] for key in dict_classes.keys()]
            # y_train_label = {dict_classes[key]: y_train[:, key] for key in dict_classes.keys()}
            # y_val_label = {dict_classes[key]: y_val[:, key] for key in dict_classes.keys()}

            # Return validation original indices (in raw dataset order)
            val_indices = indices[train_length:]

            return x_train, y_train, x_val, y_val, val_indices

        # elif platform == 'tf':
        #     # For TensorFlow
        #     # from tensorflow.keras.utils import to_categorical
        #     output_info = [(2, dict_classes[key]) for key in sorted(dict_classes.keys())]
        #     x_train = np.reshape(x_train, [-1, 1008])
        #     x_val = np.reshape(x_val, [-1, 1008])
        #     # One-hot encoding categorical labels
        #     # For train labels
        #     dict_train_labels = {
        #         dict_classes[key]: to_categorical(y_train[:, key], num_classes=2) for key in sorted(dict_classes.keys())
        #     }
        #     dict_val_labels = {
        #         dict_classes[key]: to_categorical(y_val[:, key], num_classes=2) for key in sorted(dict_classes.keys())
        #     }
        #     y_train_label = [dict_train_labels[key] for key in sorted(dict_train_labels.keys())]
        #     y_val_label = [dict_val_labels[key] for key in sorted(dict_val_labels.keys())]

        #     return x_train, y_train_label, x_val, y_val_label, output_info



if __name__ == '__main__':
    # x_train, y_train, x_val, y_val, output_info = data_preprocessing(balanced=False)
    x_train, y_train, x_val, y_val, val_indices = data_preprocessing(balanced=False, normal_class=False, platform='pytorch')
    print('done')


