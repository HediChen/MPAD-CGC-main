'''
这是数据预处理脚本，包含数据加载、特征提取、归一化和划分训练验证集等功能。
功能概述:
    1. 从指定文件夹加载.mat格式的时间序列数据和对应标签
    2. 对每条时间序列进行缺失值处理和归一化
    3. 提取统计特征并与原始序列拼接
    4. 将标签转换为多标签二进制矩阵格式
    5. 划分训练集和验证集，支持类别平衡选项
    6. 返回处理后的数据和标签，适配Pytorch格式
'''

import os.path
import numpy as np
import pandas as pd
import scipy.io as sio
import h5py
# import hdf5storage
import matplotlib.pyplot as plt
import torch
from unit.feacture_extractor import extractor # 导入特征提取函数
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, SVMSMOTE, BorderlineSMOTE, KMeansSMOTE


def nan_mean_normalization(data):
    # 功能: 对输入数据进行缺失值处理与归一化(均值中心化+极差缩放)
    # 若数据极差为0(全部相同或仅NaN), 则直接填充为0数组
    if (np.nanmax(data) - np.nanmin(data)) == 0:
        np.nan_to_num(data, copy=False, nan=0.0)  # 将NaN替换为0
        data = np.zeros_like(data)
        return data
    else:
        # 标准化: (x - nan均值) / (nan最大 - nan最小), 将数据压缩到[-0.5,0.5]附近分布
        data = (data - np.nanmean(data)) / (np.nanmax(data)-np.nanmin(data))
        np.nan_to_num(data, copy=False, nan=0.0)  # 再次处理潜在NaN
        return data

def convert_to_MultiLables(label_list, num_classes, normal_class=False):
    # 功能: 将原始标签结构转换为多标签二进制矩阵格式
    # 输入: label_list为原始标签集合(num_samples x ?), 每条记录前两列存放类别值
    # normal_class参数控制是否保留正常类0
    labels = np.zeros((len(label_list), num_classes))
    for i in range(len(label_list)):
        label = label_list[i]
        for j in label[:2]:  # 只取前两个可能的故障标签
            if j.size == 0:  # 空标签跳过
                continue
            elif normal_class == False and j == 0:  # 不保留正常类则跳过0
                continue
            elif normal_class == False and j != 0:  # 故障类索引向前移动1
                labels[i, j-1] = 1
            else:
                # 保留正常类时按原索引写入
                labels[i, j] = 1
    return labels

def data_preprocessing(balanced = False, platform = 'pytorch', normal_class=True, method='train'):
    # 功能: 数据与标签的加载、特征提取、归一化、划分训练验证集
    dict_classes = {0: '1_missing', 1: '2_trend', 2: '3_drift'}  # 类别映射
    num_classes = len(dict_classes)

    feature_length = 9  # 额外抽取特征的长度
    input_data_length = 1008 + feature_length  # 原序列长度 + 特征长度

    # 数据与标签所在文件夹路径
    datafolder = 'D:\\ShihongChen\\1-code\\7-anomaly_classification\\data\\anomaly\\fullData_Feb2Oct_20250621\\Data'
    labelfolder = 'D:\\ShihongChen\\1-code\\7-anomaly_classification\\data\\anomaly\\fullData_Feb2Oct_20250621\\Label\\Label'
    datafiles = os.listdir(datafolder)
    datafiles.sort()
    datafiles = [f for f in datafiles if f.endswith('.mat')]  # 仅保留mat文件
    data_name = [f.split('.')[0] for f in datafiles]  # 去掉后缀得到基名

    # 步骤1: 加载输入数据并进行归一化 + 特征提取 + 拼接
    x_data = []
    for n in data_name:
        x_matData_o = sio.loadmat(os.path.join(datafolder, n+'.mat'))
        x_matdata = x_matData_o['data']  # 结构: 行遍历不同样本
        for num in range(x_matdata.shape[0]):
            x_data_num = np.array(x_matdata[num][0]).T  # 转置成(长度,)向量
            x_data_num = nan_mean_normalization(x_data_num)  # 归一化
            # ！！！关键步骤：提取信号的9个统计特征并拼接
            features = extractor(np.squeeze(x_data_num), length = feature_length)  # 抽取统计特征
            x_data_num = np.append(x_data_num, features)  # 拼接原序列与特征
            x_data.append(x_data_num)
    x_data = np.asarray(x_data).astype(np.float64)
    x_data = np.reshape(x_data, (-1, input_data_length))  # 统一形状
    print("Whether there are infinite values in the data: {}".format(np.isinf(x_data).any()))
    print("Whether there are nan values in the data: {}".format(np.isnan(x_data).any()))
    print(x_data.shape)

    # 步骤2: 加载标签并转换为多标签矩阵
    y_data = []
    for n in data_name:
        y_matData_o = sio.loadmat(os.path.join(labelfolder, n+'_Label.mat'))
        y_matdata = y_matData_o['label']
        y_data_temp = np.array(y_matdata)
        multilabels = convert_to_MultiLables(y_data_temp, num_classes, normal_class=normal_class)
        y_data.append(multilabels)
    y_data = np.asarray(y_data).astype(np.int64)
    y_data = np.reshape(y_data, (-1, num_classes))

    # 方法分支: 测试模式直接返回全部数据与索引
    if method == 'test':
        x = np.reshape(x_data, [-1, input_data_length])
        y = np.reshape(y_data, [-1, num_classes])
        indices = np.arange(x_data.shape[0])
        x_temp = []  # 兼容返回结构, 空占位
        y_temp = []
        return x_temp, y_temp, x, y, indices

    # 训练模式
    elif method == 'train':
        # 步骤3: 打乱数据
        np.random.seed(1)
        indices = np.arange(x_data.shape[0])
        np.random.shuffle(indices)
        x_data = x_data[indices]
        y_data = y_data[indices]

        # 步骤4: (未做采样)划分训练与验证集 (80%:20%)
        if balanced == False:
            train_length = int(x_data.shape[0] * 0.8)
            print("balanced = False")
            x_train = np.expand_dims(x_data[0:train_length],axis=1)  # 扩展维度方便后续兼容
            y_train = y_data[0:train_length]
            x_val = np.expand_dims(x_data[train_length:],axis=1)
            y_val = y_data[train_length:]
            # 打印各类在训练与验证集的样本数量
            for c in range(num_classes):
                quality_train = np.sum(y_train[:,c])
                quality_val = np.sum(y_val[:,c])
                print("length of class {} for train is {}".format(dict_classes[c], quality_train))
                print("length of class {} for val is {}".format(dict_classes[c], quality_val))

        # 平台适配: Pytorch格式展平为二维
        if platform == 'pytorch':
            import torch
            x_train = np.reshape(x_train, [-1, input_data_length])
            x_val = np.reshape(x_val, [-1, input_data_length])
            val_indices = indices[train_length:]  # 验证集对应原始顺序索引
            return x_train, y_train, x_val, y_val, val_indices

if __name__ == '__main__':
    # 主入口: 执行数据预处理, 不保留正常类标签, 不做类别平衡, 返回训练验证数据
    x_train, y_train, x_val, y_val, val_indices = data_preprocessing(balanced=False, normal_class=False, platform='pytorch')
    print('done')


