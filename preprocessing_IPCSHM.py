import os.path
import numpy as np
import pandas as pd
import scipy.io as sio
import h5py
# import hdf5storage
import matplotlib.pyplot as plt

def normalization(data):
    data = (data - np.min(data)) / (np.max(data) - np.min(data)) # 归一化到0-1
    return data

def mean_normalization(data):
    # print(data)
    if (np.max(data) - np.min(data)) == 0:
        return data
    else:
        data = (data - np.mean(data)) / (np.max(data) - np.min(data)) # 归一化到-0.5到0.5
        print(data)
        return data

def nan_mean_normalization(data):
    if (np.nanmax(data) - np.nanmin(data)) == 0:
        np.nan_to_num(data, copy=False, nan=0.0)
        return data
    else:
        data = (data - np.nanmean(data)) / (np.nanmax(data)-np.nanmin(data)) # 归一化到-0.5到0.5
        np.nan_to_num(data, copy=False, nan=0.0)
        return data

def normalization_adj(data):
    print(data)
    if (np.max(data) - np.min(data)) == 0:
        return data
    else:
        data = (data) / (np.max(data) - np.min(data)) # 归一化到-1到1
        print(data)
        return data

def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)    
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr

def standardization(data):
    data = (data - np.mean(data)) / np.std(data) # 标准化
    return data

def standardization_adj(data):
    if np.std(data) == 0:
        return data
    else:
        data = data / np.std(data) # 标准化
        return data

np.set_printoptions(threshold=np.inf)

# train_matData = sio.loadmat('./data/Gzhu/acc_gzhu_15%.mat')
# x_data = train_matData['acc_gzhu']
# y_data = train_matData['label']
# x_data = np.transpose(np.array(x_data))
# y_data = np.transpose(np.array(y_data))
import h5py
import numpy as np
import os
import scipy.io as sio
import torch

def IPCSHM_data(balanced = True):
    y_matData_o = h5py.File('./data/IPCSHM/label.mat', 'r')
    y_matdata = y_matData_o['info']['label']['manual']

    for num in range(38):
        ref = y_matdata[num][0]
        y_data_num = np.asarray(y_matData_o[ref]).astype(np.int64)
        y_data_num = y_data_num - 1
        # num_classes = 7
        # eye_matrix = np.eye(num_classes)
        # y_data_num = eye_matrix[y_data_num].squeeze(1)

        if num == 0:
            y_data = y_data_num  # Adjusting labels to start from 0
        else:
            y_data = np.concatenate((y_data, y_data_num), axis=1)
    y_data = np.reshape(y_data, (28272, 1))

    folderpath = './data/IPCSHM/data/'
    datelist = os.listdir(folderpath)
    print(datelist)
    x_data = []
    for i in range(len(datelist)):
        for j in range(len(os.listdir(folderpath + datelist[i]))):
            print(os.listdir(folderpath + datelist[i])[j])
            data = sio.loadmat(folderpath + datelist[i] + '/' + os.listdir(folderpath + datelist[i])[j])
            x_data_date = np.asarray(data['data'])  # [72000,38]
            # print(x_data_date.shape)
            x_data_date = np.transpose(x_data_date)  # [38,72000]

            # if i == 0 and j == 0:
            #     x_data = x_data_date
            # else:
            #     x_data = np.concatenate((x_data, x_data_date), axis=0)

            # if i == 0 and j == 0:
            #     x_data = np.expand_dims(nan_mean_normalization(x_data_date[0]), axis=0)
            #     for k in range(x_data_date.shape[0])[1:]:
            #         # print(x_data_date.shape[0])
            #         x_data = np.concatenate((x_data, 
            #                                  np.expand_dims(nan_mean_normalization(x_data_date[k]), 
            #                                                 axis=0)), 
            #                                  axis=0)
            # else:
            #     for k in range(x_data_date.shape[0]):
            #         # print(x_data_date.shape[0])
            #         x_data = np.concatenate((x_data, 
            #                                  np.expand_dims(nan_mean_normalization(x_data_date[k]), 
            #                                                 axis=0)), 
            #                                  axis=0)
                    
            # print(x_data.shape)
            # if i == 5 and j == 7:
            #     plt.plot(x_data_date[4])
            #     plt.title("{}".format('4830'))
            #     plt.show()
            x_data_date_temp = []
            for k in range(x_data_date.shape[0]):
                # print(x_data_date.shape[0])
                temp = nan_mean_normalization(x_data_date[k])
                x_data_date_temp.append(temp)
            x_data_date = np.asarray(x_data_date_temp)
          
            x_data.append(x_data_date)

    x_data = np.array(x_data)  # [744,38,72000]
    print("Whether there are infinite values in the data: {}".format(np.isinf(x_data).any()))
    print("Whether there are nan values in the data: {}".format(np.isnan(x_data).any()))
    print(x_data.shape)
    x_data = np.reshape(x_data, (744 * 38, 72000))  # [28272,72000]
    # plt.plot(x_data[23276])
    # print(x_data[23276])
    # plt.title("{}".format('23276'))
    # plt.show()
    # plt.subplot(4, 1, 1)
    # plt.plot(x_data[23886])
    # plt.title("{}".format("23886"))
    # plt.subplot(4, 1, 2)
    # plt.plot(x_data[5759])
    # plt.title("{}".format("5759"))
    # plt.subplot(4, 1, 3)
    # plt.plot(x_data[3207])
    # plt.title("{}".format("3207"))
    # plt.subplot(4, 1, 4)
    # plt.plot(x_data[13109])
    # plt.title("{}".format("non_meaning"))
    # plt.show()

    # Shuffle data
    np.random.seed(15)
    indices = np.arange(x_data.shape[0])
    np.random.shuffle(indices)
    x_data = x_data[indices]
    y_data = y_data[indices]

    # balance data
    if balanced == True:
        for c in range(7):
            # test 
            # if c == 1:
            #     index_0 = np.where(np.squeeze(y_data, axis=1) == c)
            index = np.where(np.squeeze(y_data, axis=1) == c)
            temp_y = np.reshape(y_data[index], (-1, 1))
            temp_x = np.reshape(x_data[index], (-1, 72000))
            # Segment data
            train_length = int(temp_x.shape[0] * 0.8)
            if c == 0:
                y_train = temp_y[:train_length]
                x_train = temp_x[:train_length]
                y_val = temp_y[train_length:]
                x_val = temp_x[train_length:]
            else:
                y_train = np.concatenate((y_train,temp_y[:train_length]), axis=0)
                x_train = np.concatenate((x_train,temp_x[:train_length]), axis=0)
                y_val = np.concatenate((y_val,temp_y[train_length:]), axis=0)
                x_val = np.concatenate((x_val,temp_x[train_length:]), axis=0)
                
            print(y_train.shape, x_train.shape)
            print("length of class {} for train is {}".format(c, train_length))
            print("length of class {} for val is {}".format(c, temp_x.shape[0] - train_length))
        print(y_train.shape)

    if balanced == False:
        # Segment data
        print("balanced = False")
        x_train = x_data[0:22618]
        y_train = y_data[0:22618].squeeze()  # Squeeze to remove the last dimension
        x_val = x_data[22618:28272]
        y_val = y_data[22618:28272].squeeze()  # Squeeze to remove the last dimension

    # Convert to PyTorch tensors
    #torch.Size([22615, 1, 72000]) torch.Size([22615, 1]) torch.Size([5657, 1, 72000]) torch.Size([5657, 1])
    x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)  # torch.Size([22615, 1, 72000])
    # x_train = torch.where(torch.isnan(x_train), torch.zeros_like(x_train), x_train)  # Replace NaNs with zeros
    y_train = torch.tensor(y_train, dtype=torch.int64)  # torch.Size([22615, 1])
    x_val = torch.tensor(x_val, dtype=torch.float32).unsqueeze(1)  # torch.Size([5657, 1, 72000])
    # x_val = torch.where(torch.isnan(x_val), torch.zeros_like(x_val), x_val)  # Replace NaNs with zeros
    y_val = torch.tensor(y_val, dtype=torch.int64)  # torch.Size([5657, 1])

    # return x_train, y_train, x_val, y_val, indices, index_0
    return x_train, y_train, x_val, y_val

def save_IPCSHM(filepath, x_train, y_train, x_val, y_val):

# hdf5storage.savemat('./data/IPCSHM/train/IPCSHM_shuffle_segment.mat', {'x_train': x_train,
#                                                          'y_train': y_train,
#                                                          'x_val': x_val,
#                                                          'y_val': y_val},
#                                                          format=7.3,
#                                                          matlab_compatible=True,
#                                                          compress=False )
    # sio.savemat(filepath, {'x_train': x_train.numpy(),
    #                        'y_train': y_train.numpy(),
    #                          'x_val': x_val.numpy(),
    #                          'y_val': y_val.numpy()})
    sio.savemat(filepath, {'x_val': x_val.numpy(),
                             'y_val': y_val.numpy()})

if __name__ == '__main__':
    x_train, y_train, x_val, y_val = IPCSHM_data(balanced=True)
    filepath = './data/IPCSHM/train/IPCSHM_shuffle_segment_balanced_val.mat'
    save_IPCSHM(filepath, x_train, y_train, x_val, y_val)
    # a1 = x_train.squeeze(1).numpy()[10870]
    # print(a1, a1.shape)
    # a2 = x_train.squeeze(1).numpy()[10871]
    # a3 = x_train.squeeze(1).numpy()[10872]
    # a4 = x_train.squeeze(1).numpy()[10873]
    # plt.subplot(4, 1, 1)
    # plt.plot(a1)
    # plt.title("{}".format(y_train.numpy()[10870]))
    # plt.subplot(4, 1, 2)
    # plt.plot(a2)
    # plt.title("{}".format(y_train.numpy()[10871]))
    # plt.subplot(4, 1, 3)
    # plt.plot(a3)
    # plt.title("{}".format(y_train.numpy()[10872]))
    # plt.subplot(4, 1, 4)
    # plt.plot(a4)
    # plt.title("{}".format(y_train.numpy()[10873]))
    # plt.show()
    # print("污染数据index:{},{},{},{}".format(
    #                                      indices[index[0][10]],
    #                                      indices[index[0][11]],
    #                                      indices[index[0][12]],
    #                                      indices[index[0][13]]))

    # print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)

