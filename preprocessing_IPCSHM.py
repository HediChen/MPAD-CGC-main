import os
import numpy as np
import scipy.io as sio
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import gc

# ==================== Normalization Functions ====================

def nan_mean_normalization(data):
    """In-place normalization to minimize memory copies"""
    if (np.nanmax(data) - np.nanmin(data)) == 0:
        np.nan_to_num(data, copy=False, nan=0.0)
        return data
    else:
        data = (data - np.nanmean(data)) / (np.nanmax(data) - np.nanmin(data))
        np.nan_to_num(data, copy=False, nan=0.0)
        return data

# ==================== Original Correct Loading (Memory Optimized) ====================

def IPCSHM_data_memory_efficient(balanced=True):
    """
    Memory-efficient version of original loading method.
    Keeps the exact same label-data matching logic but uses streaming.
    """
    
    # Load labels (same as original)
    y_matData_o = h5py.File('./data/IPCSHM/label_new_v2.mat', 'r')
    y_matdata = y_matData_o['info']['label']['manual']

    y_data_list = []
    for num in range(38):
        ref = y_matdata[num][0]
        y_data_num = np.asarray(y_matData_o[ref]).astype(np.int64)
        y_data_num = y_data_num - 1
        y_data_list.append(y_data_num)
    
    y_data = np.concatenate(y_data_list, axis=1)
    y_data = np.reshape(y_data, (28272, 1))
    y_matData_o.close()
    
    # Load and process x_data (same sequence as original)
    folderpath = './data/IPCSHM/data/'
    datelist = sorted(os.listdir(folderpath))
    
    x_data_list = []
    
    for i in range(len(datelist)):
        date_folder = datelist[i]
        files_in_folder = sorted(os.listdir(folderpath + date_folder))
        
        for j in range(len(files_in_folder)):
            filename = files_in_folder[j]
            file_path = os.path.join(folderpath, date_folder, filename)
            
            print(f"Loading {date_folder}/{filename}")
            
            # Load and process individual file
            data = sio.loadmat(file_path)
            x_data_date = np.asarray(data['data'], dtype=np.float32)  # [72000, 38]
            x_data_date = np.transpose(x_data_date)  # [38, 72000]
            
            # Normalize each channel in-place
            x_data_date_temp = []
            for k in range(x_data_date.shape[0]):
                temp = nan_mean_normalization(x_data_date[k].copy())
                x_data_date_temp.append(temp)
            
            x_data_date = np.asarray(x_data_date_temp, dtype=np.float32)
            x_data_list.append(x_data_date)
            
            # Free memory from large intermediate arrays
            del data, x_data_date
            gc.collect()
    
    # Convert list to array
    x_data = np.array(x_data_list, dtype=np.float32)  # [744, 38, 72000]
    
    # Downsample
    x_data = x_data[:, :, ::50]  # [744, 38, 1440]
    
    print("Whether there are infinite values in the data: {}".format(np.isinf(x_data).any()))
    print("Whether there are nan values in the data: {}".format(np.isnan(x_data).any()))
    print(x_data.shape)
    
    # Reshape (same as original)
    x_data = np.reshape(x_data, (744 * 38, 1440))  # [28272, 1440]
    
    # Shuffle data (same as original)
    np.random.seed(15)
    indices = np.arange(x_data.shape[0])
    np.random.shuffle(indices)
    x_data = x_data[indices]
    y_data = y_data[indices]
    
    # Balance data (same as original)
    if balanced == True:
        x_train_list = []
        y_train_list = []
        x_val_list = []
        y_val_list = []
        
        for c in range(13):
            index = np.where(np.squeeze(y_data, axis=1) == c)
            temp_y = np.reshape(y_data[index], (-1, 1))
            temp_x = np.reshape(x_data[index], (-1, 1440))
            
            # Segment data
            train_length = int(temp_x.shape[0] * 0.8)
            
            if c == 0:
                x_train_list.append(temp_x[:train_length])
                y_train_list.append(temp_y[:train_length])
                x_val_list.append(temp_x[train_length:])
                y_val_list.append(temp_y[train_length:])
            elif c == 8:
                x_train_list.append(temp_x[:train_length])
                y_train_list.append(np.full((train_length, 1), 1))
                x_val_list.append(temp_x[train_length:])
                y_val_list.append(np.full((int(temp_x.shape[0] * (1-0.8)), 1), 1))
            elif c >= 9 and c <= 12:
                x_train_list.append(temp_x[:train_length])
                y_train_list.append(np.full((train_length, 1), c-1))
                x_val_list.append(temp_x[train_length:])
                y_val_list.append(np.full((int(temp_x.shape[0] * (1-0.8)), 1), c-1))
            else:
                x_train_list.append(temp_x[:train_length])
                y_train_list.append(temp_y[:train_length])
                x_val_list.append(temp_x[train_length:])
                y_val_list.append(temp_y[train_length:])
            
            print(f"Class {c}: train_length={train_length}, val_length={temp_x.shape[0] - train_length}")
            
            # Free memory
            del temp_x, temp_y
            gc.collect()
        
        # Concatenate using lists instead of repeated concatenate
        x_train = np.vstack(x_train_list)
        y_train = np.vstack(y_train_list)
        x_val = np.vstack(x_val_list)
        y_val = np.vstack(y_val_list)
        
        del x_train_list, y_train_list, x_val_list, y_val_list
        gc.collect()
        
        print(f"Final train shape: {x_train.shape}, {y_train.shape}")

    if balanced == False:
        # Segment data
        print("balanced = False")
        x_train = x_data[0:22618]
        y_train = y_data[0:22618].squeeze()
        x_val = x_data[22618:28272]
        y_val = y_data[22618:28272].squeeze()

    # Convert to PyTorch tensors
    x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(y_train, dtype=torch.int64)
    x_val = torch.tensor(x_val, dtype=torch.float32).unsqueeze(1)
    y_val = torch.tensor(y_val, dtype=torch.int64)

    return x_train, y_train, x_val, y_val

def save_IPCSHM(filepath, x_train, y_train, x_val, y_val):
    """Save data to .mat file"""
    sio.savemat(filepath, {'x_train': x_train.numpy(),
                           'y_train': y_train.numpy(),
                           'x_val': x_val.numpy(),
                           'y_val': y_val.numpy()})

if __name__ == '__main__':
    x_train, y_train, x_val, y_val = IPCSHM_data_memory_efficient(balanced=True)
    filepath = './data/IPCSHM/train/IPCSHM_shuffle_segment_balanced.mat'
    save_IPCSHM(filepath, x_train, y_train, x_val, y_val)
    print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)