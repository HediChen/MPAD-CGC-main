import numpy as np
import scipy.io as sio
import os
import sys
sys.path.append('./')  # Add the parent directory to the path
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
from torch.utils.data import DataLoader, TensorDataset
from pytorchModels.ple_Inception_Features import PLE

def nan_mean_normalization(data):
    if (np.nanmax(data) - np.nanmin(data)) == 0:
        np.nan_to_num(data, copy=False, nan=0.0)
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

def data_preprocessing_by_sensor(datafolder, sensor_id = 'C_27'):
    dict_classes = {0: '1_missing', 1: '2_trend', 2: '3_drift'}
    # dict_classes = {0: '0_normal', 1: '1_missing', 2: '2_trend', 3: '3_drift'}
    # dict_classes = {0: '2_trend', 1: '3_drift'}
    num_classes = len(dict_classes)
    # dict_classes = {1: '1_missing', 2: '2_trend', 3: '3_drift'}
    feature_length = 9
    input_data_length = 1008 + feature_length

    # load input data
    x_data = []
    x_matData_o = sio.loadmat(os.path.join(datafolder, sensor_id+'.mat'))
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


    x = np.reshape(x_data, [-1, input_data_length])
    indices = np.arange(x_data.shape[0])
    return x, indices




def save_labels_to_mat(sensor_id: int, labels: np.ndarray, save_dir: str):
    """
    Save labels for a sensor to a .mat file compatible with MATLAB.
    
    Parameters:
    - sensor_id (int): Sensor number (e.g., 27 for sensor C_27).
    - labels (np.ndarray): A (N, 3) numpy array with binary labels for three tasks.
    - save_dir (str): Directory where the .mat file will be saved.
    """
    assert labels.ndim == 2 and labels.shape[1] == 3, "Labels must be a (N, 3) array."
    os.makedirs(save_dir, exist_ok=True)

    file_name = f"{sensor_id}_Label.mat"
    file_path = os.path.join(save_dir, file_name)
    
    # Save the array under the key 'label' to match the MATLAB loading code
    sio.savemat(file_path, {'label': labels})
    print(f"Saved {labels.shape[0]} labels to {file_path}")

if __name__ == "__main__":

    # Simulate 100 samples of 3-task binary labels
    # sensor_id = 27
    # labels = np.random.randint(0, 2, size=(100, 3))  # Example: 100 samples
    datafolder = './sensor_labels/windowed_data_20250606(3)\Data'
    datafiles = os.listdir(datafolder)
    datafiles.sort()
    datafiles = [f for f in datafiles if f.endswith('.mat')]
    id_list = [f.split('.')[0] for f in datafiles]
    save_dir = "./sensor_labels"

    # Load sensor data
    for sensor_id in id_list:
        print(sensor_id)
        x_data, indices = data_preprocessing_by_sensor(datafolder, sensor_id = sensor_id)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x_val_tensor = torch.tensor(x_data, dtype=torch.float32).unsqueeze(1)
        val_dataset = TensorDataset(x_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)

        num_features = x_val_tensor[:,:,1008:].shape[-1]
        model = PLE(inputs_dim=num_features,
                    labels_dict={
                        '1_missing': 2,
                        '2_trend': 2,
                        '3_drift': 2,
                    },
                    dnn_dropout=0.2,
                    num_shared_experts=4,
                    num_task_experts=2,
                    expert_hidden_units=[128],
                    tower_hidden_units=[128, 64],
                    device='cuda')

        model.load_state_dict(torch.load('saved_models/PLE_mode_2025-05-30-10-29-56(best)/model_2025-05-30-10-37-15_fold_1.pth', map_location=device))
        model.to(device)
        model.eval()

        dict_classes = {0: '1_missing', 1: '2_trend', 2: '3_drift'}
        y_true, y_prob, y_pred = {}, {}, {}

        with torch.no_grad():
            for x_batch in val_loader:
                x_batch = x_batch[0]
                x_batch = x_batch.to(device)
                outputs = model(x_batch)

                for i, task in dict_classes.items():
                    probs = torch.softmax(outputs[task], dim=1)[:, 1].cpu().numpy()
                    preds = torch.argmax(outputs[task], dim=1).cpu().numpy()

                    y_prob.setdefault(task, []).extend(probs)
                    y_pred.setdefault(task, []).extend(preds)
        labels = []
        labels.append(y_pred['1_missing'])
        labels.append(y_pred['2_trend'])
        labels.append(y_pred['3_drift'])
        labels = np.array(labels).T

        save_labels_to_mat(sensor_id, labels, save_dir)

    labelfolder = "./sensor_labels"
    n = "C_27"
    y_matData_o = sio.loadmat(os.path.join(labelfolder, n + '_Label.mat'))
    y_matdata = y_matData_o['label']
    y_data_temp = np.array(y_matdata)
    print('done')
