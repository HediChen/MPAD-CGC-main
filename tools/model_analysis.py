'''
这个脚本是用于分析打印论文中的图12，图11和图10的代码。
FIGURE 12: Contribution ratio between task-specific expert and shared expert.
FIGURE 11: Representative output features from the shared multiscale expert for multi-pattern anomalous samples. 
(a) Anomalous data with missing and trend patterns. 
(b) Anomalous data with trend and drift patterns.
FIGURE 10: Visualization of statistics feature contributions across different anomaly types. 
(a) Missing anomaly data. (b) Trend anomaly data. (c) Drift anomaly data.

'''

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('./')
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
# from pytorchModels.cgc_dnn import CGC
# from pytorchModels.cgc_cnn import CGC
# from pytorchModels.cgc_Inception import CGC
# from pytorchModels.hardShare_Inception import HardShare
# from pytorchModels.mmoe_Inception import MMoe
from pytorchModels.ple_Inception_Features_analysis import PLE
# from pytorchModels.moe_Inception import Moe
from preprocessing_addFeatures import data_preprocessing
from torch.utils.data import DataLoader, TensorDataset
from ghostml import ghost  # Corrected import based on GHOST repo

import torch
import pandas as pd
from collections import defaultdict

plt.rcParams['font.family'] = 'Times New Roman'  # or 'Times New Roman', 'SimHei', 'Arial', etc.

def analyze_expert_utilization_stacked(csv_path, save_path=None):
    """
    Plot stacked bar chart showing expert utilization per task.
    Shared expert (Expert0) and task-specific expert (Expert1) are stacked vertically.
    
    Parameters:
    - csv_path (str): Path to the gate weights CSV file.
    - save_path (str or None): Path to save the figure (PNG).
    """
    # Load CSV
    df = pd.read_csv(csv_path)

    # Define task info and custom RGB colors
    task_info = {
        "Missing": {
            "expert0": "1_missing_Expert0",
            "expert1": "1_missing_Expert1",
            "expert1_color": (251/255, 169/255, 158/255)  # Red
        },
        "Trend": {
            "expert0": "2_trend_Expert0",
            "expert1": "2_trend_Expert1",
            "expert1_color": (206/255, 199/255, 233/255)  # Purple
        },
        "Drift": {
            "expert0": "3_drift_Expert0",
            "expert1": "3_drift_Expert1",
            "expert1_color": (196/255, 224/255, 178/255)  # Green
        }
    }

    shared_color = (151/255, 201/255, 241/255)  # Blue

    x_labels = list(task_info.keys())
    expert0_means, expert0_stds = [], []
    expert1_means, expert1_stds = [], []
    expert1_colors = []

    # Extract data
    for task, info in task_info.items():
        col0 = info["expert0"]
        col1 = info["expert1"]
        color1 = info["expert1_color"]

        if col0 not in df.columns or col1 not in df.columns:
            raise ValueError(f"Missing columns for task {task}: {col0}, {col1}")

        w0 = df[col0].values
        w1 = df[col1].values

        expert0_means.append(np.mean(w0))
        expert0_stds.append(np.std(w0))

        expert1_means.append(np.mean(w1))
        expert1_stds.append(np.std(w1))

        expert1_colors.append(color1)

    # Bar plot
    x = np.arange(len(x_labels))
    bar_width = 0.5

    plt.figure(figsize=(4, 3))

    # Shared expert bars (bottom)
    plt.bar(x, expert0_means, width=bar_width, color=shared_color,
            yerr=expert0_stds, capsize=5)

    # Task-specific expert bars (top)
    plt.bar(x, expert1_means, width=bar_width, bottom=expert0_means,
            color=expert1_colors, yerr=expert1_stds, capsize=5)

    # Annotate bars
    for i in range(len(x)):
        # Shared expert annotation
        plt.text(x[i], expert0_means[i] / 2,
                 f"{expert0_means[i]:.2f}±{expert0_stds[i]:.2f}",
                 ha='center', va='center', color='black', fontsize=8, fontweight='bold')
        # Task-specific expert annotation
        top_y = expert0_means[i] + expert1_means[i] / 2
        plt.text(x[i], top_y,
                 f"{expert1_means[i]:.2f}±{expert1_stds[i]:.2f}",
                 ha='center', va='center', color='black', fontsize=8, fontweight='bold')

    # Plot settings
    plt.xticks(x, x_labels)
    plt.ylabel("Mean Gate Weight")
    plt.title("Distribution of Expert Utilization (Stacked Mean ± Std)")
    plt.ylim(0, 1.2)
    # plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.grid(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to: {save_path}")

    plt.show()

def test(id, RESULTS_DIR):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train, y_train, x_all, y_all, all_indices = data_preprocessing(balanced=False, platform='pytorch', normal_class=False, method = 'test')

    # For all testing sample
    # np.random.seed(8)
    # indices = np.arange(x_all.shape[0])
    # np.random.shuffle(indices)
    # x_all = x_all[indices]
    # y_all = y_all[indices]
    # all_indices = all_indices[indices]

    # x_train_val = x_all[:1436,:]
    # y_train_val = y_all[:1436,:]
    # x_test = x_all[1436:,:]
    # y_test = y_all[1436:,:]
    # test_indices = all_indices[1436:]

    # x_val_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)
    # y_val_tensor = torch.tensor(y_test, dtype=torch.float32)
    # y_val_tensor = torch.eye(2)[y_val_tensor.long(), :]
    # val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    # val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)
    # num_features = x_val_tensor[:,:,1008:].shape[-1]
    

    # for single sample
    x_test = x_all[id].reshape(1, -1)
    y_test = y_all[id]
    # Save the 1D signal to a .csv file
    np.savetxt(f"{RESULTS_DIR}/signals/signal_id_{id}.csv", x_all[id][:1008], delimiter=",", header="Signal", comments='')
    plt.figure(figsize=(10, 4))
    plt.plot(x_all[id][:1008], color='royalblue')
    plt.title(f"Index: {id}")
    plt.xlabel("Time Step")
    plt.ylabel("Signal Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # for CGC with inception and CNN experts, HardShare, MoE, MMoE, CGC inception
    x_val_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)
    y_val_tensor = torch.tensor(y_test, dtype=torch.float32)
    y_val_tensor = torch.eye(2)[y_val_tensor.long(), :]
    # val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    # val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)
    num_features = x_val_tensor[:,:,1008:].shape[-1]

    model = PLE(inputs_dim=num_features,
                 labels_dict={
                    '1_missing': 2,
                    '2_trend': 2,
                    '3_drift': 2,
                 },
                 dnn_dropout=0.2,
                 num_shared_experts=1,
                 num_task_experts=1,
                 expert_hidden_units=[128],
                 tower_hidden_units=[128, 64, 32],
                 device='cuda')



    model.load_state_dict(torch.load(RESULTS_DIR+'.pth', map_location=device))
    model.to(device)
    model.eval()
    gate_weight_storage = defaultdict(list)

    dict_classes = {0: '1_missing', 1: '2_trend', 2: '3_drift'}
    y_true, y_prob, y_pred_orig = {}, {}, {}

    # For single sample
    with torch.no_grad():
        x = x_val_tensor.to(device)
        # x_batch = x_batch.to(device)
        # y_batch = y_batch.to(device)
        outputs = model(x)

    # For all testing sample
    # with torch.no_grad():
    #     for x_batch, y_batch in val_loader:
    #         x_batch = x_batch.to(device)
    #         y_batch = y_batch.to(device)
    #         outputs, gate_weights = model(x_batch)
    
    #         for task_name, gate_weight in gate_weights.items():
    #             # gate_weight: (batch_size, 2, 1) → squeeze to (batch_size, 2)
    #             gate_weight = gate_weight.squeeze(-1).cpu().numpy()  # shape: (B, 2)
    #             gate_weight_storage[task_name].append(gate_weight)
    # # === Concatenate and Create DataFrame ===

    # task_dfs = []
    # for task_name, weight_list in gate_weight_storage.items():
    #     all_weights = np.concatenate(weight_list, axis=0)  # shape: (total_samples, 2)
    #     df = pd.DataFrame(all_weights, columns=[f"{task_name}_Expert0", f"{task_name}_Expert1"])
    #     task_dfs.append(df)

    # save_path=RESULTS_DIR+"gate_weights.csv"

    # df_all = pd.concat(task_dfs, axis=1)  # (total_samples, 6)
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # df_all.to_csv(save_path, index=False)
    # print(f"Gate weights saved to {save_path}")
    # print('done')

if __name__ == '__main__':
    # Create output folder
    RESULTS_DIR = 'saved_models/3-model_analysis\PLE_Shared1_Task1_mode_2025-07-03-09-54-20/model_2025-07-03-09-58-06_fold_4'
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. 绘制论文中FIGURE 11和 FIGURE 10
    # 输入测试样本的索引ID，即可获取论文中FIGURE 11和 FIGURE 10的结果
    # 论文中FIGURE 11: Representative output features from the shared multiscale expert 
    # for multi-pattern anomalous samples. (a) Anomalous data with missing and trend patterns.（样本ID为1050） 
    # (b) Anomalous data with trend and drift patterns.（样本ID为476）
    # FIGURE 10: Visualization of statistics feature contributions across different anomaly types. 
    # (a) Missing anomaly data. （样本ID为192） 
    # (b) Trend anomaly data. 
    # (c) Drift anomaly data.
    # id = 1050
    # test(id, RESULTS_DIR)

    # 2. 绘制论文中FIGURE 12
    analyze_expert_utilization_stacked(csv_path =RESULTS_DIR+"gate_weights.csv", save_path=RESULTS_DIR+"Expert_Utilization.png")
