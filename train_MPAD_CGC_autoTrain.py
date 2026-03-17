'''
这是MPAD-CGC模型的自动训练主脚本，包含自动化的超参数搜索功能，
通过交叉验证评估不同的专家配置组合（共享专家和任务专家数量），
并保存每个配置的训练结果和模型参数，便于后续使用para_config.py脚本进行分析和选择最佳配置。
'''

import random
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from unit.plot_curves import plot_training_curves, plot_combined_task_curves, plot_total_loss_curve
import gc
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import torch.nn.functional as F

from pytorchModels.ple_Inception_Features_analysis_train import PLE
from preprocessing_addFeatures import data_preprocessing
from unit.summary import summary, sum_parameters_by_layer
from unit.synthetic_data import oversample_by_label_combination, oversample_by_label_combination_nonMissing
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import  time
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, SVMSMOTE, BorderlineSMOTE, KMeansSMOTE
from unit import lossfn

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

class BLFocalLoss(nn.Module):
    '''Batch-level Focal Loss'''
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(BLFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, input, target, sigma_sq, key):
        ce_loss = nn.BCELoss()(input, target)
        pt = torch.exp(-ce_loss)  # 计算概率
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss  # Focal Loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from collections import defaultdict


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from collections import defaultdict


def main(x_train, y_train, x_val, y_val, save_path, fold_id=None, num_shared_experts=1, num_task_experts=1):
    import os, time
    # Create directory for saving model
    os.makedirs('saved_models', exist_ok=True)

    # Set random seed for reproducibility
    seed = 4
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    # Model parameters
    signal_length = 1008
    features_length = 9
    cnn_channels = 16
    lstm_hidden_size = 32
    lstm_num_layers = 1
    output_size = 6
    num_epochs = 100
    learning_Rate = 0.001
    batch_size = 128

    num_classes = 3
    dict_classes = {0: '1_missing', 1: '2_trend', 2: '3_drift'}

    # x_train, y_train = oversample_multitask(x_train, y_train, random_state=SEED)
    # x_train, y_train, _, _ = oversample_by_label_combination_nonMissing(
    #     x_train[:, :1008], y_train, method='SMOTE', random_state=42, k_neighbors=2)

    x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    x_val = torch.tensor(x_val, dtype=torch.float32).unsqueeze(1)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    y_train = torch.eye(2)[y_train.long(), :]
    y_val = torch.eye(2)[y_val.long(), :]

    num_features = x_train[:,:,1008:].shape[-1]

    model = PLE(
        inputs_dim=num_features,
        labels_dict={
            '1_missing': 2,
            '2_trend': 2,
            '3_drift': 2,
        },
        dnn_dropout=0.2,
        num_shared_experts=num_shared_experts,
        num_task_experts=num_task_experts,
        expert_hidden_units=[128],
        tower_hidden_units=[128, 64, 32],
        device='cuda'
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    sum_parameters_by_layer(model)
    print(f"Using device: {device}")

    criterion = BLFocalLoss(reduction='mean')
    

    log_vars = nn.Parameter(torch.zeros(num_classes, requires_grad=True, device=device))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer_uncertainty = optim.Adam([log_vars], lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_val, y_val)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train_loss_values = []
    test_loss_values = []
    train_accuracy_values = []
    test_accuracy_values = []
    train_loss_total = []
    test_loss_total = []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss_list = torch.zeros(num_classes).to(device)
        epoch_train_correct_list = torch.zeros(num_classes).to(device)
        epoch_train_loss_total = 0
        total_train = 0

        for inputs, labels in train_data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            optimizer_uncertainty.zero_grad()

            sigma_sq = torch.exp(log_vars.detach())
            outputs = model(inputs)
            loss_list = torch.stack([criterion(outputs[dict_classes[key]], labels[:, key], log_vars, key) for key in dict_classes])
            # weighted_losses = torch.exp(-log_vars) * loss_list
            weighted_losses = loss_list
            train_loss = weighted_losses.sum() + model.l2_reg_loss

            train_loss.backward()
            optimizer.step()
            optimizer_uncertainty.step()

            epoch_train_loss_list += loss_list
            epoch_train_loss_total += train_loss.item()
            predicted_train_list = [torch.max(outputs[dict_classes[key]].data, 1) for key in dict_classes]
            correct_train_list = [torch.sum(predicted_train_list[key][1] == torch.max(labels[:, key], 1)[1]) for key in dict_classes]
            epoch_train_correct_list += torch.tensor(correct_train_list).to(device)
            total_train += labels.size(0)

        scheduler.step()

        epoch_train_loss_list /= len(train_data_loader)
        epoch_train_loss_total /= len(train_data_loader)
        epoch_train_accuracy_list = torch.stack([100 * epoch_train_correct_list[key] / total_train for key in dict_classes])

        model.eval()
        epoch_test_loss_list = torch.zeros(num_classes).to(device)
        epoch_test_correct_list = torch.zeros(num_classes).to(device)
        epoch_test_loss_total = 0
        total_test = 0

        y_true_fold = {task: [] for task in dict_classes.values()}
        y_pred_fold = {task: [] for task in dict_classes.values()}
        y_prod_fold = {task: [] for task in dict_classes.values()}

        with torch.no_grad():
            for inputs, labels in test_data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss_list = torch.stack([criterion(outputs[dict_classes[key]], labels[:, key], log_vars, key) for key in dict_classes])
                weighted_losses = loss_list
                test_loss = weighted_losses.sum() + model.l2_reg_loss
                epoch_test_loss_list += loss_list
                epoch_test_loss_total += test_loss.item()

                for key in dict_classes:
                    prods = torch.softmax(outputs[dict_classes[key]], dim=1)[:, 1].cpu().numpy()
                    preds = torch.argmax(outputs[dict_classes[key]], dim=1).cpu().numpy()
                    true = torch.argmax(labels[:, key], dim=1).cpu().numpy()
                    y_prod_fold[dict_classes[key]].extend(prods)
                    y_pred_fold[dict_classes[key]].extend(preds)
                    y_true_fold[dict_classes[key]].extend(true)

                predicted_test_list = [torch.max(outputs[dict_classes[key]].data, 1) for key in dict_classes]
                correct_test_list = [torch.sum(predicted_test_list[key][1] == torch.max(labels[:, key], 1)[1]) for key in dict_classes]
                epoch_test_correct_list += torch.tensor(correct_test_list).to(device)
                total_test += labels.size(0)

        epoch_test_loss_list /= len(test_data_loader)
        epoch_test_loss_total /= len(test_data_loader)
        epoch_test_accuracy_list = torch.stack([100 * epoch_test_correct_list[key] / total_test for key in dict_classes])

        train_loss_values.append(epoch_train_loss_list.cpu().detach().numpy())
        test_loss_values.append(epoch_test_loss_list.cpu().detach().numpy())
        train_accuracy_values.append(epoch_train_accuracy_list.cpu().detach().numpy())
        test_accuracy_values.append(epoch_test_accuracy_list.cpu().detach().numpy())
        train_loss_total.append(epoch_train_loss_total)
        test_loss_total.append(epoch_test_loss_total)

        if (epoch + 1) % 1 == 0:
            print(f'\nEpoch [{epoch + 1}/{num_epochs}]')
            print(f"Epoch [{epoch+1}/{num_epochs}], train Loss: {epoch_train_loss_total:.4f}, Log Vars: {log_vars.data.cpu().numpy()}")
            for key in dict_classes:
                print(f'Train_{dict_classes[key]}_Loss: {epoch_train_loss_list[key]:.4f}, Train_{dict_classes[key]}_Accuracy: {epoch_train_accuracy_list[key]:.2f}%')
            print('-------------------------------------------------------')
            print(f"Epoch [{epoch+1}/{num_epochs}], test Loss: {epoch_test_loss_total:.4f}, Log Vars: {log_vars.data.cpu().numpy()}")
            for key in dict_classes:
                print(f'Test_{dict_classes[key]}_Loss: {epoch_test_loss_list[key]:.4f}, Test_{dict_classes[key]}_Accuracy: {epoch_test_accuracy_list[key]:.2f}%')

    model_path = '{}/model_{}_fold_{}.pth'.format(save_path, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()),fold_id)
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

    plot_training_curves(train_loss_values, test_loss_values, train_accuracy_values, test_accuracy_values, dict_classes, save_path)
    plot_combined_task_curves(train_loss_values, test_loss_values, train_accuracy_values, test_accuracy_values, dict_classes, save_path)
    plot_total_loss_curve(train_loss_total, test_loss_total, save_path, fold_id)

    train_loss_array = np.stack(train_loss_values)  # shape: (num_epochs, num_tasks)
    test_loss_array = np.stack(test_loss_values)    # shape: (num_epochs, num_tasks)
    train_acc_array = np.stack(train_accuracy_values)
    test_acc_array = np.stack(test_accuracy_values)


    loss_df = pd.DataFrame({
        'Epoch': np.arange(1, num_epochs + 1),
        'Train_total_loss': train_loss_total,
        'Train_Missing_Loss': train_loss_array[:, 0],
        'Train_Trend_Loss': train_loss_array[:, 1],
        'Train_Drift_Loss': train_loss_array[:, 2],
        'Test_total_loss': test_loss_total,
        'Test_Missing_Loss': test_loss_array[:, 0],
        'Test_Trend_Loss': test_loss_array[:, 1],
        'Test_Drift_Loss': test_loss_array[:, 2],
    })

    acc_df = pd.DataFrame({
        'Epoch': np.arange(1, num_epochs + 1),
        'Train_Missing_Acc': train_acc_array[:, 0],
        'Train_Trend_Acc': train_acc_array[:, 1],
        'Train_Drift_Acc': train_acc_array[:, 2],
        'Test_Missing_Acc': test_acc_array[:, 0],
        'Test_Trend_Acc': test_acc_array[:, 1],
        'Test_Drift_Acc': test_acc_array[:, 2],
    })

    # Save to CSV (include fold_id if provided)
    fold_str = f"_fold{fold_id}" if fold_id is not None else ""
    csv_path = os.path.join(save_path, f"loss_values{fold_str}.csv")
    loss_df.to_csv(csv_path, index=False)
    print(f"Loss values saved to {csv_path}")

    acc_csv_path = os.path.join(save_path, f"accuracy_values{fold_str}.csv")
    acc_df.to_csv(acc_csv_path, index=False)
    print(f"Accuracy values saved to {acc_csv_path}")
    
    return y_true_fold, y_pred_fold, y_prod_fold



if __name__ == '__main__':
    _, _, x_all, y_all, _ = data_preprocessing(balanced=False, platform='pytorch', normal_class=False, method='test')

    # np.random.seed(8)
    # indices = np.arange(x_all.shape[0])
    # np.random.shuffle(indices)
    # x_all = x_all[indices]
    # y_all = y_all[indices]
    # Shuffle the data
    np.random.seed(8)
    indices = np.arange(x_all.shape[0])
    np.random.shuffle(indices)
    x_all = x_all[indices]
    y_all = y_all[indices]

    x_train_val = x_all[:1436,:]
    y_train_val = y_all[:1436,:]
    x_test = x_all[1436:,:]
    y_test = y_all[1436:,:]

    stratify_labels = y_train_val[:, 0]
    skf = StratifiedKFold(n_splits=4, shuffle=False)

    # Grid search for num_shared_experts and num_task_experts
    for num_shared_experts in range(1, 11): # 设置共享专家数量范围
        for num_task_experts in range(1, 11): # 设置任务专家数量范围
            config_name = f"Shared{num_shared_experts}_Task{num_task_experts}"
            timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            save_path = f'saved_models/para_configuration/PLE_{config_name}_{timestamp}'
            os.makedirs(save_path, exist_ok=True)
            print(f"\n--- Training with num_shared_experts={num_shared_experts}, num_task_experts={num_task_experts} ---")
            print(f"Model saved to {save_path}")

            all_y_true = defaultdict(list)
            all_y_pred = defaultdict(list)
            all_y_prod = defaultdict(list)
            report_logs = []

            for fold, (train_idx, val_idx) in enumerate(skf.split(x_train_val, stratify_labels), 1):
                if fold == 4:
                    print(f"\n=== Fold {fold} ===")
                    x_train_fold = x_train_val[train_idx]
                    y_train_fold = y_train_val[train_idx]
                    x_val_fold = x_train_val[val_idx]
                    y_val_fold = y_train_val[val_idx]

                    # Train and evaluate
                    y_true_dict, y_pred_dict, y_prod_dict = main(
                        x_train_fold, y_train_fold,
                        x_val_fold, y_val_fold,
                        save_path, fold_id=fold,
                        num_shared_experts=num_shared_experts,
                        num_task_experts=num_task_experts
                    )

                    # Optionally save or aggregate results here

                    report_logs.append(f"\n=== Classification Report: Fold {fold} ===")
                    for task in ['1_missing', '2_trend', '3_drift']:
                        report = classification_report(
                            y_true_dict[task], y_pred_dict[task], target_names=['0', '1'], digits=4
                        )
                        auc = roc_auc_score(y_true_dict[task], y_prod_dict[task])
                        report_logs.append(f"\nTask: {task}\n{report}\nROC AUC: {auc:.4f}\n")
                        print(f"\nTask: {task}\n{report}\nROC AUC: {auc:.4f}\n")
                        # report_logs.append(f"\nTask: {task}\n{report}")

                        all_y_true[task].extend(y_true_dict[task])
                        all_y_pred[task].extend(y_pred_dict[task])
                        all_y_prod[task].extend(y_prod_dict[task])
                else:
                    continue
            # Final average report
            report_logs.append("\n=== Final Average Classification Report (All Folds) ===")
            for task in ['1_missing', '2_trend', '3_drift']:
                report = classification_report(all_y_true[task], all_y_pred[task], target_names=['0', '1'], digits=4)
                auc = roc_auc_score(all_y_true[task], all_y_prod[task])
                report_logs.append(f"\nAverage Classification Report for Task: {task}\n{report}\nROC AUC: {auc}\n")
                print(f"\nAverage Classification Report for Task: {task}\n{report}\nROC AUC: {auc:.4f}\n")
                # report_logs.append(f"\nAverage Classification Report for Task: {task}\n{report}")
                # print(f"\nAverage Classification Report for Task: {task}")
                # print(report)

            # Save all reports to a text file
            with open(save_path+"/classification_reports.txt", "w") as f:
                f.write("\n".join(report_logs))
