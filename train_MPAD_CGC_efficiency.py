'''
这是MPAD-CGC模型的训练脚本，包含多任务学习功能。
功能概述:
    该脚本实现了一个多任务学习框架，使用MPAD-CGC模型对时间序列数据进行训练。
    通过交叉验证评估模型性能，并保存每个折的训练结果和模型参数。
    训练过程中记录各任务的损失和准确率，并绘制相应的曲线图。
    最终输出每个任务在验证集上的预测结果，便于后续分析和评估。
    需要指定模型的共享专家和任务专家数量等超参数。
    新增功能：训练五次并报告计算效率指标
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
import psutil
import tracemalloc
from datetime import datetime

from pytorchModels.ple_Inception_Features_analysis_train import PLE
from preprocessing_addFeatures import data_preprocessing
from unit.summary import summary, sum_parameters_by_layer
from unit.synthetic_data import oversample_by_label_combination, oversample_by_label_combination_nonMissing
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import  time
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, SVMSMOTE, BorderlineSMOTE, KMeansSMOTE
from unit import lossfn

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from collections import defaultdict


class ComputationalMetrics:
    """计算效率指标跟踪类"""
    def __init__(self):
        self.epoch_times = []
        self.batch_times = []
        self.gpu_memory_usage = []
        self.cpu_memory_usage = []
        self.forward_pass_times = []
        self.backward_pass_times = []
        self.total_samples_processed = 0
        self.epoch_start_time = None
        self.batch_start_time = None
        self.forward_start_time = None
        self.backward_start_time = None
    
    def start_epoch(self):
        self.epoch_start_time = time.time()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    def end_epoch(self):
        if self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
            self.epoch_times.append(epoch_time)
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
                self.gpu_memory_usage.append(gpu_mem)
    
    def start_batch(self):
        self.batch_start_time = time.time()
    
    def end_batch(self, batch_size):
        if self.batch_start_time:
            batch_time = time.time() - self.batch_start_time
            self.batch_times.append(batch_time)
            self.total_samples_processed += batch_size
    
    def start_forward(self):
        self.forward_start_time = time.time()
    
    def end_forward(self):
        if self.forward_start_time:
            forward_time = time.time() - self.forward_start_time
            self.forward_pass_times.append(forward_time)
    
    def start_backward(self):
        self.backward_start_time = time.time()
    
    def end_backward(self):
        if self.backward_start_time:
            backward_time = time.time() - self.backward_start_time
            self.backward_pass_times.append(backward_time)
    
    def get_cpu_memory(self):
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 2)  # MB
    
    def record_cpu_memory(self):
        self.cpu_memory_usage.append(self.get_cpu_memory())
    
    def get_summary(self):
        summary_dict = {
            'total_epochs': len(self.epoch_times),
            'avg_epoch_time': np.mean(self.epoch_times) if self.epoch_times else 0,
            'min_epoch_time': np.min(self.epoch_times) if self.epoch_times else 0,
            'max_epoch_time': np.max(self.epoch_times) if self.epoch_times else 0,
            'total_training_time': np.sum(self.epoch_times),
            'avg_batch_time': np.mean(self.batch_times) if self.batch_times else 0,
            'throughput_samples_per_sec': self.total_samples_processed / np.sum(self.batch_times) if self.batch_times else 0,
            'total_samples_processed': self.total_samples_processed,
            'avg_gpu_memory_mb': np.mean(self.gpu_memory_usage) if self.gpu_memory_usage else 0,
            'peak_gpu_memory_mb': np.max(self.gpu_memory_usage) if self.gpu_memory_usage else 0,
            'avg_cpu_memory_mb': np.mean(self.cpu_memory_usage) if self.cpu_memory_usage else 0,
            'peak_cpu_memory_mb': np.max(self.cpu_memory_usage) if self.cpu_memory_usage else 0,
            'avg_forward_time': np.mean(self.forward_pass_times) if self.forward_pass_times else 0,
            'avg_backward_time': np.mean(self.backward_pass_times) if self.backward_pass_times else 0,
            'forward_backward_ratio': (np.mean(self.forward_pass_times) / np.mean(self.backward_pass_times)) if (self.forward_pass_times and self.backward_pass_times) else 0,
        }
        return summary_dict


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


def main(x_train, y_train, x_val, y_val, save_path, fold_id=None, run_id=None, metrics=None):
    """
    函数功能:
    1. 接收当前折的训练与验证数据，搭建 PLE 多任务模型进行训练与评估
    2. 记录每个 epoch 的各任务损失与准确率，绘制曲线并保存
    3. 跟踪计算效率指标（时间、内存、吞吐量等）
    4. 返回验证集上每个任务的真实标签、预测标签与正类概率，用于后续外部评价
    参数:
        x_train, y_train: 训练集特征与标签 (原始标签为 0/1，多任务列形式)
        x_val, y_val: 验证集特征与标签
        save_path: 当前实验的保存目录
        fold_id: 交叉验证折号（用于文件命名）
        run_id: 训练运行号（用于文件命名）
        metrics: ComputationalMetrics对象
    返回:
        y_true_fold, y_pred_fold, y_prod_fold: 验证集各任务真实标签 / 预测类别 / 预测概率
    """
    if metrics is None:
        metrics = ComputationalMetrics()
    
    import os, time
    # 确保模型保存主目录存在
    os.makedirs('saved_models', exist_ok=True)

    # 设置随机种子，保证结果可复现
    seed = 4
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    # 超参数设置
    signal_length = 1008              # 原始信号长度（用于分离附加特征）
    num_epochs = 100                  # 训练轮数
    learning_Rate = 0.001             # 学习率
    batch_size = 128                  # 批大小

    num_classes = 3                   # 任务数
    dict_classes = {0: '1_missing', 1: '2_trend', 2: '3_drift'}  # 任务索引与名称映射

    # 将 numpy 数据转为 tensor，并增加通道维度 (N,1,L)
    x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    x_val = torch.tensor(x_val, dtype=torch.float32).unsqueeze(1)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    # 将每列任务标签 (0/1) 转为 one-hot，两类任务 -> shape: (N, num_tasks, 2)
    y_train = torch.eye(2)[y_train.long(), :]
    y_val = torch.eye(2)[y_val.long(), :]

    # 计算额外特征数量 (切分信号后面的附加维度)
    num_features = x_train[:,:,signal_length:].shape[-1]

    # 构建 MPAD-CGC 模型（多任务共享 + 私有专家结构）
    model = PLE(inputs_dim=num_features,
                labels_dict={
                    '1_missing': 2, # 二分类任务
                    '2_trend': 2, # 二分类任务
                    '3_drift': 2, # 二分类任务
                },
                dnn_dropout=0.2, # Dropout 比例
                num_shared_experts=1, # shared_experts数量
                num_task_experts=1, # task_experts数量
                expert_hidden_units=[128], # shared_experts最后一层隐藏层的单元数
                tower_hidden_units=[128, 64, 32], # Task Tower's FCNN 隐藏层单元数列表
                device='cuda')

    # 设备选择与迁移
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    sum_parameters_by_layer(model)  # 打印各层参数统计（自定义函数）
    print(f"Using device: {device}")

    # 定义任务损失函数（批层面的 Focal Loss）
    criterion = BLFocalLoss(reduction='mean')

    # 不确定性学习的可训练 log_vars（每个任务一个）
    log_vars = nn.Parameter(torch.zeros(num_classes, requires_grad=True, device=device))
    optimizer = optim.Adam(model.parameters(), lr=learning_Rate)          # 主优化器（模型参数）
    optimizer_uncertainty = optim.Adam([log_vars], lr=1e-3)       # 不确定性参数优化器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)  # 学习率调度

    # 构建数据集与数据加载器
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_val, y_val)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 记录训练过程指标（按任务与总损失）
    train_loss_values = []
    test_loss_values = []
    train_accuracy_values = []
    test_accuracy_values = []
    train_loss_total = []
    test_loss_total = []
    epoch_times = []
    epoch_gpu_memory = []

    # 训练主循环
    for epoch in range(num_epochs):
        metrics.start_epoch()
        model.train()
        # 每个 epoch 初始化记录容器
        epoch_train_loss_list = torch.zeros(num_classes).to(device)
        epoch_train_correct_list = torch.zeros(num_classes).to(device)
        epoch_train_loss_total = 0
        total_train = 0

        # ----------- 前向与反向（训练阶段）-----------
        for inputs, labels in train_data_loader:
            metrics.start_batch()
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            optimizer_uncertainty.zero_grad()

            # 前向传播计时
            metrics.start_forward()
            outputs = model(inputs)                  # 多任务输出字典
            metrics.end_forward()

            # 计算每个任务损失（依次取对应输出与标签列）
            loss_list = torch.stack([criterion(outputs[dict_classes[key]], labels[:, key], log_vars, key) 
                                     for key in dict_classes])
           
            # 当前未对各任务加权（直接求和+L2正则）
            weighted_losses = loss_list
            train_loss = weighted_losses.sum() + model.l2_reg_loss

            # 反向传播计时
            metrics.start_backward()
            train_loss.backward()
            metrics.end_backward()
            
            optimizer.step()
            optimizer_uncertainty.step()

            # 累积任务损失
            epoch_train_loss_list += loss_list
            epoch_train_loss_total += train_loss.item()

            # 计算每任务预测正确数（取最大值类别）
            predicted_train_list = [torch.max(outputs[dict_classes[key]].data, 1) for key in dict_classes]
            correct_train_list = [torch.sum(predicted_train_list[key][1] == torch.max(labels[:, key], 1)[1]) 
                                  for key in dict_classes]
            epoch_train_correct_list += torch.tensor(correct_train_list).to(device)
            total_train += labels.size(0)
            
            metrics.end_batch(batch_size)
            metrics.record_cpu_memory()

        # 调度学习率
        scheduler.step()

        # 计算平均损失与准确率（按任务）
        epoch_train_loss_list /= len(train_data_loader)
        epoch_train_loss_total /= len(train_data_loader)
        epoch_train_accuracy_list = torch.stack([100 * epoch_train_correct_list[key] / total_train 
                                                 for key in dict_classes])

        # ----------- 验证阶段（评估不反向）-----------
        model.eval()
        epoch_test_loss_list = torch.zeros(num_classes).to(device)
        epoch_test_correct_list = torch.zeros(num_classes).to(device)
        epoch_test_loss_total = 0
        total_test = 0

        # 保存当前折验证集预测情况（后续主脚本使用）
        y_true_fold = {task: [] for task in dict_classes.values()}
        y_pred_fold = {task: [] for task in dict_classes.values()}
        y_prod_fold = {task: [] for task in dict_classes.values()}

        with torch.no_grad():
            for inputs, labels in test_data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                # 验证集任务损失
                loss_list = torch.stack([criterion(outputs[dict_classes[key]], labels[:, key], log_vars, key) 
                                         for key in dict_classes])
                weighted_losses = loss_list
                test_loss = weighted_losses.sum() + model.l2_reg_loss
                epoch_test_loss_list += loss_list
                epoch_test_loss_total += test_loss.item()

                # 收集各任务预测概率/类别/真实标签（概率取第二类 softmax 后值）
                for key in dict_classes:
                    prods = torch.softmax(outputs[dict_classes[key]], dim=1)[:, 1].cpu().numpy()
                    preds = torch.argmax(outputs[dict_classes[key]], dim=1).cpu().numpy()
                    true = torch.argmax(labels[:, key], dim=1).cpu().numpy()
                    y_prod_fold[dict_classes[key]].extend(prods)
                    y_pred_fold[dict_classes[key]].extend(preds)
                    y_true_fold[dict_classes[key]].extend(true)

                # 统计分类正确数
                predicted_test_list = [torch.max(outputs[dict_classes[key]].data, 1) for key in dict_classes]
                correct_test_list = [torch.sum(predicted_test_list[key][1] == torch.max(labels[:, key], 1)[1]) 
                                     for key in dict_classes]
                epoch_test_correct_list += torch.tensor(correct_test_list).to(device)
                total_test += labels.size(0)

        # 验证集平均损失与准确率
        epoch_test_loss_list /= len(test_data_loader)
        epoch_test_loss_total /= len(test_data_loader)
        epoch_test_accuracy_list = torch.stack([100 * epoch_test_correct_list[key] / total_test 
                                                 for key in dict_classes])

        # 记录历史（用于绘图与 CSV）
        train_loss_values.append(epoch_train_loss_list.cpu().detach().numpy())
        test_loss_values.append(epoch_test_loss_list.cpu().detach().numpy())
        train_accuracy_values.append(epoch_train_accuracy_list.cpu().detach().numpy())
        test_accuracy_values.append(epoch_test_accuracy_list.cpu().detach().numpy())
        train_loss_total.append(epoch_train_loss_total)
        test_loss_total.append(epoch_test_loss_total)
        
        metrics.end_epoch()
        epoch_times.append(metrics.epoch_times[-1] if metrics.epoch_times else 0)
        if metrics.gpu_memory_usage:
            epoch_gpu_memory.append(metrics.gpu_memory_usage[-1])

        # 每个 epoch 打印汇总信息
        if (epoch + 1) % 10 == 0:
            print(f'\nEpoch [{epoch + 1}/{num_epochs}]')
            print(f"Epoch [{epoch+1}/{num_epochs}], train Loss: {epoch_train_loss_total:.4f}, Log Vars: {log_vars.data.cpu().numpy()}")
            for key in dict_classes:
                print(f'Train_{dict_classes[key]}_Loss: {epoch_train_loss_list[key]:.4f}, Train_{dict_classes[key]}_Accuracy: {epoch_train_accuracy_list[key]:.2f}%')
            print('-------------------------------------------------------')
            print(f"Epoch [{epoch+1}/{num_epochs}], test Loss: {epoch_test_loss_total:.4f}")
            for key in dict_classes:
                print(f'Test_{dict_classes[key]}_Loss: {epoch_test_loss_list[key]:.4f}, Test_{dict_classes[key]}_Accuracy: {epoch_test_accuracy_list[key]:.2f}%')
            print(f"Epoch Time: {epoch_times[-1]:.2f}s")

    # 保存当前折模型权重
    run_str = f"_run{run_id}" if run_id is not None else ""
    fold_str = f"_fold{fold_id}" if fold_id is not None else ""
    model_path = '{}/model_{}{}{}.pth'.format(save_path, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()), run_str, fold_str)
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

    # 绘制训练与验证曲线（按任务与总损失）
    plot_training_curves(train_loss_values, test_loss_values, train_accuracy_values, test_accuracy_values, dict_classes, save_path)
    plot_combined_task_curves(train_loss_values, test_loss_values, train_accuracy_values, test_accuracy_values, dict_classes, save_path)
    plot_total_loss_curve(train_loss_total, test_loss_total, save_path, fold_id)

    # 将记录列表转为 ndarray 便于构建 DataFrame
    train_loss_array = np.stack(train_loss_values)      # shape: (num_epochs, num_tasks)
    test_loss_array = np.stack(test_loss_values)        # shape: (num_epochs, num_tasks)
    train_acc_array = np.stack(train_accuracy_values)
    test_acc_array = np.stack(test_accuracy_values)

    # 构造损失指标 DataFrame
    loss_df = pd.DataFrame({
        'Epoch': np.arange(1, num_epochs + 1),
        'Epoch_Time_Sec': epoch_times,
        'Train_total_loss': train_loss_total,
        'Train_Missing_Loss': train_loss_array[:, 0],
        'Train_Trend_Loss': train_loss_array[:, 1],
        'Train_Drift_Loss': train_loss_array[:, 2],
        'Test_total_loss': test_loss_total,
        'Test_Missing_Loss': test_loss_array[:, 0],
        'Test_Trend_Loss': test_loss_array[:, 1],
        'Test_Drift_Loss': test_loss_array[:, 2],
    })

    # 构造准确率指标 DataFrame
    acc_df = pd.DataFrame({
        'Epoch': np.arange(1, num_epochs + 1),
        'Train_Missing_Acc': train_acc_array[:, 0],
        'Train_Trend_Acc': train_acc_array[:, 1],
        'Train_Drift_Acc': train_acc_array[:, 2],
        'Test_Missing_Acc': test_acc_array[:, 0],
        'Test_Trend_Acc': test_acc_array[:, 1],
        'Test_Drift_Acc': test_acc_array[:, 2],
    })

    # 保存 CSV 文件（带折标识）
    csv_path = os.path.join(save_path, f"loss_values{run_str}{fold_str}.csv")
    loss_df.to_csv(csv_path, index=False)
    print(f"Loss values saved to {csv_path}")

    acc_csv_path = os.path.join(save_path, f"accuracy_values{run_str}{fold_str}.csv")
    acc_df.to_csv(acc_csv_path, index=False)
    print(f"Accuracy values saved to {acc_csv_path}")
    
    # 返回验证集各任务真实标签 / 预测标签 / 预测概率和计算指标
    return y_true_fold, y_pred_fold, y_prod_fold, metrics


if __name__ == '__main__':
    # 生成模型保存目录（使用当前时间戳区分不同实验）
    save_path = 'saved_models/PLE_mode_{}'.format(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    os.makedirs(save_path, exist_ok=True)
    print(f"\nExperiment directory: {save_path}")

    # 数据预处理：返回 x,y 等，其中此处只取全部样本与标签
    _, _, x_all, y_all, _ = data_preprocessing(balanced=False, platform='pytorch', normal_class=False, method='test')
    
    # 数据打乱（确保随机性）
    np.random.seed(8)
    indices = np.arange(x_all.shape[0])
    np.random.shuffle(indices)
    x_all = x_all[indices]
    y_all = y_all[indices]

    # 划分训练验证整体集合与测试集合
    x_train_val = x_all[:1436,:] # train+val共占比80%， 1436/1794=0.8
    y_train_val = y_all[:1436,:] # 80%
    x_test = x_all[1436:,:] # test占比20%
    y_test = y_all[1436:,:] # 20%

    # 存储所有运行和折的预测与真实标签（用于最终汇总）
    all_runs_metrics = []  # 存储所有运行的计算指标
    all_y_true = defaultdict(list)
    all_y_pred = defaultdict(list)
    all_y_prod = defaultdict(list)
    report_logs = []  # 保存各折分类报告的文本

    num_runs = 5  # 训练五次
    
    # 五次运行循环
    for run_id in range(1, num_runs + 1):
        print(f"\n{'='*60}")
        print(f"Training Run {run_id}/{num_runs}")
        print(f"{'='*60}")
        
        run_start_time = time.time()
        run_metrics = ComputationalMetrics()

        # 使用第一个任务的标签作为分层依据（保证折划分类别均衡）
        stratify_labels = y_train_val[:, 0]
        skf = StratifiedKFold(n_splits=4, shuffle=False)

        # 交叉验证循环（这里只训练第4折）
        for fold, (train_idx, val_idx) in enumerate(skf.split(x_train_val, stratify_labels), 1):
            if fold == 4:
                print(f"\n--- Fold {fold} ---")
                # 当前折的训练与验证数据
                x_train_fold = x_train_val[train_idx]
                y_train_fold = y_train_val[train_idx]
                x_val_fold = x_train_val[val_idx]
                y_val_fold = y_train_val[val_idx]

                # 训练当前折模型并返回该折的预测结果与计算指标
                y_true_dict, y_pred_dict, y_prod_dict, fold_metrics = main(
                    x_train_fold, y_train_fold, x_val_fold, y_val_fold, 
                    save_path, fold_id=fold, run_id=run_id, metrics=run_metrics
                )

                # 记录本折的分类报告与 AUC
                report_logs.append(f"\n=== Run {run_id}, Fold {fold} ===")
                for task in ['1_missing', '2_trend', '3_drift']:
                    report = classification_report(
                        y_true_dict[task], y_pred_dict[task], target_names=['0', '1'], digits=4
                    )
                    auc = roc_auc_score(y_true_dict[task], y_prod_dict[task])
                    report_logs.append(f"\nTask: {task}\n{report}\nROC AUC: {auc:.4f}\n")
                    print(f"\nTask: {task}\nROC AUC: {auc:.4f}\n")

                    # 汇总到全局（用于最终平均报告）
                    all_y_true[task].extend(y_true_dict[task])
                    all_y_pred[task].extend(y_pred_dict[task])
                    all_y_prod[task].extend(y_prod_dict[task])
            else:
                continue
        
        run_elapsed = time.time() - run_start_time
        run_metrics_summary = run_metrics.get_summary()
        run_metrics_summary['run_id'] = run_id
        run_metrics_summary['total_run_time'] = run_elapsed
        all_runs_metrics.append(run_metrics_summary)
        
        print(f"\nRun {run_id} completed in {run_elapsed:.2f}s")

    # 计算汇总平均分类报告与 AUC
    report_logs.append("\n" + "="*60)
    report_logs.append("=== Final Average Classification Report (All Runs) ===")
    report_logs.append("="*60)
    
    for task in ['1_missing', '2_trend', '3_drift']:
        report = classification_report(all_y_true[task], all_y_pred[task], target_names=['0', '1'], digits=4)
        auc = roc_auc_score(all_y_true[task], all_y_prod[task])
        report_logs.append(f"\nAverage Classification Report for Task: {task}\n{report}\nROC AUC: {auc:.4f}\n")
        print(f"\nAverage Classification Report for Task: {task}\nROC AUC: {auc:.4f}\n")

    # 生成计算效率报告
    print("\n" + "="*60)
    print("COMPUTATIONAL EFFICIENCY REPORT")
    print("="*60)
    
    efficiency_report = []
    efficiency_report.append("\n" + "="*60)
    efficiency_report.append("COMPUTATIONAL EFFICIENCY ANALYSIS")
    efficiency_report.append("="*60)
    efficiency_report.append(f"\nReport Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    efficiency_report.append(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    if torch.cuda.is_available():
        efficiency_report.append(f"GPU: {torch.cuda.get_device_name(0)}")
    
    efficiency_report.append(f"\nTotal Training Runs: {num_runs}")
    efficiency_report.append(f"Total Folds per Run: 1 (only fold 4)")
    efficiency_report.append(f"Total Epochs per Run: 100")
    
    # 聚合所有运行的指标
    all_epoch_times = []
    all_gpu_memory = []
    all_cpu_memory = []
    all_throughputs = []
    all_run_times = []
    
    efficiency_report.append("\n" + "-"*60)
    efficiency_report.append("PER-RUN METRICS")
    efficiency_report.append("-"*60)
    
    for run_summary in all_runs_metrics:
        efficiency_report.append(f"\nRun {int(run_summary['run_id'])}:")
        efficiency_report.append(f"  Total Run Time: {run_summary['total_run_time']:.2f}s")
        efficiency_report.append(f"  Average Epoch Time: {run_summary['avg_epoch_time']:.4f}s")
        efficiency_report.append(f"  Min Epoch Time: {run_summary['min_epoch_time']:.4f}s")
        efficiency_report.append(f"  Max Epoch Time: {run_summary['max_epoch_time']:.4f}s")
        efficiency_report.append(f"  Total Training Time: {run_summary['total_training_time']:.2f}s")
        efficiency_report.append(f"  Throughput: {run_summary['throughput_samples_per_sec']:.2f} samples/sec")
        efficiency_report.append(f"  Total Samples Processed: {int(run_summary['total_samples_processed'])}")
        efficiency_report.append(f"  Average GPU Memory: {run_summary['avg_gpu_memory_mb']:.2f} MB")
        efficiency_report.append(f"  Peak GPU Memory: {run_summary['peak_gpu_memory_mb']:.2f} MB")
        efficiency_report.append(f"  Average CPU Memory: {run_summary['avg_cpu_memory_mb']:.2f} MB")
        efficiency_report.append(f"  Peak CPU Memory: {run_summary['peak_cpu_memory_mb']:.2f} MB")
        efficiency_report.append(f"  Average Forward Pass Time: {run_summary['avg_forward_time']:.6f}s")
        efficiency_report.append(f"  Average Backward Pass Time: {run_summary['avg_backward_time']:.6f}s")
        efficiency_report.append(f"  Forward/Backward Ratio: {run_summary['forward_backward_ratio']:.2f}")
        
        all_epoch_times.append(run_summary['avg_epoch_time'])
        all_gpu_memory.append(run_summary['avg_gpu_memory_mb'])
        all_cpu_memory.append(run_summary['avg_cpu_memory_mb'])
        all_throughputs.append(run_summary['throughput_samples_per_sec'])
        all_run_times.append(run_summary['total_run_time'])
    
    # 计算跨运行的平均值和标准差
    efficiency_report.append("\n" + "-"*60)
    efficiency_report.append("AGGREGATE METRICS (All Runs)")
    efficiency_report.append("-"*60)
    
    if all_epoch_times:
        efficiency_report.append(f"\nAverage Epoch Time: {np.mean(all_epoch_times):.4f}s (±{np.std(all_epoch_times):.4f}s)")
    if all_gpu_memory:
        efficiency_report.append(f"Average GPU Memory Usage: {np.mean(all_gpu_memory):.2f}MB (±{np.std(all_gpu_memory):.2f}MB)")
    if all_cpu_memory:
        efficiency_report.append(f"Average CPU Memory Usage: {np.mean(all_cpu_memory):.2f}MB (±{np.std(all_cpu_memory):.2f}MB)")
    if all_throughputs:
        efficiency_report.append(f"Average Throughput: {np.mean(all_throughputs):.2f} samples/sec (±{np.std(all_throughputs):.2f})")
    if all_run_times:
        total_training_time = np.sum(all_run_times)
        efficiency_report.append(f"Average Run Time: {np.mean(all_run_times):.2f}s (±{np.std(all_run_times):.2f}s)")
        efficiency_report.append(f"Total Training Time (All Runs): {total_training_time:.2f}s ({total_training_time/3600:.2f}h)")
    
    # 打印报告到控制台和文件
    report_text = "\n".join(efficiency_report)
    print(report_text)
    
    # 保存所有报告到文本文件
    with open(os.path.join(save_path, "classification_reports.txt"), "w") as f:
        f.write("\n".join(report_logs))
    
    with open(os.path.join(save_path, "efficiency_report.txt"), "w") as f:
        f.write(report_text)
    
    print(f"\nClassification reports saved to {os.path.join(save_path, 'classification_reports.txt')}")
    print(f"Efficiency report saved to {os.path.join(save_path, 'efficiency_report.txt')}")
    
    # 保存效率指标为CSV
    efficiency_df = pd.DataFrame(all_runs_metrics)
    efficiency_csv_path = os.path.join(save_path, "computational_efficiency_metrics.csv")
    efficiency_df.to_csv(efficiency_csv_path, index=False)
    print(f"Computational metrics saved to {efficiency_csv_path}")