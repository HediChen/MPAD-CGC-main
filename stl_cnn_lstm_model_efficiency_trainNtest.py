"""
Single-Task Learning CNN-LSTM Model for Multi-Class Anomaly Detection
Uses pre-split datasets: x_train_fold, y_train_fold, x_val_fold, y_val_fold, x_test, y_test

Addresses peer-review comments on class imbalance and baseline comparison
Added computational efficiency testing metrics for both training and inference.

References:
- Zhou et al., 2016: "A Deep Learning Algorithm with a Physical Constraint for Magnetic Resonance Image Reconstruction"
- Sainath et al., 2015: "Convolutional, Recurrent, and Fully Connected Deep Neural Networks for Speech Recognition"
- Lin et al., 2017: "Focal Loss for Dense Object Detection"
- Wang et al., 2019: "Calibrating Deep Neural Networks using Focal Loss"
- Cui et al., 2021: "Class-Balanced Loss Based on Effective Number of Samples"
- Li et al., 2021: "Balanced Softmax with Margin Loss for Long-Tailed Visual Recognition"
- Tan et al., 2020: "Equalization Loss for Long-Tailed Object Detection"
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    auc, f1_score, precision_score, recall_score, accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
import time
import gc
import psutil
from datetime import datetime
from preprocessing_addFeatures import data_preprocessing
from collections import defaultdict
from unit.summary import summary, sum_parameters_by_layer

import pandas as pd

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# ==================== LOSS FUNCTION OPTIONS ====================

LOSS_FUNCTION = 'class_balanced'  # Change this to select loss function
# Options: 'focal_loss', 'class_balanced', 'balanced_softmax', 'ldam', 'weighted_ce'

class ComputationalMetrics:
    """计算效率指标跟踪类 (Training)"""
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

class InferenceMetrics:
    """推理效率指标跟踪类 (Inference)"""
    def __init__(self):
        self.batch_times = []
        self.inference_times = []
        self.dataload_times = []
        self.gpu_memory_usage = []
        self.cpu_memory_usage = []
        self.total_samples_processed = 0
        self.peak_gpu_memory = 0
        self.peak_cpu_memory = 0
        self.batch_start_time = None
        self.inference_start_time = None
        self.dataload_start_time = None
        self.run_start_time = None
        self.model_size_mb = 0
    
    def start_run(self):
        self.run_start_time = time.time()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    def end_run(self):
        if self.run_start_time:
            return time.time() - self.run_start_time
        return 0
    
    def start_dataload(self):
        self.dataload_start_time = time.time()
    
    def end_dataload(self):
        if self.dataload_start_time:
            self.dataload_times.append(time.time() - self.dataload_start_time)
    
    def start_inference(self):
        self.inference_start_time = time.time()
    
    def end_inference(self, batch_size):
        if self.inference_start_time:
            inference_time = time.time() - self.inference_start_time
            self.inference_times.append(inference_time)
            self.total_samples_processed += batch_size
            self.batch_times.append(inference_time)
            self.record_memory()
    
    def record_memory(self):
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / (1024 ** 2)
            self.gpu_memory_usage.append(gpu_mem)
            peak_gpu = torch.cuda.max_memory_allocated() / (1024 ** 2)
            if peak_gpu > self.peak_gpu_memory: self.peak_gpu_memory = peak_gpu
        
        process = psutil.Process(os.getpid())
        cpu_mem = process.memory_info().rss / (1024 ** 2)
        self.cpu_memory_usage.append(cpu_mem)
        if cpu_mem > self.peak_cpu_memory: self.peak_cpu_memory = cpu_mem
    
    def set_model_size(self, model):
        param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
        self.model_size_mb = (param_size + buffer_size) / (1024 ** 2)
    
    def get_summary(self):
        return {
            'total_batches': len(self.batch_times),
            'total_samples_processed': self.total_samples_processed,
            'avg_batch_inference_time': np.mean(self.inference_times) if self.inference_times else 0,
            'min_batch_inference_time': np.min(self.inference_times) if self.inference_times else 0,
            'max_batch_inference_time': np.max(self.inference_times) if self.inference_times else 0,
            'std_batch_inference_time': np.std(self.inference_times) if self.inference_times else 0,
            'total_inference_time': np.sum(self.inference_times),
            'avg_dataload_time': np.mean(self.dataload_times) if self.dataload_times else 0,
            'total_dataload_time': np.sum(self.dataload_times),
            'throughput_samples_per_sec': self.total_samples_processed / np.sum(self.inference_times) if self.inference_times else 0,
            'throughput_batches_per_sec': len(self.batch_times) / np.sum(self.batch_times) if self.batch_times else 0,
            'avg_gpu_memory_mb': np.mean(self.gpu_memory_usage) if self.gpu_memory_usage else 0,
            'peak_gpu_memory_mb': self.peak_gpu_memory,
            'avg_cpu_memory_mb': np.mean(self.cpu_memory_usage) if self.cpu_memory_usage else 0,
            'peak_cpu_memory_mb': self.peak_cpu_memory,
            'model_size_mb': self.model_size_mb,
            'memory_per_sample_mb': self.peak_gpu_memory / self.total_samples_processed if self.total_samples_processed > 0 else 0,
        }

# ==================== DATASET CONVERSION ====================

def convert_mtl_to_stl(y_mtl):
    label_mapping = {
        (0, 0, 0): 0,  # Normal
        (1, 0, 0): 1,  # Missing
        (0, 1, 0): 2,  # Trend
        (0, 0, 1): 3,  # Drift
        (1, 1, 0): 4,  # Missing+Trend
        (1, 0, 1): 5,  # Missing+Drift
        (0, 1, 1): 6,  # Trend+Drift
    }
    
    y_stl = np.zeros(len(y_mtl), dtype=np.int64)
    for idx, label in enumerate(y_mtl):
        label_tuple = tuple(label)
        if label_tuple not in label_mapping:
            raise ValueError(f"Unknown label combination at index {idx}: {label_tuple}")
        y_stl[idx] = label_mapping[label_tuple]
    return y_stl, label_mapping


def compute_class_weights(y_stl, num_classes=7):
    counter = Counter(y_stl)
    weights = np.zeros(num_classes)
    total_samples = len(y_stl)
    for class_id in range(num_classes):
        count = counter.get(class_id, 0)
        if count > 0:
            weights[class_id] = total_samples / (num_classes * count)
        else:
            weights[class_id] = 1.0
    weights = weights / weights.sum() * num_classes
    return torch.from_numpy(weights).float()


# ==================== CUSTOM LOSS FUNCTIONS ====================

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, reduction='none')
    
    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            if self.alpha.device != focal_loss.device:
                self.alpha = self.alpha.to(focal_loss.device)
            focal_loss = self.alpha[targets] * focal_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ClassBalancedLoss(nn.Module):
    def __init__(self, num_classes=7, samples_per_class=None, beta=0.9999, weight=None, reduction='mean'):
        super(ClassBalancedLoss, self).__init__()
        self.num_classes = num_classes
        self.beta = beta
        self.reduction = reduction
        if samples_per_class is not None:
            self.effective_num = 1.0 - np.power(beta, samples_per_class)
            self.weights = (1.0 - beta) / np.asarray(self.effective_num)
            self.weights = self.weights / self.weights.sum() * num_classes
            self.weights = torch.tensor(self.weights, dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu') if weight is not None else None
        else:
            self.weights = None
        self.ce_loss = nn.CrossEntropyLoss(weight=self.weights, reduction='none')
    
    def forward(self, inputs, targets):
        loss = self.ce_loss(inputs, targets)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, num_classes=7, samples_per_class=None, reduction='mean'):
        super(BalancedSoftmaxLoss, self).__init__()
        self.reduction = reduction
        if samples_per_class is not None:
            total_samples = np.sum(samples_per_class)
            class_priors = samples_per_class / total_samples
            self.class_log_priors = torch.from_numpy(np.log(class_priors)).float()
        else:
            self.class_log_priors = None
    
    def forward(self, inputs, targets):
        if self.class_log_priors is not None:
            if self.class_log_priors.device != inputs.device:
                self.class_log_priors = self.class_log_priors.to(inputs.device)
            adjusted_inputs = inputs + self.class_log_priors.unsqueeze(0)
        else:
            adjusted_inputs = inputs
        log_softmax = torch.nn.functional.log_softmax(adjusted_inputs, dim=1)
        loss = -log_softmax.gather(1, targets.unsqueeze(1)).squeeze(1)
        if self.reduction == 'mean': return loss.mean()
        elif self.reduction == 'sum': return loss.sum()
        else: return loss

class LDAMLoss(nn.Module):
    def __init__(self, num_classes=7, samples_per_class=None, margin_scale=0.5, reduction='mean'):
        super(LDAMLoss, self).__init__()
        self.reduction = reduction
        if samples_per_class is not None:
            margins = 1.0 / (np.power(samples_per_class, 0.25))
            margins = margins / np.max(margins) * margin_scale
            self.margins = torch.from_numpy(margins).float()
        else:
            self.margins = torch.zeros(num_classes)
    
    def forward(self, inputs, targets):
        if self.margins.device != inputs.device:
            self.margins = self.margins.to(inputs.device)
        batch_margins = self.margins[targets]
        adjusted_inputs = inputs.clone()
        adjusted_inputs[range(len(targets)), targets] -= batch_margins
        log_softmax = torch.nn.functional.log_softmax(adjusted_inputs, dim=1)
        loss = -log_softmax.gather(1, targets.unsqueeze(1)).squeeze(1)
        if self.reduction == 'mean': return loss.mean()
        elif self.reduction == 'sum': return loss.sum()
        else: return loss

def create_loss_function(loss_type, num_classes=7, samples_per_class=None, class_weights=None, device='cuda'):
    if loss_type == 'focal_loss':
        return FocalLoss(alpha=class_weights.to(device) if class_weights is not None else None, gamma=2.0)
    elif loss_type == 'class_balanced':
        return ClassBalancedLoss(num_classes=num_classes, samples_per_class=samples_per_class)
    elif loss_type == 'balanced_softmax':
        return BalancedSoftmaxLoss(num_classes=num_classes, samples_per_class=samples_per_class)
    elif loss_type == 'ldam':
        return LDAMLoss(num_classes=num_classes, samples_per_class=samples_per_class)
    elif loss_type == 'weighted_ce':
        return nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    raise ValueError(f"Unknown loss function: {loss_type}")

# ==================== DATASET & DATALOADER ====================

class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series anomaly detection"""
    
    def __init__(self, X, y, transform=None):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        # Add channel dimension for CNN: (1008,) -> (1, 1008)
        x = x.unsqueeze(0)
        
        if self.transform:
            x = self.transform(x)
        
        return x, y

# ==================== SIMPLIFIED CNN-LSTM MODEL ====================

class SimplifiedCNNLSTMModel(nn.Module):
    def __init__(self, input_channels=1, sequence_length=1008, num_classes=7, dropout_rate=0.3):
        super(SimplifiedCNNLSTMModel, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=5, padding=2, stride=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2, stride=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, padding=2, stride=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        
        self.lstm = nn.LSTM(
            input_size=256, hidden_size=256, num_layers=2, 
            batch_first=True, dropout=0.3, bidirectional=False
        )
        
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.dropout(x)
        
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        x = torch.mean(lstm_out, dim=1)
        
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ==================== TRAINING & EVALUATION ====================

class ModelTrainer:
    def __init__(self, model, device, loss_fn=None, class_weights=None):
        self.model = model
        self.device = device
        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss(
            weight=class_weights.to(device) if class_weights is not None else None
        )
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1)
    
    def train_epoch(self, train_loader, metrics=None, batch_size=128):
        self.model.train()
        total_loss = 0.0
        for batch_idx, (x, y) in enumerate(train_loader):
            if metrics: metrics.start_batch()
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            
            if metrics: metrics.start_forward()
            outputs = self.model(x)
            if metrics: metrics.end_forward()
            
            loss = self.loss_fn(outputs, y)
            
            if metrics: metrics.start_backward()
            loss.backward()
            if metrics: metrics.end_backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
            
            if metrics: 
                metrics.end_batch(x.size(0))
                metrics.record_cpu_memory()
                
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                loss = self.loss_fn(outputs, y)
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        return total_loss / len(val_loader), accuracy, f1, all_preds, all_labels
    
    def test(self, test_loader):
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        return np.array(all_preds), np.array(all_labels), np.array(all_probs)

def run_inference(model, device, test_loader, metrics):
    """用于执行带指标跟踪的单次推理"""
    model.eval()
    y_true, y_prob, y_pred = [], [], []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            metrics.start_dataload()
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            metrics.end_dataload()
            
            metrics.start_inference()
            outputs = model(x_batch)
            metrics.end_inference(x_batch.size(0))

            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            y_true.extend(y_batch.cpu().numpy())
            y_prob.extend(probs)
            y_pred.extend(preds)

    return np.array(y_true), np.array(y_prob), np.array(y_pred)

def convert_stl_predictions_to_mtl(y_pred_stl):
    mapping = {0: [0,0,0], 1: [1,0,0], 2: [0,1,0], 3: [0,0,1], 4: [1,1,0], 5: [1,0,1], 6: [0,1,1]}
    return np.array([mapping[pred] for pred in y_pred_stl])

def evaluate_mtl_metrics(y_true_mtl, y_pred_mtl):
    task_names = ["1_missing", "2_trend", "3_drift"]
    results = {}
    for task_idx, task_name in enumerate(task_names):
        y_true_task = y_true_mtl[:, task_idx]
        y_pred_task = y_pred_mtl[:, task_idx]
        class_report = classification_report(y_true_task, y_pred_task, target_names=['0', '1'], output_dict=True)
        roc_auc = roc_auc_score(y_true_task, y_pred_task) if len(np.unique(y_true_task)) > 1 else np.nan
        results[task_name] = {
            'classification_report': class_report,
            'roc_auc': roc_auc,
            'accuracy': accuracy_score(y_true_task, y_pred_task)
        }
    return results

def main(x_train_fold, y_train_fold, x_val_fold, y_val_fold, x_test, y_test,
         num_epochs=50, batch_size=32, loss_fn_type='weighted_ce', metrics=None):
    y_train_stl, _ = convert_mtl_to_stl(y_train_fold)
    y_val_stl, _ = convert_mtl_to_stl(y_val_fold)
    y_test_stl, _ = convert_mtl_to_stl(y_test)
    
    counter = Counter(y_train_stl)
    samples_per_class = np.array([counter.get(i, 0) for i in range(7)])
    
    scaler = StandardScaler()
    x_train_fold = scaler.fit_transform(x_train_fold)
    x_val_fold = scaler.transform(x_val_fold)
    x_test = scaler.transform(x_test)
    
    train_dataset = TimeSeriesDataset(x_train_fold, y_train_stl)
    val_dataset = TimeSeriesDataset(x_val_fold, y_val_stl)
    test_dataset = TimeSeriesDataset(x_test, y_test_stl)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    class_weights = compute_class_weights(y_train_stl, num_classes=7)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SimplifiedCNNLSTMModel().to(device)
    loss_fn = create_loss_function(loss_fn_type, num_classes=7, samples_per_class=samples_per_class, class_weights=class_weights, device=device)
    trainer = ModelTrainer(model, device, loss_fn=loss_fn, class_weights=class_weights)
    
    best_val_f1 = 0.0
    for epoch in range(num_epochs):
        if metrics: metrics.start_epoch()
        
        train_loss = trainer.train_epoch(train_loader, metrics=metrics, batch_size=batch_size)
        _, _, val_f1, _, _ = trainer.validate(val_loader)
        trainer.scheduler.step()
        
        if metrics: metrics.end_epoch()
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'best_model_CNNLSTM.pth')

    torch.save(model.state_dict(), 'last_model_CNNLSTM.pth')
    model.load_state_dict(torch.load('last_model_CNNLSTM.pth'))
    
    y_pred_stl, y_true_stl_test, _ = trainer.test(test_loader)
    y_pred_mtl = convert_stl_predictions_to_mtl(y_pred_stl)
    mtl_results = evaluate_mtl_metrics(y_test, y_pred_mtl)
    
    return model, {}, mtl_results, y_pred_stl, y_true_stl_test, test_loader, device


if __name__ == "__main__":
    save_path = 'saved_models/PLE_mode_{}'.format(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    os.makedirs(save_path, exist_ok=True)
    print(f"\nExperiment directory: {save_path}")

    _, _, x_all, y_all, _ = data_preprocessing(balanced=False, platform='pytorch', normal_class=False, method='test')
    
    np.random.seed(8)
    indices = np.arange(x_all.shape[0])
    np.random.shuffle(indices)
    x_all = x_all[indices]
    y_all = y_all[indices]

    x_train_val, y_train_val = x_all[:1436,:], y_all[:1436,:]
    x_test, y_test = x_all[1436:,:], y_all[1436:,:]

    all_runs_metrics = []
    report_logs = []
    num_runs = 5
    
    final_model = None
    final_test_loader = None
    final_device = None
    
    for run_id in range(1, num_runs + 1):
        print(f"\n{'='*60}\nTraining Run {run_id}/{num_runs}\n{'='*60}")
        run_start_time = time.time()
        run_metrics = ComputationalMetrics()

        stratify_labels = y_train_val[:, 0]
        skf = StratifiedKFold(n_splits=4, shuffle=False)

        for fold, (train_idx, val_idx) in enumerate(skf.split(x_train_val, stratify_labels), 1):
            if fold == 4:
                x_train_fold, y_train_fold = x_train_val[train_idx], y_train_val[train_idx]
                x_val_fold, y_val_fold = x_train_val[val_idx], y_train_val[val_idx]

                model, _, mtl_results, _, _, test_loader, device = main(
                    x_train_fold=x_train_fold[:, :1008],
                    y_train_fold=y_train_fold,
                    x_val_fold=x_val_fold[:, :1008],
                    y_val_fold=y_val_fold,
                    x_test=x_test[:, :1008],
                    y_test=y_test,
                    num_epochs=100,
                    batch_size=128,
                    loss_fn_type=LOSS_FUNCTION,
                    metrics=run_metrics
                )
                
                final_model = model
                final_test_loader = test_loader
                final_device = device

        run_elapsed = time.time() - run_start_time
        run_metrics_summary = run_metrics.get_summary()
        run_metrics_summary['run_id'] = run_id
        run_metrics_summary['total_run_time'] = run_elapsed
        all_runs_metrics.append(run_metrics_summary)

    # =========================================================================
    # INFERENCE EFFICIENCY TESTING (Similar to test_MPAD_CGC_efficiency.py)
    # =========================================================================
    print(f"\n{'='*60}\nINFERENCE COMPUTATIONAL EFFICIENCY TESTING\n{'='*60}")
    
    inference_runs_metrics = []
    
    # Calculate Model Size
    initial_inf_metrics = InferenceMetrics()
    initial_inf_metrics.set_model_size(final_model)
    model_size_mb = initial_inf_metrics.model_size_mb
    print(f"Model Size: {model_size_mb:.2f} MB\n")
    
    for run_id in range(1, num_runs + 1):
        print(f"{'-'*40}\nInference Run {run_id}/{num_runs}\n{'-'*40}")
        inf_metrics = InferenceMetrics()
        inf_metrics.set_model_size(final_model)
        
        inf_metrics.start_run()
        y_true, y_prob, y_pred = run_inference(final_model, final_device, final_test_loader, inf_metrics)
        run_time = inf_metrics.end_run()
        
        inf_summary = inf_metrics.get_summary()
        inf_summary['run_id'] = run_id
        inf_summary['total_run_time'] = run_time
        inference_runs_metrics.append(inf_summary)
        
        print(f"Run {run_id} Inference completed in {run_time:.4f}s")
        print(f"Throughput: {inf_summary['throughput_samples_per_sec']:.2f} samples/sec")

    # Generate Inference Report
    efficiency_report = [
        "\n" + "="*60, "INFERENCE COMPUTATIONAL EFFICIENCY ANALYSIS", "="*60,
        f"\nReport Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Device: {final_device}",
        f"Model Size: {model_size_mb:.2f} MB",
        f"\nTotal Inference Runs: {num_runs}",
        f"Test Samples: {len(x_test)}",
        f"Total Batches per Run: {len(final_test_loader)}\n",
        "-"*60, "PER-RUN INFERENCE METRICS", "-"*60
    ]
    
    all_inf_times, all_inf_throughputs, all_inf_gpu_mem, all_inf_cpu_mem, all_inf_run_times = [], [], [], [], []
    
    for inf_summary in inference_runs_metrics:
        efficiency_report.extend([
            f"\nRun {int(inf_summary['run_id'])}:",
            f"  Total Run Time: {inf_summary['total_run_time']:.4f}s",
            f"  Total Inference Time: {inf_summary['total_inference_time']:.4f}s",
            f"  Average Batch Inference Time: {inf_summary['avg_batch_inference_time']:.6f}s",
            f"  Throughput (Samples/sec): {inf_summary['throughput_samples_per_sec']:.2f}",
            f"  Throughput (Batches/sec): {inf_summary['throughput_batches_per_sec']:.2f}",
            f"  Average GPU Memory: {inf_summary['avg_gpu_memory_mb']:.2f} MB",
            f"  Peak GPU Memory: {inf_summary['peak_gpu_memory_mb']:.2f} MB",
            f"  Average CPU Memory: {inf_summary['avg_cpu_memory_mb']:.2f} MB"
        ])
        all_inf_times.append(inf_summary['total_inference_time'])
        all_inf_throughputs.append(inf_summary['throughput_samples_per_sec'])
        all_inf_gpu_mem.append(inf_summary['avg_gpu_memory_mb'])
        all_inf_cpu_mem.append(inf_summary['avg_cpu_memory_mb'])
        all_inf_run_times.append(inf_summary['total_run_time'])

    efficiency_report.extend(["\n" + "-"*60, "AGGREGATE METRICS (All Runs)", "-"*60])
    if all_inf_times:
        efficiency_report.extend([
            f"\nInference Time Statistics:",
            f"  Mean: {np.mean(all_inf_times):.4f}s (±{np.std(all_inf_times):.4f}s)",
            f"\nThroughput Statistics (Samples/sec):",
            f"  Mean: {np.mean(all_inf_throughputs):.2f} (±{np.std(all_inf_throughputs):.2f})",
            f"\nGPU Memory Statistics (MB):",
            f"  Mean: {np.mean(all_inf_gpu_mem):.2f} (±{np.std(all_inf_gpu_mem):.2f})",
            f"\nLatency per Sample: {np.mean(all_inf_times) / len(x_test) * 1000:.2f} ms",
        ])

    report_text = "\n".join(efficiency_report)
    print(report_text)

    # Save to disk
    eff_report_path = os.path.join(save_path, "inference_efficiency_report.txt")
    with open(eff_report_path, "w") as f:
        f.write(report_text)
    
    eff_csv_path = os.path.join(save_path, "inference_efficiency_metrics.csv")
    pd.DataFrame(inference_runs_metrics).to_csv(eff_csv_path, index=False)
    
    print(f"\n✅ Inference efficiency report saved to {eff_report_path}")
    print(f"✅ Inference efficiency metrics saved to {eff_csv_path}")