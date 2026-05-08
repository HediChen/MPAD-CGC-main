"""
Advanced Single-Task Learning DenseNet+Transformer Model for Multi-Class Anomaly Detection
Inference and Computational Efficiency Testing Script

This script loads a pre-trained DenseNet+Transformer model, performs inference on the test set,
and strictly evaluates the computational efficiency over 5 runs (throughput, latency, memory).
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import time
import gc
import psutil
from datetime import datetime
from preprocessing_addFeatures import data_preprocessing
import pandas as pd

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ==================== INFERENCE METRICS ====================

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

# ==================== DATASET & DATALOADER ====================

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

# ==================== DENSENET BLOCK ====================

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate=32, num_layers=4):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        self.growth_rate = growth_rate
        
        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.BatchNorm1d(in_channels + i * growth_rate),
                nn.ReLU(inplace=True),
                nn.Conv1d(in_channels + i * growth_rate, growth_rate, 
                         kernel_size=3, padding=1, stride=1)
            ))
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, 1))
            features.append(out)
        return torch.cat(features, 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionBlock, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.AvgPool1d(kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        return self.block(x)

# ==================== POSITIONAL ENCODING ====================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# ==================== MULTI-HEAD ATTENTION ====================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=256, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = np.sqrt(self.d_k)
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.d_model)
        
        output = self.out_linear(context)
        return output, attn_weights

# ==================== TRANSFORMER ENCODER LAYER ====================

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, num_heads=8, ff_dim=1024, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        attn_out, attn_weights = self.attn(x, x, x, mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)
        
        return x, attn_weights

# ==================== DENSENET + TRANSFORMER MODEL ====================

class DenseNetTransformerModel(nn.Module):
    def __init__(self, input_channels=1, sequence_length=1008, num_classes=7,
                 growth_rate=32, num_dense_layers=4, num_dense_blocks=3,
                 d_model=256, num_heads=8, num_transformer_layers=4,
                 ff_dim=1024, dropout_rate=0.3):
        super(DenseNetTransformerModel, self).__init__()
        
        self.conv_init = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        current_channels = 64
        
        self.dense_blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        
        for i in range(num_dense_blocks):
            dense_block = DenseBlock(current_channels, growth_rate, num_dense_layers)
            self.dense_blocks.append(dense_block)
            current_channels = current_channels + num_dense_layers * growth_rate
            
            if i < num_dense_blocks - 1:
                transition_channels = current_channels // 2
                transition = TransitionBlock(current_channels, transition_channels)
                self.transitions.append(transition)
                current_channels = transition_channels
        
        self.final_bn = nn.BatchNorm1d(current_channels)
        
        feature_map_size = 1008 // 4 
        for i in range(num_dense_blocks - 1):
            feature_map_size = feature_map_size // 2 
        
        self.feature_projection = nn.Conv1d(current_channels, d_model, kernel_size=1)
        self.pos_encoding = PositionalEncoding(d_model, max_len=feature_map_size + 10)
        
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, ff_dim, dropout_rate)
            for _ in range(num_transformer_layers)
        ])
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(d_model, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv_init(x) 
        
        for i, dense_block in enumerate(self.dense_blocks):
            x = dense_block(x)
            if i < len(self.transitions):
                x = self.transitions[i](x)
        
        x = self.final_bn(x) 
        x = self.feature_projection(x) 
        x = x.transpose(1, 2)
        x = self.pos_encoding(x)
        
        for transformer_layer in self.transformer_layers:
            x, _ = transformer_layer(x)
        
        x = torch.mean(x, dim=1) 
        
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

# ==================== INFERENCE PIPELINE ====================

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

def test(model_path, results_dir):
    """
    加载模型和数据（仅加载一次），准备推理环境
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 数据预处理
    _, _, x_all, y_all, _ = data_preprocessing(balanced=False, platform='pytorch', normal_class=False, method='test')
    
    np.random.seed(8)
    indices = np.arange(x_all.shape[0])
    np.random.shuffle(indices)
    x_all = x_all[indices]
    y_all = y_all[indices]

    # 切分和标准化 (仅保留测试集数据)
    x_train_val = x_all[:1436, :1008]
    x_test = x_all[1436:, :1008]
    y_test = y_all[1436:, :]

    y_test_stl, _ = convert_mtl_to_stl(y_test)
    
    scaler = StandardScaler()
    scaler.fit(x_train_val)
    x_test = scaler.transform(x_test)
    
    test_dataset = TimeSeriesDataset(x_test, y_test_stl)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # 建立模型
    model = DenseNetTransformerModel(
        input_channels=1,
        sequence_length=1008,
        num_classes=7,
        growth_rate=32,
        num_dense_layers=4,
        num_dense_blocks=3,
        d_model=256,
        num_heads=8,
        num_transformer_layers=1,
        ff_dim=1024,
        dropout_rate=0.3
    )
    
    # 加载权重
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Successfully loaded checkpoint from: {model_path}")
    except FileNotFoundError:
        print(f"⚠️ Warning: Checkpoint not found at {model_path}. Initializing with random weights for efficiency testing.")
        
    model.to(device)
    model.eval()

    return model, device, test_loader, x_test

if __name__ == "__main__":
    # 模型权重文件路径（请替换为实际训练保存的 .pth 文件路径）
    model_path = 'saved_models/best_densenet_transformer_model.pth'
    
    # 结果输出目录
    RESULTS_DIR = 'saved_models/DenseNet_Transformer_Inference_Results'
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print("DenseNet+Transformer Model Inference & Efficiency Testing")
    print(f"{'='*60}\n")

    # 1. 加载模型和数据（仅加载一次）
    model, device, test_loader, x_test = test(model_path, RESULTS_DIR)

    # 2. 计算并输出模型大小
    initial_inf_metrics = InferenceMetrics()
    initial_inf_metrics.set_model_size(model)
    model_size_mb = initial_inf_metrics.model_size_mb
    print(f"Model Size: {model_size_mb:.2f} MB\n")

    # 3. 开始多次推理执行性能测试
    inference_runs_metrics = []
    num_runs = 5
    
    for run_id in range(1, num_runs + 1):
        print(f"{'-'*40}\nInference Run {run_id}/{num_runs}\n{'-'*40}")
        inf_metrics = InferenceMetrics()
        inf_metrics.set_model_size(model)
        
        inf_metrics.start_run()
        y_true, y_prob, y_pred = run_inference(model, device, test_loader, inf_metrics)
        run_time = inf_metrics.end_run()
        
        inf_summary = inf_metrics.get_summary()
        inf_summary['run_id'] = run_id
        inf_summary['total_run_time'] = run_time
        inference_runs_metrics.append(inf_summary)
        
        print(f"Run {run_id} Inference completed in {run_time:.4f}s")
        print(f"Throughput: {inf_summary['throughput_samples_per_sec']:.2f} samples/sec")

    # 4. 汇总与输出效率报告
    print(f"\n{'='*60}")
    print("INFERENCE EFFICIENCY REPORT")
    print(f"{'='*60}\n")

    efficiency_report = [
        "\n" + "="*60, "INFERENCE COMPUTATIONAL EFFICIENCY ANALYSIS", "="*60,
        f"\nReport Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Device: {device}",
        f"Model Size: {model_size_mb:.2f} MB",
        f"\nTotal Inference Runs: {num_runs}",
        f"Test Samples: {len(x_test)}",
        f"Total Batches per Run: {len(test_loader)}\n",
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

    # 5. 保存至文件
    eff_report_path = os.path.join(RESULTS_DIR, "inference_efficiency_report.txt")
    with open(eff_report_path, "w") as f:
        f.write(report_text)
    
    eff_csv_path = os.path.join(RESULTS_DIR, "inference_efficiency_metrics.csv")
    pd.DataFrame(inference_runs_metrics).to_csv(eff_csv_path, index=False)
    
    print(f"\n✅ Inference efficiency report saved to {eff_report_path}")
    print(f"✅ Inference efficiency metrics saved to {eff_csv_path}")