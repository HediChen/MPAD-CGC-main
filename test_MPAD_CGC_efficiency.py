'''
测试脚本：使用训练好的多任务MPAD-CGC模型对测试集进行推理与评估
功能概述:   
    该脚本加载预训练的MPAD-CGC模型，对数据集中预划分的测试子集进行推理与评估。
    评估指标包括分类报告（精确率、召回率、F1分数）、ROC AUC、混淆矩阵等。
    评估结果（每个任务的真实标签、预测标签、概率等）将保存到 RESULTS_DIR/original_argmax 目录。
    新增功能：执行五次推理并报告计算效率指标。
'''
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from pytorchModels.ple_Inception_Features_analysis_train import PLE
from preprocessing_addFeatures import data_preprocessing
from torch.utils.data import DataLoader, TensorDataset
import time
import psutil
import pandas as pd
import gc
from datetime import datetime
import matplotlib
matplotlib.use("Agg")


class InferenceMetrics:
    """推理效率指标跟踪类"""
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
        """开始推理运行"""
        self.run_start_time = time.time()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    def end_run(self):
        """结束推理运行"""
        if self.run_start_time:
            run_time = time.time() - self.run_start_time
            return run_time
        return 0
    
    def start_dataload(self):
        """开始数据加载计时"""
        self.dataload_start_time = time.time()
    
    def end_dataload(self):
        """结束数据加载计时"""
        if self.dataload_start_time:
            dataload_time = time.time() - self.dataload_start_time
            self.dataload_times.append(dataload_time)
    
    def start_inference(self):
        """开始推理计时"""
        self.inference_start_time = time.time()
    
    def end_inference(self, batch_size):
        """结束推理计时"""
        if self.inference_start_time:
            inference_time = time.time() - self.inference_start_time
            self.inference_times.append(inference_time)
            self.total_samples_processed += batch_size
            self.batch_times.append(inference_time)
            self.record_memory()
    
    def record_memory(self):
        """记录当前内存使用"""
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
            self.gpu_memory_usage.append(gpu_mem)
            peak_gpu = torch.cuda.max_memory_allocated() / (1024 ** 2)
            if peak_gpu > self.peak_gpu_memory:
                self.peak_gpu_memory = peak_gpu
        
        process = psutil.Process(os.getpid())
        cpu_mem = process.memory_info().rss / (1024 ** 2)  # MB
        self.cpu_memory_usage.append(cpu_mem)
        if cpu_mem > self.peak_cpu_memory:
            self.peak_cpu_memory = cpu_mem
    
    def set_model_size(self, model):
        """计算模型大小"""
        param_size = 0
        buffer_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        self.model_size_mb = (param_size + buffer_size) / (1024 ** 2)
    
    def get_summary(self):
        """获取推理指标摘要"""
        summary_dict = {
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
        return summary_dict


# 绘制混淆矩阵
def plot_confusion_matrix(cm, task_name, class_names=['0', '1'], save_path=None):
    """绘制混淆矩阵"""
    plt.figure(figsize=(3, 3))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {task_name}')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, f"{cm[i, j]}", ha='center', va='center',
                 color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


# 绘制错误分类样本的原始信号
def plot_misclassified_signal(signal, pred, true, task, index, save_path=None):
    """绘制错分样本的信号"""
    plt.figure(figsize=(10, 3))
    plt.plot(signal, label='Signal')
    plt.title(f'Task: {task} | Index: {index} | Predicted: {pred} | True: {true}')
    plt.xlabel('Timestep')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


# 评估单任务预测结果
def evaluate_predictions(y_true, y_pred, y_prob, label, x_val, val_indices, results_dir, label_prefix="original", threshold='None'):
    """评估预测结果：分类报告、AUC、混淆矩阵、错分样本可视化"""
    report = classification_report(y_true, y_pred, target_names=['0', '1'], digits=4)
    cm = confusion_matrix(y_true, y_pred)

    with open(os.path.join(results_dir, f'{label_prefix}_metrics.txt'), 'a') as f:
        f.write(f"\n=== Task: {label} ===\n")
        f.write(f"threshold: {threshold}\n")
        f.write(report + '\n')
        try:
            auc = roc_auc_score(y_true, y_prob)
            f.write(f"ROC AUC: {auc:.4f}\n")
        except:
            f.write("ROC AUC: Undefined (only one class present in y_true)\n")

    cm_path = os.path.join(results_dir, f'{label_prefix}_confusion_matrix_{label}.png')
    plot_confusion_matrix(cm, label, save_path=cm_path)

    task_dir = os.path.join(results_dir, f'misclassified_{label_prefix}_{label}')
    os.makedirs(task_dir, exist_ok=True)
    for idx, (pred, true_label) in enumerate(zip(y_pred, y_true)):
        if pred != true_label:
            raw_idx = val_indices[idx]
            save_path = os.path.join(task_dir, f'raw_index_{raw_idx}_pred_{pred}_true_{true_label}.png')
            plot_misclassified_signal(x_val[idx][:1008], pred, true_label, label, raw_idx, save_path)


# 测试入口：加载数据与模型，推理并评估三个子任务
def test(model_path, results_dir):
    """
    功能概述:
        使用训练好的多任务MPAD-CGC模型对数据集中预划分的测试子集进行推理与评估。
        该函数不返回值，而是将评估结果（每个任务的真实标签、预测标签、概率等）保存到
        指定目录，并在控制台打印完成提示。
    参数说明:
        model_path (str): 已训练模型的权重文件路径（.pth文件）
        results_dir (str): 结果输出目录路径
    """
    # 设备选择：优先CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 数据预处理：返回训练、全量集合及其索引（此处只用全量做测试切分）
    x_train, y_train, x_all, y_all, all_indices = data_preprocessing(
        balanced=False, platform='pytorch', normal_class=False, method='test'
    )
    
    # 固定随机种子并打乱顺序，保持可复现
    np.random.seed(8)
    indices = np.arange(x_all.shape[0])
    np.random.shuffle(indices)
    x_all = x_all[indices]
    y_all = y_all[indices]
    all_indices = all_indices[indices]

    # 划分测试集（从索引1436开始）
    x_test = x_all[1436:, :]
    y_test = y_all[1436:, :]
    test_indices = all_indices[1436:]

    # 构建验证（测试）集的Tensor形式（增加通道维度）
    x_val_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)
    y_val_tensor = torch.tensor(y_test, dtype=torch.float32)
    y_val_tensor = torch.eye(2)[y_val_tensor.long(), :]
    
    # 抽取附加特征维度（假设主序列长度为1008，后面是特征）
    num_features = x_val_tensor[:, :, 1008:].shape[-1]

    # 构建MPAD-CGC模型
    model = PLE(
        inputs_dim=num_features,
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
        device='cuda'
    )

    # 加载训练好的模型参数并设置评估模式
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 任务名称映射
    dict_classes = {0: '1_missing', 1: '2_trend', 2: '3_drift'}
    
    # 创建数据加载器
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=358, shuffle=False)
    
    return model, device, dict_classes, val_loader, x_test, test_indices


def run_inference(model, device, dict_classes, val_loader, metrics):
    """执行单次推理"""
    y_true, y_prob, y_pred_orig = {}, {}, {}

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            metrics.start_dataload()
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            metrics.end_dataload()
            
            metrics.start_inference()
            outputs = model(x_batch)
            metrics.end_inference(x_batch.size(0))

            # 遍历每个任务，提取对应结果
            for i, task in dict_classes.items():
                probs = torch.softmax(outputs[task], dim=1)[:, 1].cpu().numpy()
                preds = torch.argmax(outputs[task], dim=1).cpu().numpy()
                labels = torch.argmax(y_batch[:, i], dim=1).cpu().numpy()

                y_true.setdefault(task, []).extend(labels)
                y_prob.setdefault(task, []).extend(probs)
                y_pred_orig.setdefault(task, []).extend(preds)

    return y_true, y_prob, y_pred_orig


# 主入口：设置模型与结果目录并启动测试
if __name__ == '__main__':
    # 模型权重文件路径（训练阶段保存的 .pth 文件）
    model_path = 'saved_models/PLE_mode_2026-03-17-09-44-47/model_2026-03-17-09-45-51_run1_fold4.pth'
    
    # 结果输出目录（用于存放评估指标、混淆矩阵、错分样本等）
    RESULTS_DIR = 'saved_models/PLE_mode_2026-03-17-09-44-47/model_2026-03-17-09-45-51_run1_fold4_results'
    
    # 若目录不存在则创建，保证写文件不报错
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print("MPAD-CGC Model Inference Testing")
    print(f"{'='*60}\n")

    # 加载模型和数据（仅加载一次）
    model, device, dict_classes, val_loader, x_test, test_indices = test(model_path, RESULTS_DIR)
    
    # 设置模型大小
    initial_metrics = InferenceMetrics()
    initial_metrics.set_model_size(model)
    model_size_mb = initial_metrics.model_size_mb
    print(f"Model Size: {model_size_mb:.2f} MB\n")

    # 存储所有运行的指标
    all_runs_metrics = []
    num_runs = 5
    
    # 五次推理运行
    for run_id in range(1, num_runs + 1):
        print(f"{'='*60}")
        print(f"Inference Run {run_id}/{num_runs}")
        print(f"{'='*60}")
        
        metrics = InferenceMetrics()
        metrics.set_model_size(model)
        
        metrics.start_run()
        y_true, y_prob, y_pred_orig = run_inference(model, device, dict_classes, val_loader, metrics)
        run_time = metrics.end_run()
        
        # 评估当前运行的分类性能
        original_dir = os.path.join(RESULTS_DIR, f'run_{run_id}', 'original_argmax')
        os.makedirs(original_dir, exist_ok=True)
        
        # 初始化指标文件（清空之前的内容）
        with open(os.path.join(original_dir, 'original_metrics.txt'), 'w') as f:
            f.write(f"Inference Run {run_id} Classification Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n")
        
        for task in dict_classes.values():
            evaluate_predictions(
                y_true[task],
                y_pred_orig[task],
                y_prob[task],
                task,
                x_test,
                test_indices,
                original_dir,
                label_prefix="original"
            )
        
        # 获取指标摘要
        run_metrics_summary = metrics.get_summary()
        run_metrics_summary['run_id'] = run_id
        run_metrics_summary['total_run_time'] = run_time
        all_runs_metrics.append(run_metrics_summary)
        
        print(f"\nRun {run_id} completed in {run_time:.2f}s")
        print(f"Total Inference Time: {metrics.get_summary()['total_inference_time']:.2f}s")
        print(f"Throughput: {metrics.get_summary()['throughput_samples_per_sec']:.2f} samples/sec")
        print(f"Peak GPU Memory: {metrics.get_summary()['peak_gpu_memory_mb']:.2f} MB")
        print()

    # ��成推理效率报告
    print(f"\n{'='*60}")
    print("INFERENCE EFFICIENCY REPORT")
    print(f"{'='*60}\n")
    
    efficiency_report = []
    efficiency_report.append("\n" + "="*60)
    efficiency_report.append("INFERENCE COMPUTATIONAL EFFICIENCY ANALYSIS")
    efficiency_report.append("="*60)
    efficiency_report.append(f"\nReport Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    efficiency_report.append(f"Device: {device}")
    if torch.cuda.is_available():
        efficiency_report.append(f"GPU: {torch.cuda.get_device_name(0)}")
    efficiency_report.append(f"Model Size: {model_size_mb:.2f} MB")
    efficiency_report.append(f"\nTotal Inference Runs: {num_runs}")
    efficiency_report.append(f"Test Samples: {len(x_test)}")
    efficiency_report.append(f"Batch Size: 10")
    efficiency_report.append(f"Total Batches per Run: {len(val_loader)}")
    
    # 聚合所有运行的指标
    all_inference_times = []
    all_throughputs = []
    all_gpu_memory = []
    all_cpu_memory = []
    all_run_times = []
    
    efficiency_report.append("\n" + "-"*60)
    efficiency_report.append("PER-RUN INFERENCE METRICS")
    efficiency_report.append("-"*60)
    
    for run_summary in all_runs_metrics:
        efficiency_report.append(f"\nRun {int(run_summary['run_id'])}:")
        efficiency_report.append(f"  Total Run Time: {run_summary['total_run_time']:.4f}s")
        efficiency_report.append(f"  Total Inference Time: {run_summary['total_inference_time']:.4f}s")
        efficiency_report.append(f"  Total Data Loading Time: {run_summary['total_dataload_time']:.4f}s")
        efficiency_report.append(f"  Average Batch Inference Time: {run_summary['avg_batch_inference_time']:.6f}s")
        efficiency_report.append(f"  Min Batch Inference Time: {run_summary['min_batch_inference_time']:.6f}s")
        efficiency_report.append(f"  Max Batch Inference Time: {run_summary['max_batch_inference_time']:.6f}s")
        efficiency_report.append(f"  Std Batch Inference Time: {run_summary['std_batch_inference_time']:.6f}s")
        efficiency_report.append(f"  Throughput (Samples/sec): {run_summary['throughput_samples_per_sec']:.2f}")
        efficiency_report.append(f"  Throughput (Batches/sec): {run_summary['throughput_batches_per_sec']:.2f}")
        efficiency_report.append(f"  Average GPU Memory: {run_summary['avg_gpu_memory_mb']:.2f} MB")
        efficiency_report.append(f"  Peak GPU Memory: {run_summary['peak_gpu_memory_mb']:.2f} MB")
        efficiency_report.append(f"  Average CPU Memory: {run_summary['avg_cpu_memory_mb']:.2f} MB")
        efficiency_report.append(f"  Peak CPU Memory: {run_summary['peak_cpu_memory_mb']:.2f} MB")
        efficiency_report.append(f"  Memory per Sample: {run_summary['memory_per_sample_mb']:.6f} MB")
        
        all_inference_times.append(run_summary['total_inference_time'])
        all_throughputs.append(run_summary['throughput_samples_per_sec'])
        all_gpu_memory.append(run_summary['avg_gpu_memory_mb'])
        all_cpu_memory.append(run_summary['avg_cpu_memory_mb'])
        all_run_times.append(run_summary['total_run_time'])
    
    # 计算跨运行的聚合统计
    efficiency_report.append("\n" + "-"*60)
    efficiency_report.append("AGGREGATE METRICS (All Runs)")
    efficiency_report.append("-"*60)
    
    if all_inference_times:
        efficiency_report.append(f"\nInference Time Statistics:")
        efficiency_report.append(f"  Mean: {np.mean(all_inference_times):.4f}s (±{np.std(all_inference_times):.4f}s)")
        efficiency_report.append(f"  Min: {np.min(all_inference_times):.4f}s")
        efficiency_report.append(f"  Max: {np.max(all_inference_times):.4f}s")
    
    if all_throughputs:
        efficiency_report.append(f"\nThroughput Statistics (Samples/sec):")
        efficiency_report.append(f"  Mean: {np.mean(all_throughputs):.2f} (±{np.std(all_throughputs):.2f})")
        efficiency_report.append(f"  Min: {np.min(all_throughputs):.2f}")
        efficiency_report.append(f"  Max: {np.max(all_throughputs):.2f}")
    
    if all_gpu_memory:
        efficiency_report.append(f"\nGPU Memory Statistics (MB):")
        efficiency_report.append(f"  Mean: {np.mean(all_gpu_memory):.2f} (±{np.std(all_gpu_memory):.2f})")
        efficiency_report.append(f"  Min: {np.min(all_gpu_memory):.2f}")
        efficiency_report.append(f"  Max: {np.max(all_gpu_memory):.2f}")
    
    if all_cpu_memory:
        efficiency_report.append(f"\nCPU Memory Statistics (MB):")
        efficiency_report.append(f"  Mean: {np.mean(all_cpu_memory):.2f} (±{np.std(all_cpu_memory):.2f})")
        efficiency_report.append(f"  Min: {np.min(all_cpu_memory):.2f}")
        efficiency_report.append(f"  Max: {np.max(all_cpu_memory):.2f}")
    
    if all_run_times:
        total_inference_time = np.sum(all_inference_times)
        efficiency_report.append(f"\nRun Time Statistics (seconds):")
        efficiency_report.append(f"  Mean: {np.mean(all_run_times):.4f}s (±{np.std(all_run_times):.4f}s)")
        efficiency_report.append(f"  Total: {np.sum(all_run_times):.4f}s")
        efficiency_report.append(f"  Min: {np.min(all_run_times):.4f}s")
        efficiency_report.append(f"  Max: {np.max(all_run_times):.4f}s")
    
    efficiency_report.append(f"\nLatency per Sample: {np.mean(all_inference_times) / len(x_test) * 1000:.2f} ms")
    efficiency_report.append(f"Energy Efficiency (samples/sec/MB GPU): {np.mean(all_throughputs) / np.mean(all_gpu_memory):.4f}")
    
    # 打印报告到控制台
    report_text = "\n".join(efficiency_report)
    print(report_text)
    
    # 保存效率报告到文本文件
    efficiency_report_path = os.path.join(RESULTS_DIR, "inference_efficiency_report.txt")
    with open(efficiency_report_path, "w") as f:
        f.write(report_text)
    print(f"\n✅ Efficiency report saved to {efficiency_report_path}")
    
    # 保存效率指标为CSV
    efficiency_df = pd.DataFrame(all_runs_metrics)
    efficiency_csv_path = os.path.join(RESULTS_DIR, "inference_efficiency_metrics.csv")
    efficiency_df.to_csv(efficiency_csv_path, index=False)
    print(f"✅ Efficiency metrics saved to {efficiency_csv_path}")
    
    print(f"\n✅ Inference testing complete - Results saved to {RESULTS_DIR}\n")