'''
测试脚本：使用训练好的多任务MPAD-CGC模型对测试集进行推理与评估
功能概述:   
    该脚本加载预训练的MPAD-CGC模型，对数据集中预划分的测试子集进行推理与评估。
    评估指标包括分类报告（精确率、召回率、F1分数）、ROC AUC、混淆矩阵等。
    评估结果（每个任务的真实标签、预测标签、概率等）将保存到 RESULTS_DIR/original_argmax 目录，
    并在控制台打印完成提示。
'''
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from pytorchModels.ple_Inception_Features_analysis_train import PLE
from preprocessing_addFeatures import data_preprocessing
from torch.utils.data import DataLoader, TensorDataset

# 绘制混淆矩阵
def plot_confusion_matrix(cm, task_name, class_names=['0', '1'], save_path=None):
    # 创建图像
    plt.figure(figsize=(3, 3))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)  # 使用蓝色渐变展示矩阵
    plt.title(f'Confusion Matrix - {task_name}')  # 标题包含任务名称
    plt.colorbar()  # 颜色条用于参考
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)  # 设置预测标签刻度
    plt.yticks(tick_marks, class_names)  # 设置真实标签刻度

    # 在每个单元格写入数值，颜色根据背景变化
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, f"{cm[i, j]}", ha='center', va='center',
                 color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.xlabel('Predicted')  # x轴：预测
    plt.ylabel('True')       # y轴：真实
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)  # 保存高清图
    plt.close()  # 关闭避免内存累积

# 绘制错误分类样本的原始信号
def plot_misclassified_signal(signal, pred, true, task, index, save_path=None):
    plt.figure(figsize=(10, 3))
    plt.plot(signal, label='Signal')  # 绘制信号曲线
    # 标题包含任务名、索引、预测标签、真实标签
    plt.title(f'Task: {task} | Index: {index} | Predicted: {pred} | True: {true}')
    plt.xlabel('Timestep')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)  # 保存图像
    plt.close()

# 评估单任务预测结果：分类报告、AUC、混淆矩阵、错分样本可视化
def evaluate_predictions(y_true, y_pred, y_prob, label, x_val, val_indices, results_dir, label_prefix="original", threshold='None'):
    # 生成分类报告（精确率、召回率、F1）
    report = classification_report(y_true, y_pred, target_names=['0', '1'], digits=4)
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 写入评估指标到文本文件
    with open(os.path.join(results_dir, f'{label_prefix}_metrics.txt'), 'a') as f:
        f.write(f"\n=== Task: {label} ===\n")
        f.write(f"threshold: {threshold}\n")
        f.write(report + '\n')
        # 计算并写入ROC AUC（若只有一种类别则异常处理）
        try:
            auc = roc_auc_score(y_true, y_prob)
            f.write(f"ROC AUC: {auc:.4f}\n")
        except:
            f.write("ROC AUC: Undefined (only one class present in y_true)\n")

    # 保存混淆矩阵图
    cm_path = os.path.join(results_dir, f'{label_prefix}_confusion_matrix_{label}.png')
    plot_confusion_matrix(cm, label, save_path=cm_path)

    # 错误分类样本的图像保存目录
    task_dir = os.path.join(results_dir, f'misclassified_{label_prefix}_{label}')
    os.makedirs(task_dir, exist_ok=True)
    # 遍历预测结果，挑选错误样本进行信号可视化
    for idx, (pred, true_label) in enumerate(zip(y_pred, y_true)):
        if pred != true_label:
            raw_idx = val_indices[idx]  # 恢复原始数据索引
            save_path = os.path.join(task_dir, f'raw_index_{raw_idx}_pred_{pred}_true_{true_label}.png')
            # 只截取前1008长度的主时序部分进行展示
            plot_misclassified_signal(x_val[idx][:1008], pred, true_label, label, raw_idx, save_path)

# 测试入口：加载数据与模型，推理并评估三个子任务
def test(model_path):
    """
    功能概述:
        使用训练好的多任务MPAD-CGC模型对数据集中预划分的测试子集进行推理与评估。
        该函数不返回值，而是将评估结果（每个任务的真实标签、预测标签、概率等）保存到
        RESULTS_DIR/original_argmax 目录，并在控制台打印完成提示。
    参数说明:
        model_path (str | pathlib.Path):
            已训练模型的权重文件路径（例如 .pt 或 .pth），用于加载 state_dict。
    数据流程与处理细节:
        1. 设备选择: 优先使用 CUDA，可回退到 CPU。
        2. 数据预处理: 调用 data_preprocessing(balanced=False, platform='pytorch',
           normal_class=False, method='test')，返回:
               x_train, y_train: 训练集（当前函数不使用）
               x_all, y_all: 全量样本特征与标签
               all_indices: 原始索引，用于结果回写或追踪
        3. 打乱顺序: 固定随机种子 8，保证结果可复现。
        4. 划分测试集: 使用打乱后的 x_all / y_all / all_indices 的第 1436 行及之后作为测试集。
        5. 特征张量构造: x_test 转为形状 (N, 1, L+F)，其中:
               主序列长度假设为 1008，后续额外特征从索引 1008 开始。
        6. 标签处理: 原始 y_test 为整数类别（0/1）；转换为 one-hot，形状 (N, 2)；
           多任务组织方式: 外层批次对应任务顺序 [1_missing, 2_trend, 3_drift]。
        7. 数据加载: 使用 DataLoader(batch_size=10, shuffle=False) 顺序推理。
    示例使用 (Example):
        test("checkpoints/ple_epoch20.pt")
    """
    # 设备选择：优先CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据预处理：返回训练、全量集合及其索引（此处只用全量做测试切分）
    x_train, y_train, x_all, y_all, all_indices = data_preprocessing(balanced=False, platform='pytorch', normal_class=False, method='test')
    
    # 固定随机种子并打乱顺序，保持可复现
    np.random.seed(8)
    indices = np.arange(x_all.shape[0])
    np.random.shuffle(indices)
    x_all = x_all[indices]
    y_all = y_all[indices]
    all_indices = all_indices[indices]

    # 划分测试集（从索引1436开始）
    x_test = x_all[1436:, :] # 占比20%
    y_test = y_all[1436:, :] # 占比20%
    test_indices = all_indices[1436:]

    # 构建验证（测试）集的Tensor形式（增加通道维度）
    x_val_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)
    y_val_tensor = torch.tensor(y_test, dtype=torch.float32)
    # 将标签转换为 one-hot（每个任务两类）
    y_val_tensor = torch.eye(2)[y_val_tensor.long(), :]
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)
    # 抽取附加特征维度（假设主序列长度为1008，后面是特征）
    num_features = x_val_tensor[:, :, 1008:].shape[-1]

    # 构建MPAD-CGC模型（参数根据具体结构设定）
    # ！！！请注意，这里的全部模型参数必须与训练时保持一致（特别注意num_shared_experts和num_task_experts），否则会报错！！！
    model = PLE(
        inputs_dim=num_features,
        labels_dict={
            '1_missing': 2,
            '2_trend': 2,
            '3_drift': 2,
        },
        dnn_dropout=0.2,
        num_shared_experts=9,
        num_task_experts=6,
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
    # 结果容器：真实标签、概率、预测标签
    y_true, y_prob, y_pred_orig = {}, {}, {}

    # 推理阶段（不计算梯度）
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(x_batch)  # 输出为每个任务的logits

            # 遍历每个任务，提取对应结果
            for i, task in dict_classes.items():
                probs = torch.softmax(outputs[task], dim=1)[:, 1].cpu().numpy()  # 正类概率
                preds = torch.argmax(outputs[task], dim=1).cpu().numpy()        # 预测类别
                labels = torch.argmax(y_batch[:, i], dim=1).cpu().numpy()       # 真实标签

                y_true.setdefault(task, []).extend(labels)
                y_prob.setdefault(task, []).extend(probs)
                y_pred_orig.setdefault(task, []).extend(preds)

    # 原始 argmax 策略结果保存目录
    original_dir = os.path.join(RESULTS_DIR, 'original_argmax')
    os.makedirs(original_dir, exist_ok=True)
    # 分任务评估与输出
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

    print(f"\n✅ Evaluation complete - {original_dir}\n")

# 主入口：设置模型与结果目录并启动测试
if __name__ == '__main__':
    # 模型权重文件路径（训练阶段保存的 .pth 文件）
    model_path = 'saved_models\\PLE_mode_2025-11-24-15-20-20\\model_2025-11-24-15-24-31_fold_4.pth'
    # 结果输出目录（用于存放评估指标、混淆矩阵、错分样本等）
    RESULTS_DIR = 'saved_models\\PLE_mode_2025-11-24-15-20-20\\model_2025-11-24-15-24-31_fold_4'
    # 若目录不存在则创建，保证写文件不报错
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # 调用测试函数：加载模型并对测试集推理与评估
    test(model_path)