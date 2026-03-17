'''
这是用于加载和分析多任务MPAD-CGC模型不同专家配置（共享专家和任务专家数量）分类性能的脚本。
当跑完train_MPAD_CGC_autoTrain.py脚本后，会在指定目录下生成多个配置文件夹，
每个文件夹包含该配置下的分类报告。
该脚本会遍历这些配置文件夹，提取每个配置在各折交叉验证中的分类性能指标（准确率和ROC AUC），
并进行排名和可视化分析，帮助选择最佳的专家配置组合。
'''

import sys
sys.path.append('./')
import os
import re
from collections import defaultdict

import os
import re
from collections import defaultdict
from typing import Dict, Tuple

def load_classification_metrics(base_dir: str) -> Tuple[
    Dict[Tuple[int, int], Dict[int, Dict[str, Dict[str, float]]]],
    Dict[Tuple[int, int], Dict[str, Dict[str, float]]]
]:
    """
    Load per-fold and final average classification metrics (accuracy & ROC AUC)
    from all PLE configuration folders under the specified base directory.

    Parameters:
        base_dir (str): Path to the folder containing PLE_Shared*_Task*_... folders.

    Returns:
        metrics_data: dict storing per-fold metrics
            {(num_shared, num_task): {fold: {task: {'accuracy': ..., 'roc_auc': ...}}}}

        final_metrics_data: dict storing final averaged metrics
            {(num_shared, num_task): {task: {'accuracy': ..., 'roc_auc': ...}}}
    """
    metrics_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    final_metrics_data = defaultdict(lambda: defaultdict(dict))

    # Regex patterns
    fold_pattern = re.compile(r"=== Classification Report: Fold (\d+) ===")
    task_pattern = re.compile(r"Task: (\d+_\w+)")
    accuracy_line_pattern = re.compile(r"^\s*accuracy\s+([\d.]+)\s+\d+")
    roc_auc_pattern = re.compile(r"ROC AUC:\s+([\d.]+)")
    final_start_pattern = re.compile(r"=== Final Average Classification Report \(All Folds\) ===")
    avg_task_pattern = re.compile(r"^Average Classification Report for Task: (\d+_\w+)")

    for folder in os.listdir(base_dir):
        if not folder.startswith("PLE_Shared"):
            continue

        match = re.match(r"PLE_Shared(\d+)_Task(\d+)_", folder)
        if not match:
            continue
        num_shared = int(match.group(1))
        num_task = int(match.group(2))

        report_path = os.path.join(base_dir, folder, "classification_reports.txt")
        if not os.path.isfile(report_path):
            continue

        with open(report_path, 'r') as file:
            lines = file.readlines()

        current_fold = None
        current_task = None
        in_final_section = False
        final_task = None

        for line in lines:
            # Final section
            if final_start_pattern.match(line):
                in_final_section = True
                current_fold = None
                continue

            if in_final_section:
                avg_task_match = avg_task_pattern.match(line)
                if avg_task_match:
                    final_task = avg_task_match.group(1)
                    continue

                acc_match = accuracy_line_pattern.match(line)
                if acc_match and final_task:
                    final_metrics_data[(num_shared, num_task)][final_task]['accuracy'] = float(acc_match.group(1))

                auc_match = roc_auc_pattern.search(line)
                if auc_match and final_task:
                    final_metrics_data[(num_shared, num_task)][final_task]['roc_auc'] = float(auc_match.group(1))

            else:
                fold_match = fold_pattern.match(line)
                if fold_match:
                    current_fold = int(fold_match.group(1))
                    continue

                task_match = task_pattern.match(line)
                if task_match:
                    current_task = task_match.group(1)
                    continue

                acc_match = accuracy_line_pattern.match(line)
                if acc_match and current_fold and current_task:
                    metrics_data[(num_shared, num_task)][current_fold][current_task]['accuracy'] = float(acc_match.group(1))

                auc_match = roc_auc_pattern.search(line)
                if auc_match and current_fold and current_task:
                    metrics_data[(num_shared, num_task)][current_fold][current_task]['roc_auc'] = float(auc_match.group(1))

    return metrics_data, final_metrics_data

def rank_configurations(final_metrics_data, metric='roc_auc', top_k=10):
    """
    Rank configurations by average metric (accuracy or ROC AUC).

    Parameters:
        final_metrics_data (dict): Output from load_ple_classification_metrics().
        metric (str): Either 'accuracy' or 'roc_auc'.
        top_k (int): Number of top configurations to return.

    Returns:
        List of tuples: [((shared, task), avg_metric), ...] sorted descending.
    """
    assert metric in ['accuracy', 'roc_auc'], "metric must be 'accuracy' or 'roc_auc'"

    config_scores = []
    for config, task_metrics in final_metrics_data.items():
        scores = []
        for task_name in ['1_missing', '2_trend', '3_drift']:
            task_metric = task_metrics.get(task_name, {}).get(metric)
            if task_metric is not None:
                scores.append(task_metric)

        if scores:
            avg_score = sum(scores) / len(scores) # Average of all tasks
            config_scores.append((config, avg_score))

    # Sort by average metric descending
    ranked = sorted(config_scores, key=lambda x: x[1], reverse=True)

    print(f"\n🔝 Top {top_k} configurations ranked by average {metric.upper()}:\n")
    for i, ((shared, task), score) in enumerate(ranked[:top_k], 1):
        print(f"{i:2d}. Shared: {shared}, Task: {task} → Avg {metric.upper()}: {score:.4f}")

    return ranked[:top_k]

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def rank_and_analyze_configs(final_metrics_data, metric='roc_auc', csv_path='config_ranking.csv'):
    assert metric in ['accuracy', 'roc_auc'], "metric must be 'accuracy' or 'roc_auc'"

    results = []

    for (shared, task), task_metrics in final_metrics_data.items():
        values = []
        for task_name in ['1_missing', '2_trend', '3_drift']:
            m = task_metrics.get(task_name, {}).get(metric)
            if m is not None:
                values.append(m)

        if values:
            avg = np.mean(values)
            std = np.std(values)
            results.append({
                'shared_experts': shared,
                'task_experts': task,
                f'avg_{metric}': avg,
                f'std_{metric}': std
            })

    # Convert to DataFrame
    df = pd.DataFrame(results)
    df.sort_values(by=f'avg_{metric}', ascending=False, inplace=True)

    # Save to CSV
    df.to_csv(csv_path, index=False)
    print(f"📁 Saved ranking and analysis to: {csv_path}")

    return df


def plot_heatmap(df, metric='roc_auc', title=None, save_path=None):
    """
    Plot heatmap for Shared × Task config with avg metric.
    """
    pivot = df.pivot(index='shared_experts', columns='task_experts', values=f'avg_{metric}')
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap='YlGnBu', cbar_kws={'label': f'Average {metric.upper()}'})
    plt.title(title or f'PLE Config Heatmap: Average {metric.upper()}')
    plt.xlabel("Task Experts")
    plt.ylabel("Shared Experts")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"🖼️ Saved heatmap to: {save_path}")
    plt.show()

def rank_with_joint_metric(final_metrics_data, csv_path='ple_config_joint_ranking.csv'):
    results = []

    for (shared, task), task_metrics in final_metrics_data.items():
        acc_values = []
        auc_values = []

        for task_name in ['1_missing', '2_trend', '3_drift']:
            acc = task_metrics.get(task_name, {}).get('accuracy')
            auc = task_metrics.get(task_name, {}).get('roc_auc')
            if acc is not None: acc_values.append(acc)
            if auc is not None: auc_values.append(auc)

        if acc_values and auc_values:
            avg_acc = np.mean(acc_values)
            std_acc = np.std(acc_values)
            avg_auc = np.mean(auc_values)
            std_auc = np.std(auc_values)
            joint_score = avg_auc + avg_acc - (std_auc + std_acc)  # penalize unstable metrics

            results.append({
                'shared_experts': shared,
                'task_experts': task,
                'avg_accuracy': avg_acc,
                'std_accuracy': std_acc,
                'avg_roc_auc': avg_auc,
                'std_roc_auc': std_auc,
                'joint_score': joint_score
            })

    df = pd.DataFrame(results)
    df.sort_values(by='joint_score', ascending=False, inplace=True)
    df.to_csv(csv_path, index=False)
    print(f"📁 Saved joint ranking results to: {csv_path}")
    return df

def plot_combined_heatmaps(df, save_path=None):
    acc_pivot = df.pivot(index='shared_experts', columns='task_experts', values='avg_accuracy')
    auc_pivot = df.pivot(index='shared_experts', columns='task_experts', values='avg_roc_auc')

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(acc_pivot, annot=True, fmt=".4f", cmap='Blues', ax=axes[0])
    axes[0].set_title("Average Accuracy")
    axes[0].set_xlabel("Task Experts")
    axes[0].set_ylabel("Shared Experts")

    sns.heatmap(auc_pivot, annot=True, fmt=".4f", cmap='Greens', ax=axes[1])
    axes[1].set_title("Average ROC AUC")
    axes[1].set_xlabel("Task Experts")
    axes[1].set_ylabel("")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"🖼️ Saved combined heatmap to: {save_path}")
    plt.show()

def plot_joint_score_heatmap(df, save_path=None):
    pivot = df.pivot(index='shared_experts', columns='task_experts', values='joint_score')
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap='magma', cbar_kws={'label': 'Joint Score'})
    plt.title("PLE Config Heatmap: Joint Score (Accuracy + AUC - Variance)")
    plt.xlabel("Task Experts")
    plt.ylabel("Shared Experts")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"🖼️ Saved joint score heatmap to: {save_path}")
    plt.show()

def rank_ple_by_fold_average(metrics_data, metric='roc_auc', csv_path='ple_fold_avg_ranking.csv'):
    """
    Rank configurations based on average per-fold metrics.

    Parameters:
        metrics_data (dict): Output from load_ple_classification_metrics()[0]
        metric (str): 'accuracy' or 'roc_auc'
        csv_path (str): Path to save the ranking as a CSV

    Returns:
        pd.DataFrame: Ranked dataframe
    """
    assert metric in ['accuracy', 'roc_auc'], "Metric must be 'accuracy' or 'roc_auc'"
    
    results = []
    
    for (shared, task), fold_data in metrics_data.items():
        task_scores = []
        for fold_id, task_metrics in fold_data.items():
            fold_scores = []
            for task_name in ['1_missing', '2_trend', '3_drift']:
                score = task_metrics.get(task_name, {}).get(metric)
                if score is not None:
                    fold_scores.append(score)
            if fold_scores:
                task_scores.append(np.mean(fold_scores))  # mean score across tasks for this fold

        if task_scores:
            avg_score = np.mean(task_scores)
            std_score = np.std(task_scores)
            results.append({
                'shared_experts': shared,
                'task_experts': task,
                f'avg_{metric}_per_fold': avg_score,
                f'std_{metric}_per_fold': std_score
            })

    df = pd.DataFrame(results)
    df.sort_values(by=f'avg_{metric}_per_fold', ascending=False, inplace=True)
    df.to_csv(csv_path, index=False)
    print(f"📁 Saved per-fold {metric.upper()} ranking to: {csv_path}")
    return df

def plot_per_fold_metric_heatmap(df, metric='roc_auc', save_path=None):
    col_name = f'avg_{metric}_per_fold'
    pivot = df.pivot(index='shared_experts', columns='task_experts', values=col_name)
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap='coolwarm', cbar_kws={'label': f'Per-Fold Avg {metric.upper()}'})
    plt.title(f'PLE Config Heatmap: Per-Fold Avg {metric.upper()}')
    plt.xlabel("Task Experts")
    plt.ylabel("Shared Experts")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"🖼️ Saved heatmap to: {save_path}")
    plt.show()

def rank_best_fold_across_configs(metrics_data, metric='roc_auc', csv_path='best_folds_across_configs.csv'):
    """
    Rank the best-performing folds across all parameter configs.

    Parameters:
        metrics_data (dict): Output from load_ple_classification_metrics()[0]
        metric (str): 'accuracy' or 'roc_auc'
        csv_path (str): CSV output path

    Returns:
        pd.DataFrame: DataFrame with best folds ranked by per-fold average metric
    """
    assert metric in ['accuracy', 'roc_auc'], "Metric must be 'accuracy' or 'roc_auc'"

    fold_results = []

    for (shared, task), fold_data in metrics_data.items():
        for fold_id, task_metrics in fold_data.items():
            scores = []
            for task_name in ['1_missing', '2_trend', '3_drift']:
                value = task_metrics.get(task_name, {}).get(metric)
                if value is not None:
                    scores.append(value)

            if scores:
                avg_metric = sum(scores) / len(scores)
                fold_results.append({
                    'shared_experts': shared,
                    'task_experts': task,
                    'fold_id': fold_id,
                    f'avg_{metric}': avg_metric
                })

    df = pd.DataFrame(fold_results)
    df.sort_values(by=f'avg_{metric}', ascending=False, inplace=True)
    df.to_csv(csv_path, index=False)
    print(f"📁 Saved best fold ranking to: {csv_path}")

    # 🔍 Print best fold info
    best_row = df.iloc[0]
    print("\n🏆 Best Fold Across All Configurations:")
    print(f"  Shared Experts   : {int(best_row['shared_experts'])}")
    print(f"  Task Experts     : {int(best_row['task_experts'])}")
    print(f"  Fold ID          : {int(best_row['fold_id'])}")
    print(f"  Avg {metric.upper()}: {best_row[f'avg_{metric}']:.4f}")

    return df

def rank_best_fold_by_joint_score(metrics_data, csv_path='best_folds_by_joint_score.csv'):
    """
    Rank the best fold across all (shared, task) configurations using joint score:
    joint_score = avg_roc_auc + avg_accuracy - (std_roc_auc + std_accuracy)

    Parameters:
        metrics_data (dict): Output from load_ple_classification_metrics()[0]
        csv_path (str): CSV output path

    Returns:
        pd.DataFrame: Ranked fold DataFrame
    """
    fold_results = []

    for (shared, task), fold_data in metrics_data.items():
        for fold_id, task_metrics in fold_data.items():
            acc_scores = []
            auc_scores = []

            for task_name in ['1_missing', '2_trend', '3_drift']:
                acc = task_metrics.get(task_name, {}).get('accuracy')
                auc = task_metrics.get(task_name, {}).get('roc_auc')
                if acc is not None:
                    acc_scores.append(acc)
                if auc is not None:
                    auc_scores.append(auc)

            if acc_scores and auc_scores:
                avg_acc = np.mean(acc_scores)
                std_acc = np.std(acc_scores)
                avg_auc = np.mean(auc_scores)
                std_auc = np.std(auc_scores)
                joint_score = avg_auc + avg_acc - (std_auc + std_acc)

                fold_results.append({
                    'shared_experts': shared,
                    'task_experts': task,
                    'fold_id': fold_id,
                    'avg_accuracy': avg_acc,
                    'std_accuracy': std_acc,
                    'avg_roc_auc': avg_auc,
                    'std_roc_auc': std_auc,
                    'joint_score': joint_score
                })

    df = pd.DataFrame(fold_results)
    df.sort_values(by='joint_score', ascending=False, inplace=True)
    df.to_csv(csv_path, index=False)
    print(f"📁 Saved best joint-score fold ranking to: {csv_path}")

    # 🏆 Print top fold
    best_row = df.iloc[0]
    print("\n🏆 Best Fold Across All Configurations (Joint Score):")
    print(f"  Shared Experts   : {int(best_row['shared_experts'])}")
    print(f"  Task Experts     : {int(best_row['task_experts'])}")
    print(f"  Fold ID          : {int(best_row['fold_id'])}")
    print(f"  Avg Accuracy     : {best_row['avg_accuracy']:.4f}")
    print(f"  Avg ROC AUC      : {best_row['avg_roc_auc']:.4f}")
    print(f"  Joint Score      : {best_row['joint_score']:.4f}")

    return df

def rank_configs_by_fold(metrics_data, fold_id=1, save_dir='fold_ranking_outputs'):
    """
    这里对应的是论文中的2.2. Expert configuration optimization strategy
    Rank all (shared, task) configs for a specified fold by:
      - avg_accuracy
      - avg_roc_auc
      - joint_score = avg_acc + avg_auc - (std_acc + std_auc)

    Saves 3 CSVs and returns 3 DataFrames.

    Parameters:
        metrics_data (dict): Output from load_ple_classification_metrics()[0]
        fold_id (int): Fold index to rank
        save_dir (str): Directory to save result CSVs

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            (rank_by_acc_df, rank_by_auc_df, rank_by_joint_df)
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    records = []

    for (shared, task), fold_data in metrics_data.items():
        if fold_id not in fold_data:
            continue

        task_metrics = fold_data[fold_id]
        acc_scores = []
        auc_scores = []

        for task_name in ['1_missing', '2_trend', '3_drift']:
            acc = task_metrics.get(task_name, {}).get('accuracy')
            auc = task_metrics.get(task_name, {}).get('roc_auc')
            if acc is not None:
                acc_scores.append(acc)
            if auc is not None:
                auc_scores.append(auc)

        if acc_scores and auc_scores:
            avg_acc = np.mean(acc_scores)
            std_acc = np.std(acc_scores)
            avg_auc = np.mean(auc_scores)
            std_auc = np.std(auc_scores)
            joint_score = avg_acc + avg_auc - (std_acc + std_auc)

            records.append({
                'shared_experts': shared,
                'task_experts': task,
                'fold_id': fold_id,
                'avg_accuracy': avg_acc,
                'std_accuracy': std_acc,
                'avg_roc_auc': avg_auc,
                'std_roc_auc': std_auc,
                'joint_score': joint_score
            })

    df_all = pd.DataFrame(records)

    df_acc = df_all.sort_values(by='avg_accuracy', ascending=False)
    df_auc = df_all.sort_values(by='avg_roc_auc', ascending=False)
    df_joint = df_all.sort_values(by='joint_score', ascending=False)

    df_acc.to_csv(os.path.join(save_dir, f'fold{fold_id}_rank_by_accuracy.csv'), index=False)
    df_auc.to_csv(os.path.join(save_dir, f'fold{fold_id}_rank_by_roc_auc.csv'), index=False)
    df_joint.to_csv(os.path.join(save_dir, f'fold{fold_id}_rank_by_joint_score.csv'), index=False)

    print(f"\n📁 Rankings for Fold {fold_id} saved in '{save_dir}'")
    print(f"🏆 Top Config by Joint Score:")
    best = df_joint.iloc[0]
    print(f"  Shared Experts : {int(best['shared_experts'])}")
    print(f"  Task Experts   : {int(best['task_experts'])}")
    print(f"  Avg Accuracy   : {best['avg_accuracy']:.4f}")
    print(f"  Avg ROC AUC    : {best['avg_roc_auc']:.4f}")
    print(f"  Joint Score    : {best['joint_score']:.4f}")

    return df_acc, df_auc, df_joint

import matplotlib.pyplot as plt
import seaborn as sns

def plot_config_metric_comparisons(df, title_prefix='', save_path=None):
    """
    Plot bar charts comparing average accuracy, ROC AUC, and joint score per (shared, task) config.

    Parameters:
        df (pd.DataFrame): DataFrame containing columns:
            ['shared_experts', 'task_experts', 'avg_accuracy', 'avg_roc_auc', 'joint_score']
        title_prefix (str): Optional string to prefix plot titles.
        save_path (str): If given, saves the plot to this path.
    """
    # Create a new column for labels
    df['config'] = df.apply(lambda row: f"S{row['shared_experts']}_T{row['task_experts']}", axis=1)

    fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    sns.barplot(x='config', y='avg_accuracy', data=df, ax=axs[0], palette='Blues_d')
    axs[0].set_title(f'{title_prefix} Average Accuracy per Config')
    axs[0].set_ylabel("Accuracy")
    axs[0].tick_params(axis='x', rotation=45)

    sns.barplot(x='config', y='avg_roc_auc', data=df, ax=axs[1], palette='Greens_d')
    axs[1].set_title(f'{title_prefix} Average ROC AUC per Config')
    axs[1].set_ylabel("ROC AUC")
    axs[1].tick_params(axis='x', rotation=45)

    sns.barplot(x='config', y='joint_score', data=df, ax=axs[2], palette='Purples_d')
    axs[2].set_title(f'{title_prefix} Joint Score per Config\n(acc + auc - stds)')
    axs[2].set_ylabel("Joint Score")
    axs[2].set_xlabel("PLE Configuration (S = Shared, T = Task)")
    axs[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"🖼️ Plot saved to {save_path}")
    
    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_config_heatmaps(df, title_prefix='', save_path=None):
    """
    Plot heatmaps for avg_accuracy, avg_roc_auc, and joint_score over (shared, task) config grid.

    Parameters:
        df (pd.DataFrame): DataFrame with columns:
            ['shared_experts', 'task_experts', 'avg_accuracy', 'avg_roc_auc', 'joint_score']
        title_prefix (str): Prefix for plot titles
        save_path (str): If set, saves the figure to this path
    """
    # Pivot tables for heatmap
    acc_matrix = df.pivot(index='shared_experts', columns='task_experts', values='avg_accuracy')
    auc_matrix = df.pivot(index='shared_experts', columns='task_experts', values='avg_roc_auc')
    std_acc_matrix = df.pivot(index='shared_experts', columns='task_experts', values='std_accuracy')
    std_auc_matrix = df.pivot(index='shared_experts', columns='task_experts', values='std_roc_auc')
    joint_matrix = df.pivot(index='shared_experts', columns='task_experts', values='joint_score')

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))

    sns.heatmap(acc_matrix, annot=False, fmt=".4f", cmap="Blues", ax=axes[0][0])
    axes[0][0].set_title(f'{title_prefix} Accuracy')
    axes[0][0].set_xlabel("Task Experts")
    axes[0][0].set_ylabel("Shared Experts")

    sns.heatmap(auc_matrix, annot=False, fmt=".4f", cmap="Greens", ax=axes[0][1])
    axes[0][1].set_title(f'{title_prefix} ROC AUC')
    axes[0][1].set_xlabel("Task Experts")
    axes[0][1].set_ylabel("")

    sns.heatmap(joint_matrix, annot=False, fmt=".4f", cmap="Purples", ax=axes[0][2])
    axes[0][2].set_title(f'{title_prefix} Joint Score')
    axes[0][2].set_xlabel("Task Experts")
    axes[0][2].set_ylabel("")

    sns.heatmap(std_acc_matrix, annot=False, fmt=".4f", cmap="Blues", ax=axes[1][0])
    axes[1][0].set_title(f'{title_prefix} std of Accuracy')
    axes[1][0].set_xlabel("Task Experts")
    axes[1][0].set_ylabel("Shared Experts")

    sns.heatmap(std_auc_matrix, annot=False, fmt=".4f", cmap="Greens", ax=axes[1][1])
    axes[1][1].set_title(f'{title_prefix} std of ROC AUC')
    axes[1][1].set_xlabel("Task Experts")
    axes[1][1].set_ylabel("")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"🖼️ Heatmaps saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    base_dir = "saved_models/para_configuration_20250626"
    metrics_data, final_metrics_data = load_classification_metrics(base_dir)

    # Example Output
    for (shared, task), folds in sorted(metrics_data.items()):
        print(f"\n📊 Config: Shared Experts = {shared}, Task Experts = {task}")
        for fold_id, task_metrics in sorted(folds.items()):
            print(f"  🔁 Fold {fold_id}:")
            for task_name, metrics in task_metrics.items():
                acc = metrics.get('accuracy', 'N/A')
                auc = metrics.get('roc_auc', 'N/A')
                print(f"    {task_name}: Accuracy = {acc:.4f}, ROC AUC = {auc:.4f}")

        # Final average metrics
        print(f"  📌 Final Average over Folds:")
        for task_name, metrics in final_metrics_data[(shared, task)].items():
            acc = metrics.get('accuracy', 'N/A')
            auc = metrics.get('roc_auc', 'N/A')
            print(f"    {task_name}: Avg Accuracy = {acc:.4f}, Avg ROC AUC = {auc:.4f}")


    # # Rank by ROC AUC
    # rank_configurations(final_metrics_data, metric='roc_auc', top_k=10)

    # # Rank by Accuracy
    # rank_configurations(final_metrics_data, metric='accuracy', top_k=10)

    # # Step 2: Rank and Save
    # df_results = rank_and_analyze_configs(
    #     final_metrics_data, 
    #     metric='roc_auc', 
    #     csv_path=base_dir+'/'+'config_ranking_roc_auc.csv'
    # )

    # # Step 3: Plot Heatmap
    # plot_heatmap(
    #     df_results, 
    #     metric='roc_auc', 
    #     title='Config Heatmap (ROC AUC)', 
    #     save_path=base_dir+'/'+'heatmap_roc_auc.png'
    # )

    # # Step 4: Print Stability (std deviation)
    # print("\n🔬 Top Stable Configurations by ROC AUC Std Dev:")
    # stable_df = df_results.sort_values(by='std_roc_auc')
    # print(stable_df[['shared_experts', 'task_experts', 'std_roc_auc']].head(10))

    # # Step 2: Rank and export
    # df_joint = rank_with_joint_metric(final_metrics_data, csv_path=base_dir+'/'+'joint_ranking.csv')

    # # Step 3: Plot heatmaps
    # plot_combined_heatmaps(df_joint, save_path=base_dir+'/'+'combined_accuracy_auc_heatmap.png')
    # plot_joint_score_heatmap(df_joint, save_path=base_dir+'/'+'joint_score_heatmap.png')


    # # Rank based on average ROC AUC across folds
    # df_auc = rank_ple_by_fold_average(metrics_data, metric='roc_auc', csv_path=base_dir+'/'+'ple_auc_per_fold.csv')

    # # Rank based on average Accuracy across folds
    # df_acc = rank_ple_by_fold_average(metrics_data, metric='accuracy', csv_path=base_dir+'/'+'ple_acc_per_fold.csv')

    # plot_per_fold_metric_heatmap(df_auc, metric='roc_auc', save_path=base_dir+'/'+'per_fold_auc_heatmap.png')


    # # Rank folds by ROC AUC
    # df_best_folds_auc = rank_best_fold_across_configs(metrics_data, metric='roc_auc', csv_path=base_dir+'/'+'best_folds_by_auc.csv')

    # # Rank folds by Accuracy
    # df_best_folds_acc = rank_best_fold_across_configs(metrics_data, metric='accuracy', csv_path=base_dir+'/'+'best_folds_by_acc.csv')


    # df_best_joint = rank_best_fold_by_joint_score(metrics_data, csv_path=base_dir+'/'+'best_folds_joint_score.csv')

    fold_id = 4
    df_acc, df_auc, df_joint = rank_configs_by_fold(metrics_data, fold_id=4, save_dir=base_dir+'/'+'fold_ranking_outputs')
    # For fold 3, with the joint-scored rankings
    # plot_config_metric_comparisons(df_joint, title_prefix=f'Fold {fold_id}', save_path=base_dir+'/'+f'fold{fold_id}_config_comparison.png')
    # Example: Fold 3 heatmaps
    plot_config_heatmaps(df_joint, title_prefix=f'Fold {fold_id}', save_path=base_dir+'/'+f'fold{fold_id}_config_heatmaps.png')
