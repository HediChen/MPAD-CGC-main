'''
Multi-Run Configuration Optimization Experiments
================================================================================
This script runs complete multi-run experiments for MPAD-CGC model configuration
optimization. It executes 5 independent runs with different random seeds for
all expert configurations (shared_experts × task_experts combinations), and
generates the multi_run_results.csv file required for statistical analysis.

Features:
  - Runs experiments 5 times with different random seeds
  - Tests all configurations (1-10 shared experts × 1-10 task experts)
  - Tracks Joint Score, accuracy, and AUC for each run
  - Supports checkpoint/resume functionality
  - Generates comprehensive results CSV
  - Creates summary statistics and visualization
'''

import random
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
from collections import defaultdict
import pickle
import json

# Import from existing modules
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pytorchModels.ple_Inception_Features_analysis_train import PLE
from preprocessing_addFeatures import data_preprocessing
from unit.summary import sum_parameters_by_layer
from sklearn.model_selection import StratifiedKFold
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class BLFocalLoss(nn.Module):
    '''Batch-level Focal Loss'''
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(BLFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, input, target, sigma_sq, key):
        ce_loss = nn.BCELoss()(input, target)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def setup_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def compute_joint_score(y_true_dict, y_pred_dict, y_prod_dict):
    """
    Compute Joint Score across all tasks.
    
    Joint Score = mean_accuracy + mean_auc - (std_accuracy + std_auc)
    
    where:
      - mean_accuracy: Average accuracy across tasks
      - mean_auc: Average AUC across tasks
      - std_accuracy: Std dev of accuracy across tasks
      - std_auc: Std dev of AUC across tasks
    """
    accuracies = []
    aucs = []
    
    for task in ['1_missing', '2_trend', '3_drift']:
        # Compute accuracy
        y_true = np.array(y_true_dict[task])
        y_pred = np.array(y_pred_dict[task])
        acc = np.mean(y_true == y_pred)
        accuracies.append(acc)
        
        # Compute AUC
        y_prob = np.array(y_prod_dict[task])
        try:
            auc = roc_auc_score(y_true, y_prob)
        except:
            auc = 0.5  # Default if only one class present
        aucs.append(auc)
    
    mean_acc = np.mean(accuracies)
    mean_auc = np.mean(aucs)
    std_acc = np.std(accuracies)
    std_auc = np.std(aucs)
    
    joint_score = mean_auc + mean_acc - (std_auc + std_acc)
    
    return {
        'joint_score': joint_score,
        'avg_accuracy': mean_acc,
        'avg_auc': mean_auc,
        'std_accuracy': std_acc,
        'std_auc': std_auc,
        'task_accuracies': {task: acc for task, acc in zip(['1_missing', '2_trend', '3_drift'], accuracies)},
        'task_aucs': {task: auc for task, auc in zip(['1_missing', '2_trend', '3_drift'], aucs)}
    }


def train_single_fold(x_train, y_train, x_val, y_val, num_shared_experts, num_task_experts,
                      num_epochs=100, device='cuda', seed=4):
    """
    Train model for a single fold and return metrics.
    
    Returns:
        dict: Contains y_true_dict, y_pred_dict, y_prod_dict for all tasks
    """
    setup_seed(seed)
    
    # Model parameters
    num_classes = 3
    dict_classes = {0: '1_missing', 1: '2_trend', 2: '3_drift'}
    batch_size = 128
    learning_rate = 0.001
    
    # Convert to tensors
    x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    x_val = torch.tensor(x_val, dtype=torch.float32).unsqueeze(1)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    
    # Convert to one-hot
    y_train = torch.eye(2)[y_train.long(), :]
    y_val = torch.eye(2)[y_val.long(), :]
    
    # Get number of features
    num_features = x_train[:, :, 1008:].shape[-1]
    
    # Build model
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
        device=device
    )
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Define loss and optimizer
    criterion = BLFocalLoss(reduction='mean')
    log_vars = nn.Parameter(torch.zeros(num_classes, requires_grad=True, device=device))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer_uncertainty = optim.Adam([log_vars], lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    
    # Data loaders
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            optimizer_uncertainty.zero_grad()
            
            outputs = model(inputs)
            loss_list = torch.stack([criterion(outputs[dict_classes[key]], labels[:, key], log_vars, key) 
                                     for key in dict_classes])
            train_loss = loss_list.sum() + model.l2_reg_loss
            
            train_loss.backward()
            optimizer.step()
            optimizer_uncertainty.step()
        
        scheduler.step()
    
    # Validation phase
    model.eval()
    y_true_fold = {task: [] for task in dict_classes.values()}
    y_pred_fold = {task: [] for task in dict_classes.values()}
    y_prod_fold = {task: [] for task in dict_classes.values()}
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            for key in dict_classes:
                probs = torch.softmax(outputs[dict_classes[key]], dim=1)[:, 1].cpu().numpy()
                preds = torch.argmax(outputs[dict_classes[key]], dim=1).cpu().numpy()
                true = torch.argmax(labels[:, key], dim=1).cpu().numpy()
                
                y_prod_fold[dict_classes[key]].extend(probs)
                y_pred_fold[dict_classes[key]].extend(preds)
                y_true_fold[dict_classes[key]].extend(true)
    
    return y_true_fold, y_pred_fold, y_prod_fold


class MultiRunExperimentRunner:
    """
    Runs multi-run configuration optimization experiments.
    """
    
    def __init__(self, num_runs=5, num_shared_experts_range=(1, 11), num_task_experts_range=(1, 11),
                 output_dir='multirun_results', checkpoint_dir='multirun_checkpoints'):
        """
        Initialize experiment runner.
        
        Parameters:
            num_runs: Number of independent runs (with different seeds)
            num_shared_experts_range: Range for shared experts (start, end)
            num_task_experts_range: Range for task experts (start, end)
            output_dir: Directory to save results
            checkpoint_dir: Directory to save checkpoints
        """
        self.num_runs = num_runs
        self.shared_experts_range = range(num_shared_experts_range[0], num_shared_experts_range[1])
        self.task_experts_range = range(num_task_experts_range[0], num_task_experts_range[1])
        self.output_dir = output_dir
        self.checkpoint_dir = checkpoint_dir
        self.results = []
        self.seeds = [42 + i for i in range(num_runs)]  # Different seeds for each run
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def load_data(self):
        """Load and prepare data."""
        print("\n" + "="*80)
        print("Loading data...")
        print("="*80)
        
        _, _, x_all, y_all, _ = data_preprocessing(balanced=False, platform='pytorch', 
                                                     normal_class=False, method='test')
        
        # Shuffle data
        np.random.seed(8)
        indices = np.arange(x_all.shape[0])
        np.random.shuffle(indices)
        x_all = x_all[indices]
        y_all = y_all[indices]
        
        # Split into train+val and test
        x_train_val = x_all[:1436, :]
        y_train_val = y_all[:1436, :]
        
        print(f"✓ Train+Val size: {x_train_val.shape[0]}")
        print(f"✓ Feature dimension: {x_train_val.shape[1]}")
        
        return x_train_val, y_train_val
    
    def run_experiments(self, fold_id=4):
        """
        Run all experiments (all configs × all runs).
        
        Parameters:
            fold_id: Which fold to use (default: 4, as in original script)
        """
        print("\n" + "="*80)
        print(f"MULTI-RUN CONFIGURATION OPTIMIZATION EXPERIMENTS")
        print(f"Total Runs: {self.num_runs}")
        print(f"Shared Experts Range: {min(self.shared_experts_range)}-{max(self.shared_experts_range)}")
        print(f"Task Experts Range: {min(self.task_experts_range)}-{max(self.task_experts_range)}")
        print(f"Total Configurations: {len(self.shared_experts_range) * len(self.task_experts_range)}")
        print("="*80 + "\n")
        
        # Load data once
        x_train_val, y_train_val = self.load_data()
        
        # Setup stratified k-fold
        stratify_labels = y_train_val[:, 0]
        skf = StratifiedKFold(n_splits=4, shuffle=False)
        
        # Track progress
        total_configs = len(self.shared_experts_range) * len(self.task_experts_range)
        config_count = 0
        
        # Iterate through all configurations
        for num_shared in self.shared_experts_range:
            for num_task in self.task_experts_range:
                config_count += 1
                config_tuple = (num_shared, num_task)
                
                print(f"\n[{config_count}/{total_configs}] Config: Shared={num_shared}, Task={num_task}")
                print("-" * 80)
                
                # Check if already completed (for resume functionality)
                checkpoint_file = os.path.join(self.checkpoint_dir, 
                                               f'config_{num_shared}_{num_task}_results.pkl')
                if os.path.exists(checkpoint_file):
                    print(f"  ⚠ Found checkpoint, resuming from this configuration...")
                    with open(checkpoint_file, 'rb') as f:
                        config_results = pickle.load(f)
                    self.results.extend(config_results)
                    continue
                
                config_results = []
                
                # Run multiple times with different seeds
                for run_id in range(1, self.num_runs + 1):
                    seed = self.seeds[run_id - 1]
                    print(f"  Run {run_id}/{self.num_runs} (seed={seed})...", end=' ', flush=True)
                    
                    try:
                        # Extract fold 4 data
                        for fold, (train_idx, val_idx) in enumerate(skf.split(x_train_val, stratify_labels), 1):
                            if fold == fold_id:
                                x_train_fold = x_train_val[train_idx]
                                y_train_fold = y_train_val[train_idx]
                                x_val_fold = x_train_val[val_idx]
                                y_val_fold = y_train_val[val_idx]
                                break
                        
                        # Train and get results
                        y_true_dict, y_pred_dict, y_prod_dict = train_single_fold(
                            x_train_fold, y_train_fold, x_val_fold, y_val_fold,
                            num_shared_experts=num_shared,
                            num_task_experts=num_task,
                            num_epochs=100,
                            device='cuda',
                            seed=seed
                        )
                        
                        # Compute Joint Score
                        metrics = compute_joint_score(y_true_dict, y_pred_dict, y_prod_dict)
                        
                        # Store result
                        result = {
                            'shared_experts': num_shared,
                            'task_experts': num_task,
                            'run_id': run_id,
                            'seed': seed,
                            'joint_score': metrics['joint_score'],
                            'avg_accuracy': metrics['avg_accuracy'],
                            'avg_auc': metrics['avg_auc'],
                            'std_accuracy': metrics['std_accuracy'],
                            'std_auc': metrics['std_auc'],
                            'task_accuracies': json.dumps(metrics['task_accuracies']),
                            'task_aucs': json.dumps(metrics['task_aucs'])
                        }
                        
                        config_results.append(result)
                        self.results.append(result)
                        
                        print(f"✓ JS={metrics['joint_score']:.6f}")
                        
                    except Exception as e:
                        print(f"✗ Error: {str(e)[:50]}")
                        continue
                
                # Save checkpoint for this configuration
                if config_results:
                    with open(checkpoint_file, 'wb') as f:
                        pickle.dump(config_results, f)
                    print(f"  ✓ Checkpoint saved")
        
        print("\n" + "="*80)
        print(f"✓ Experiments completed!")
        print(f"✓ Total results collected: {len(self.results)}")
        print("="*80 + "\n")
    
    def save_results_csv(self, filename='multi_run_results.csv'):
        """Save results to CSV file."""
        output_path = os.path.join(self.output_dir, filename)
        
        df_results = pd.DataFrame(self.results)
        df_results.to_csv(output_path, index=False)
        
        print(f"✓ Results saved to: {output_path}")
        print(f"  Shape: {df_results.shape}")
        print(f"\nFirst few rows:")
        print(df_results.head(10))
        
        return output_path, df_results
    
    def generate_summary_statistics(self, df_results):
        """Generate and save summary statistics."""
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80 + "\n")
        
        # Statistics by configuration
        config_stats = df_results.groupby(['shared_experts', 'task_experts']).agg({
            'joint_score': ['mean', 'std', 'min', 'max'],
            'avg_accuracy': ['mean', 'std'],
            'avg_auc': ['mean', 'std']
        }).round(6)
        
        config_stats_path = os.path.join(self.output_dir, 'config_summary_statistics.csv')
        config_stats.to_csv(config_stats_path)
        print(f"✓ Config statistics saved to: {config_stats_path}")
        
        # Top 10 configurations
        top_configs = df_results.groupby(['shared_experts', 'task_experts'])['joint_score'].mean().sort_values(ascending=False).head(10)
        print("\nTop 10 Configurations by Mean Joint Score:")
        for i, (config, score) in enumerate(top_configs.items(), 1):
            print(f"  {i:2d}. S{config[0]}_T{config[1]}: {score:.6f}")
        
        return config_stats
    
    def plot_results(self, df_results):
        """Create visualization plots."""
        print("\n" + "="*80)
        print("GENERATING PLOTS")
        print("="*80 + "\n")
        
        # Plot 1: Heatmap of mean Joint Score
        pivot_js = df_results.groupby(['shared_experts', 'task_experts'])['joint_score'].mean().unstack()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_js, annot=True, fmt='.4f', cmap='YlGnBu', cbar_kws={'label': 'Mean Joint Score'})
        plt.title('Mean Joint Score across Configurations (All Runs Combined)')
        plt.xlabel('Task Experts')
        plt.ylabel('Shared Experts')
        plt.tight_layout()
        plot_path1 = os.path.join(self.output_dir, 'heatmap_mean_joint_score.png')
        plt.savefig(plot_path1, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Heatmap saved to: {plot_path1}")
        
        # Plot 2: Heatmap of std Joint Score
        pivot_std = df_results.groupby(['shared_experts', 'task_experts'])['joint_score'].std().unstack()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_std, annot=True, fmt='.4f', cmap='RdYlGn_r', cbar_kws={'label': 'Std Joint Score'})
        plt.title('Standard Deviation of Joint Score (Stability Across Runs)')
        plt.xlabel('Task Experts')
        plt.ylabel('Shared Experts')
        plt.tight_layout()
        plot_path2 = os.path.join(self.output_dir, 'heatmap_std_joint_score.png')
        plt.savefig(plot_path2, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Std heatmap saved to: {plot_path2}")
        
        # Plot 3: Joint Score distribution for top 5 configurations
        top_5_configs = df_results.groupby(['shared_experts', 'task_experts'])['joint_score'].mean().sort_values(ascending=False).head(5)
        
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes = axes.flatten()
        
        for i, (config, _) in enumerate(top_5_configs.items()):
            config_data = df_results[(df_results['shared_experts'] == config[0]) & 
                                     (df_results['task_experts'] == config[1])]['joint_score']
            
            axes[i].hist(config_data, bins=self.num_runs, alpha=0.7, color='skyblue', edgecolor='black')
            axes[i].set_title(f'S{config[0]}_T{config[1]}\nMean: {config_data.mean():.6f}')
            axes[i].set_xlabel('Joint Score')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
        
        # Hide the 6th subplot
        axes[5].axis('off')
        
        plt.tight_layout()
        plot_path3 = os.path.join(self.output_dir, 'top5_joint_score_distributions.png')
        plt.savefig(plot_path3, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Distribution plot saved to: {plot_path3}")
        
        # Plot 4: Box plot of Joint Score for top 10 configurations
        top_10_configs = df_results.groupby(['shared_experts', 'task_experts'])['joint_score'].mean().sort_values(ascending=False).head(10)
        
        plot_data = []
        config_labels = []
        
        for config, _ in top_10_configs.items():
            config_data = df_results[(df_results['shared_experts'] == config[0]) & 
                                     (df_results['task_experts'] == config[1])]['joint_score'].values
            plot_data.append(config_data)
            config_labels.append(f"S{config[0]}_T{config[1]}")
        
        plt.figure(figsize=(12, 6))
        bp = plt.boxplot(plot_data, labels=config_labels, patch_artist=True)
        
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        
        plt.ylabel('Joint Score')
        plt.title('Joint Score Distribution for Top 10 Configurations')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plot_path4 = os.path.join(self.output_dir, 'top10_joint_score_boxplot.png')
        plt.savefig(plot_path4, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Boxplot saved to: {plot_path4}")
    
    def run_full_pipeline(self, fold_id=4):
        """Run complete pipeline: experiments → CSV → statistics → plots."""
        start_time = time.time()
        
        # Run experiments
        self.run_experiments(fold_id=fold_id)
        
        # Save results
        csv_path, df_results = self.save_results_csv()
        
        # Generate statistics
        self.generate_summary_statistics(df_results)
        
        # Create plots
        self.plot_results(df_results)
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETED")
        print("="*80)
        print(f"✓ Total time: {elapsed_time/3600:.2f} hours")
        print(f"✓ Results CSV: {csv_path}")
        print(f"✓ Output directory: {self.output_dir}")
        print("="*80 + "\n")
        
        return csv_path, df_results


# Main execution
if __name__ == '__main__':
    print("\n" + "="*80)
    print("MPAD-CGC MULTI-RUN CONFIGURATION OPTIMIZATION")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Initialize runner
    runner = MultiRunExperimentRunner(
        num_runs=5,  # Run 5 times
        num_shared_experts_range=(1, 11),  # 1-10 shared experts
        num_task_experts_range=(1, 11),    # 1-10 task experts
        output_dir='multirun_results',
        checkpoint_dir='multirun_checkpoints'
    )
    
    # Run full pipeline
    csv_path, df_results = runner.run_full_pipeline(fold_id=4)
    
    print(f"\n✅ Multi-run results saved to: {csv_path}")
    print(f"✅ Now run: python statistical_significance_test.py")
    print(f"   Or use: analyzer = run_comprehensive_statistical_analysis('{csv_path}')\n")