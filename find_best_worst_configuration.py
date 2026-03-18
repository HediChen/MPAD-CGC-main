'''
Best and Worst Configuration Analysis
================================================================================
This script identifies the best and worst expert configurations based on:
1. Top 10 configurations by mean Joint Score
2. Best configuration: Among top 10, select most stable (narrowest CI/least variance)
   then choose one with least expert quantity (shared + task experts)
3. Worst configuration: Among top 10, select least stable (broadest CI/largest variance)
   then choose one with most expert quantity (shared + task experts)

This multi-criteria approach ensures we find:
- Configurations with good performance AND stability
- Configurations with minimal complexity (parameter efficiency)
- Configurations with poor performance AND instability (to avoid)
'''

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class ConfigurationAnalyzer:
    """Analyzer for finding best and worst configurations from multi-run results."""
    
    def __init__(self, csv_file):
        """
        Initialize analyzer and load data.
        
        Parameters:
            csv_file: Path to multi-run results CSV
        """
        self.csv_file = csv_file
        self.df_raw = None
        self.config_stats = None
        self.top_10_configs = None
        self.best_config = None
        self.worst_config = None
        self.results_dict = {}
        self.confidence_level = 0.95
        
    def load_data(self):
        """Load and validate CSV data."""
        print("Loading CSV data...")
        self.df_raw = pd.read_csv(self.csv_file)
        
        # Build results_dict for compatibility with distribution plotting
        for (shared, task), group in self.df_raw.groupby(['shared_experts', 'task_experts']):
            config_tuple = (int(shared), int(task))
            self.results_dict[config_tuple] = {
                run_id: {'joint_score': js} 
                for run_id, js in zip(group['run_id'], group['joint_score'])
            }
        
        print(f"✓ Loaded {len(self.df_raw)} rows")
        print(f"✓ Columns: {list(self.df_raw.columns)}")
        print(f"✓ Unique configurations: {self.df_raw.groupby(['shared_experts', 'task_experts']).ngroups}")
        
        return self.df_raw
    
    def _compute_ci_t(self, data, confidence_level):
        """Compute confidence interval using t-distribution."""
        n = len(data)
        mean = np.mean(data)
        std_err = stats.sem(data)
        dof = n - 1
        t_critical = stats.t.ppf((1 + confidence_level) / 2, dof)
        margin = t_critical * std_err
        return mean - margin, mean + margin
    
    def compute_configuration_statistics(self, ci_method='t_distribution'):
        """
        Compute statistics for each configuration.
        
        Parameters:
            ci_method: Method for CI calculation
                      't_distribution': Uses t-distribution (recommended)
                      'bootstrap': Uses bootstrap percentiles
        
        Returns:
            pd.DataFrame with statistics for each configuration
        """
        print(f"\nComputing configuration statistics (CI method: {ci_method})...")
        
        stats_list = []
        
        for (shared, task), group in self.df_raw.groupby(['shared_experts', 'task_experts']):
            config_tuple = (int(shared), int(task))
            n_runs = len(group)
            
            # Joint Score statistics
            js_values = group['joint_score'].values
            js_mean = np.mean(js_values)
            js_std = np.std(js_values, ddof=1)
            js_median = np.median(js_values)
            js_min = np.min(js_values)
            js_max = np.max(js_values)
            js_range = js_max - js_min
            js_var = np.var(js_values, ddof=1)
            js_cv = js_std / js_mean if js_mean > 0 else 0  # Coefficient of variation
            
            # Compute 95% CI
            if ci_method == 't_distribution':
                js_se = js_std / np.sqrt(n_runs)
                dof = n_runs - 1
                t_crit = stats.t.ppf(0.975, dof)  # 95% CI
                js_ci_lower = js_mean - t_crit * js_se
                js_ci_upper = js_mean + t_crit * js_se
            else:  # bootstrap
                bootstrap_means = []
                for _ in range(10000):
                    sample = np.random.choice(js_values, size=n_runs, replace=True)
                    bootstrap_means.append(np.mean(sample))
                js_ci_lower = np.percentile(bootstrap_means, 2.5)
                js_ci_upper = np.percentile(bootstrap_means, 97.5)
            
            js_ci_width = js_ci_upper - js_ci_lower
            
            # Accuracy statistics
            acc_values = group['avg_accuracy'].values
            acc_mean = np.mean(acc_values)
            acc_std = np.std(acc_values, ddof=1)
            
            # AUC statistics
            auc_values = group['avg_auc'].values
            auc_mean = np.mean(auc_values)
            auc_std = np.std(auc_values, ddof=1)
            
            # Total expert quantity
            total_experts = shared + task
            
            stats_list.append({
                'shared_experts': int(shared),
                'task_experts': int(task),
                'total_experts': int(total_experts),
                'config': f"S{int(shared)}_T{int(task)}",
                'n_runs': n_runs,
                'js_mean': float(js_mean),
                'js_std': float(js_std),
                'js_median': float(js_median),
                'js_min': float(js_min),
                'js_max': float(js_max),
                'js_range': float(js_range),
                'js_var': float(js_var),
                'js_cv': float(js_cv),
                'js_ci_lower': float(js_ci_lower),
                'js_ci_upper': float(js_ci_upper),
                'js_ci_width': float(js_ci_width),
                'acc_mean': float(acc_mean),
                'acc_std': float(acc_std),
                'auc_mean': float(auc_mean),
                'auc_std': float(auc_std)
            })
        
        df_stats = pd.DataFrame(stats_list)
        df_stats = df_stats.sort_values('js_mean', ascending=False)
        self.config_stats = df_stats
        
        print(f"✓ Computed statistics for {len(df_stats)} configurations")
        
        return df_stats
    
    def get_top_10_configurations(self):
        """
        Get top 10 configurations by mean Joint Score.
        
        Returns:
            pd.DataFrame with top 10 configurations
        """
        if self.config_stats is None:
            self.compute_configuration_statistics()
        
        print("\n" + "="*80)
        print("TOP 10 CONFIGURATIONS BY MEAN JOINT SCORE")
        print("="*80)
        
        self.top_10_configs = self.config_stats.head(10).copy()
        self.top_10_configs['rank_by_mean'] = range(1, 11)
        
        # Add ranking by stability metrics
        self.top_10_configs['rank_by_ci_width'] = self.top_10_configs['js_ci_width'].rank(method='min').astype(int)
        self.top_10_configs['rank_by_variance'] = self.top_10_configs['js_var'].rank(method='min').astype(int)
        self.top_10_configs['rank_by_cv'] = self.top_10_configs['js_cv'].rank(method='min').astype(int)
        
        # Add ranking by instability (broadest CI, largest variance)
        self.top_10_configs['rank_by_ci_width_worst'] = self.top_10_configs['js_ci_width'].rank(ascending=False, method='min').astype(int)
        self.top_10_configs['rank_by_variance_worst'] = self.top_10_configs['js_var'].rank(ascending=False, method='min').astype(int)
        
        print("\nTop 10 Configurations:\n")
        print(self.top_10_configs[['rank_by_mean', 'config', 'js_mean', 'js_std', 'js_ci_width', 'total_experts']].to_string(index=False))
        
        return self.top_10_configs
    
    def get_best_configuration(self, stability_metric='ci_width', expert_efficiency='min'):
        """
        Find best configuration among top 10.
        
        Criteria:
        1. Most stable: Among top 10, select 3 with narrowest CI or least variance
        2. Most efficient: Among those 3, select one with least expert quantity
        
        Parameters:
            stability_metric: 'ci_width' (narrowest CI) or 'variance' (least variance)
            expert_efficiency: 'min' (least experts) or 'max' (most experts)
        
        Returns:
            dict with best configuration details
        """
        if self.top_10_configs is None:
            self.get_top_10_configurations()
        
        print("\n" + "="*80)
        print("FINDING BEST CONFIGURATION")
        print("="*80)
        print(f"Stability metric: {stability_metric}")
        print(f"Expert efficiency: {expert_efficiency} (least total experts)\n")
        
        # Step 1: Select top 3 most stable configurations
        if stability_metric == 'ci_width':
            top_3_stable = self.top_10_configs.nsmallest(3, 'js_ci_width').copy()
            metric_name = "CI Width"
            metric_col = 'js_ci_width'
        elif stability_metric == 'variance':
            top_3_stable = self.top_10_configs.nsmallest(3, 'js_var').copy()
            metric_name = "Variance"
            metric_col = 'js_var'
        else:
            raise ValueError(f"Unknown stability_metric: {stability_metric}")
        
        print(f"Top 3 Most Stable by {metric_name}:\n")
        print(top_3_stable[['config', 'js_mean', metric_col, 'total_experts']].to_string(index=False))
        
        # Step 2: Among top 3, select one with least expert quantity
        best_config_row = top_3_stable.loc[top_3_stable['total_experts'].idxmin()]
        
        print(f"\nBest Configuration (most stable + least experts):")
        print(f"  Configuration: {best_config_row['config']}")
        print(f"  Shared Experts: {best_config_row['shared_experts']}")
        print(f"  Task Experts: {best_config_row['task_experts']}")
        print(f"  Total Experts: {best_config_row['total_experts']}")
        print(f"  Mean JS: {best_config_row['js_mean']:.6f}")
        print(f"  JS Std: {best_config_row['js_std']:.6f}")
        print(f"  JS {metric_name}: {best_config_row[metric_col]:.6f}")
        print(f"  95% CI Width: {best_config_row['js_ci_width']:.6f}")
        print(f"  95% CI: [{best_config_row['js_ci_lower']:.6f}, {best_config_row['js_ci_upper']:.6f}]")
        print(f"  Number of runs: {best_config_row['n_runs']}")
        
        self.best_config = {
            'config': best_config_row['config'],
            'shared_experts': int(best_config_row['shared_experts']),
            'task_experts': int(best_config_row['task_experts']),
            'total_experts': int(best_config_row['total_experts']),
            'js_mean': float(best_config_row['js_mean']),
            'js_std': float(best_config_row['js_std']),
            'js_ci_lower': float(best_config_row['js_ci_lower']),
            'js_ci_upper': float(best_config_row['js_ci_upper']),
            'js_ci_width': float(best_config_row['js_ci_width']),
            'stability_metric': stability_metric,
            'stability_value': float(best_config_row[metric_col]),
            'acc_mean': float(best_config_row['acc_mean']),
            'auc_mean': float(best_config_row['auc_mean']),
            'n_runs': int(best_config_row['n_runs'])
        }
        
        return self.best_config
    
    def get_worst_configuration(self, stability_metric='ci_width', expert_penalty='max'):
        """
        Find worst configuration among top 10.
        
        Criteria:
        1. Least stable: Among top 10, select 3 with broadest CI or largest variance
        2. Least efficient: Among those 3, select one with most expert quantity
        
        Parameters:
            stability_metric: 'ci_width' (broadest CI) or 'variance' (largest variance)
            expert_penalty: 'max' (most experts)
        
        Returns:
            dict with worst configuration details
        """
        if self.top_10_configs is None:
            self.get_top_10_configurations()
        
        print("\n" + "="*80)
        print("FINDING WORST CONFIGURATION (Among Top 10)")
        print("="*80)
        print(f"Stability metric: {stability_metric}")
        print(f"Expert penalty: {expert_penalty} (most total experts)\n")
        
        # Step 1: Select bottom 3 least stable configurations (among top 10)
        if stability_metric == 'ci_width':
            bottom_3_unstable = self.top_10_configs.nlargest(3, 'js_ci_width').copy()
            metric_name = "CI Width"
            metric_col = 'js_ci_width'
        elif stability_metric == 'variance':
            bottom_3_unstable = self.top_10_configs.nlargest(3, 'js_var').copy()
            metric_name = "Variance"
            metric_col = 'js_var'
        else:
            raise ValueError(f"Unknown stability_metric: {stability_metric}")
        
        print(f"Bottom 3 Least Stable by {metric_name}:\n")
        print(bottom_3_unstable[['config', 'js_mean', metric_col, 'total_experts']].to_string(index=False))
        
        # Step 2: Among bottom 3, select one with most expert quantity
        worst_config_row = bottom_3_unstable.loc[bottom_3_unstable['total_experts'].idxmax()]
        
        print(f"\nWorst Configuration (least stable + most experts):")
        print(f"  Configuration: {worst_config_row['config']}")
        print(f"  Shared Experts: {worst_config_row['shared_experts']}")
        print(f"  Task Experts: {worst_config_row['task_experts']}")
        print(f"  Total Experts: {worst_config_row['total_experts']}")
        print(f"  Mean JS: {worst_config_row['js_mean']:.6f}")
        print(f"  JS Std: {worst_config_row['js_std']:.6f}")
        print(f"  JS {metric_name}: {worst_config_row[metric_col]:.6f}")
        print(f"  95% CI Width: {worst_config_row['js_ci_width']:.6f}")
        print(f"  95% CI: [{worst_config_row['js_ci_lower']:.6f}, {worst_config_row['js_ci_upper']:.6f}]")
        print(f"  Number of runs: {worst_config_row['n_runs']}")
        
        self.worst_config = {
            'config': worst_config_row['config'],
            'shared_experts': int(worst_config_row['shared_experts']),
            'task_experts': int(worst_config_row['task_experts']),
            'total_experts': int(worst_config_row['total_experts']),
            'js_mean': float(worst_config_row['js_mean']),
            'js_std': float(worst_config_row['js_std']),
            'js_ci_lower': float(worst_config_row['js_ci_lower']),
            'js_ci_upper': float(worst_config_row['js_ci_upper']),
            'js_ci_width': float(worst_config_row['js_ci_width']),
            'stability_metric': stability_metric,
            'instability_value': float(worst_config_row[metric_col]),
            'acc_mean': float(worst_config_row['acc_mean']),
            'auc_mean': float(worst_config_row['auc_mean']),
            'n_runs': int(worst_config_row['n_runs'])
        }
        
        return self.worst_config
    
    def plot_joint_score_distributions(self, top_k=10, save_path=None):
        """
        Plot Joint Score distributions for top configurations across runs.
        
        Creates two subplots:
        1. Box plot showing distribution for each configuration
        2. Mean with 95% CI error bars
        """
        if self.config_stats is None:
            self.compute_configuration_statistics()
        
        if self.top_10_configs is None:
            self.get_top_10_configurations()
        
        # Use top_k from config_stats
        top_configs_data = self.config_stats.head(top_k).copy()
        top_configs = [tuple(row[['shared_experts', 'task_experts']].astype(int)) 
                      for _, row in top_configs_data.iterrows()]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Prepare data
        data_for_box = []
        labels_for_box = []
        colors_palette = sns.color_palette("husl", top_k)
        
        for config in top_configs:
            if config in self.results_dict:
                data = [d['joint_score'] for d in self.results_dict[config].values()]
                data_for_box.append(data)
                labels_for_box.append(f"S{config[0]}_T{config[1]}")
        
        # Plot 1: Box plot
        bp = axes[0].boxplot(data_for_box, labels=labels_for_box, patch_artist=True, 
                            widths=0.6, showmeans=True,
                            meanprops=dict(marker='D', markerfacecolor='red', 
                                         markersize=8, markeredgecolor='darkred'))
        
        for patch, color in zip(bp['boxes'], colors_palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[0].set_ylabel('Joint Score', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Configuration', fontsize=12, fontweight='bold')
        axes[0].set_title(f'Joint Score Distribution - Top {top_k} Configurations', 
                         fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Mean with 95% CI
        positions = range(len(top_configs))
        means = []
        ci_widths = []
        ci_lowers = []
        ci_uppers = []
        
        for idx, config in enumerate(top_configs):
            if config in self.results_dict:
                data = np.array([d['joint_score'] for d in self.results_dict[config].values()])
                mean = np.mean(data)
                ci_lower, ci_upper = self._compute_ci_t(data, self.confidence_level)
                
                means.append(mean)
                ci_lowers.append(ci_lower)
                ci_uppers.append(ci_upper)
                ci_widths.append(ci_upper - ci_lower)
                
                # Plot CI as error bar
                axes[1].errorbar(idx, mean, 
                               yerr=[[mean - ci_lower], [ci_upper - mean]],
                               fmt='o', markersize=10, capsize=8, capthick=2.5,
                               color=colors_palette[idx], alpha=0.8,
                               elinewidth=2.5, markeredgecolor='black', markeredgewidth=1.5)
        
        axes[1].set_xticks(positions)
        axes[1].set_xticklabels(labels_for_box, fontsize=11)
        axes[1].set_ylabel('Joint Score', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Configuration', fontsize=12, fontweight='bold')
        axes[1].set_title(f'Mean Joint Score with 95% CI - Top {top_k} Configurations', 
                         fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Add horizontal line for reference
        axes[1].axhline(y=np.mean(means), color='gray', linestyle='--', 
                       alpha=0.5, label=f'Mean across all: {np.mean(means):.4f}')
        axes[1].legend(loc='best')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Distribution plot saved to: {save_path}")
        plt.show()
    
    def plot_top10_mean_with_ci(self, output_dir='plots', save_filename='top10_mean_with_ci.png'):
        """
        Enhanced plot of Top 10 configurations with mean and 95% CI.
        Highlights best and worst configurations with special markers.
        """
        if self.top_10_configs is None:
            self.get_top_10_configurations()
        
        os.makedirs(output_dir, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        top_10 = self.top_10_configs.head(10)
        x = np.arange(len(top_10))
        
        # Color codes for bars
        colors = []
        for config in top_10['config']:
            if self.best_config and config == self.best_config['config']:
                colors.append('#2ecc71')  # Green for best
            elif self.worst_config and config == self.worst_config['config']:
                colors.append('#e74c3c')  # Red for worst
            else:
                colors.append('#3498db')  # Blue for others
        
        # Plot bars
        bars = ax.bar(x, top_10['js_mean'], alpha=0.7, color=colors, 
                     edgecolor='black', linewidth=1.5, width=0.6)
        
        # Plot error bars (CI)
        ax.errorbar(x, top_10['js_mean'], 
                   yerr=[top_10['js_mean'] - top_10['js_ci_lower'],
                        top_10['js_ci_upper'] - top_10['js_mean']],
                   fmt='none', color='black', capsize=7, capthick=2.5,
                   elinewidth=2, label='95% Confidence Interval')
        
        # Highlight best and worst with special markers
        if self.best_config:
            best_idx = list(top_10['config']).index(self.best_config['config'])
            ax.scatter(best_idx, top_10.iloc[best_idx]['js_mean'], 
                     s=600, marker='*', color='darkgreen', zorder=5, 
                     edgecolors='black', linewidth=2, label='Best Config')
        
        if self.worst_config:
            worst_idx = list(top_10['config']).index(self.worst_config['config'])
            ax.scatter(worst_idx, top_10.iloc[worst_idx]['js_mean'], 
                     s=400, marker='X', color='darkred', zorder=5, 
                     edgecolors='black', linewidth=2, label='Worst Config')
        
        # Formatting
        ax.set_xlabel('Configuration', fontsize=13, fontweight='bold')
        ax.set_ylabel('Joint Score', fontsize=13, fontweight='bold')
        ax.set_title('Top 10 Configurations: Mean Joint Score with 95% Confidence Interval', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(top_10['config'], fontsize=11, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Add value labels on bars
        for i, (bar, mean, ci_low, ci_up) in enumerate(zip(bars, top_10['js_mean'], 
                                                            top_10['js_ci_lower'], 
                                                            top_10['js_ci_upper'])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Legend
        ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
        
        # Add horizontal reference line
        overall_mean = top_10['js_mean'].mean()
        ax.axhline(y=overall_mean, color='gray', linestyle='--', alpha=0.5, 
                  linewidth=1.5, label=f'Overall mean: {overall_mean:.4f}')
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, save_filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Top 10 plot saved to: {save_path}")
        
        return save_path
    
    def plot_stability_comparison(self, output_dir='plots', save_filename='stability_comparison.png'):
        """
        Enhanced stability comparison plot (CI Width vs Variance).
        """
        if self.top_10_configs is None:
            self.get_top_10_configurations()
        
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        top_10 = self.top_10_configs.head(10)
        
        # Color codes
        colors = []
        for config in top_10['config']:
            if self.best_config and config == self.best_config['config']:
                colors.append('#2ecc71')  # Green
            elif self.worst_config and config == self.worst_config['config']:
                colors.append('#e74c3c')  # Red
            else:
                colors.append('#3498db')  # Blue
        
        # CI Width (smaller = more stable)
        axes[0].barh(range(len(top_10)), top_10['js_ci_width'], 
                    color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        axes[0].set_yticks(range(len(top_10)))
        axes[0].set_yticklabels(top_10['config'], fontsize=11)
        axes[0].set_xlabel('95% CI Width', fontsize=12, fontweight='bold')
        axes[0].set_title('Stability: Confidence Interval Width\n(Narrower = More Stable)', 
                         fontsize=12, fontweight='bold')
        axes[0].invert_yaxis()
        axes[0].grid(True, alpha=0.3, axis='x', linestyle='--')
        
        # Variance (smaller = more stable)
        axes[1].barh(range(len(top_10)), top_10['js_var'], 
                    color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        axes[1].set_yticks(range(len(top_10)))
        axes[1].set_yticklabels(top_10['config'], fontsize=11)
        axes[1].set_xlabel('Variance', fontsize=12, fontweight='bold')
        axes[1].set_title('Stability: Variance Across Runs\n(Lower = More Stable)', 
                         fontsize=12, fontweight='bold')
        axes[1].invert_yaxis()
        axes[1].grid(True, alpha=0.3, axis='x', linestyle='--')
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, save_filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Stability comparison plot saved to: {save_path}")
        
        return save_path
    
    def plot_performance_vs_complexity(self, output_dir='plots', save_filename='performance_vs_complexity.png'):
        """
        Enhanced scatter plot: Performance vs Complexity.
        """
        if self.top_10_configs is None:
            self.get_top_10_configurations()
        
        os.makedirs(output_dir, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        top_10 = self.top_10_configs.head(10)
        
        # Create scatter plot with bubble size = CI width
        scatter = ax.scatter(top_10['total_experts'], top_10['js_mean'], 
                           s=top_10['js_ci_width']*300, c=top_10['js_mean'],
                           cmap='viridis', alpha=0.6, edgecolors='black', linewidth=2)
        
        # Annotate points
        for idx, (_, row) in enumerate(top_10.iterrows()):
            ax.annotate(row['config'], 
                       (row['total_experts'], row['js_mean']),
                       xytext=(8, 8), textcoords='offset points', 
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
        
        # Highlight best and worst
        if self.best_config:
            best_row = top_10[top_10['config'] == self.best_config['config']].iloc[0]
            ax.scatter(best_row['total_experts'], best_row['js_mean'], 
                     s=800, marker='*', color='green', zorder=5, 
                     edgecolors='darkgreen', linewidth=2.5, label='Best Config')
        
        if self.worst_config:
            worst_row = top_10[top_10['config'] == self.worst_config['config']].iloc[0]
            ax.scatter(worst_row['total_experts'], worst_row['js_mean'], 
                     s=500, marker='X', color='red', zorder=5, 
                     edgecolors='darkred', linewidth=2.5, label='Worst Config')
        
        # Formatting
        ax.set_xlabel('Total Expert Quantity (Shared + Task)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Mean Joint Score', fontsize=13, fontweight='bold')
        ax.set_title('Configuration Analysis: Performance vs Complexity\n(Bubble size = CI width; Color = Mean JS)', 
                    fontsize=13, fontweight='bold', pad=20)
        cbar = plt.colorbar(scatter, ax=ax, label='Mean Joint Score')
        cbar.ax.tick_params(labelsize=10)
        
        ax.legend(loc='best', fontsize=12, framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, save_filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Performance vs Complexity plot saved to: {save_path}")
        
        return save_path
    
    def generate_comprehensive_report(self, output_file='best_worst_configuration_report.txt'):
        """
        Generate comprehensive analysis report.
        """
        if self.best_config is None or self.worst_config is None:
            print("Error: Run get_best_configuration() and get_worst_configuration() first")
            return
        
        report_lines = []
        
        report_lines.append("="*80)
        report_lines.append("BEST AND WORST CONFIGURATION ANALYSIS")
        report_lines.append("="*80)
        report_lines.append(f"\nReport Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Data File: {self.csv_file}")
        
        # Section 1: Overview
        report_lines.append("\n" + "="*80)
        report_lines.append("1. ANALYSIS OVERVIEW")
        report_lines.append("="*80)
        report_lines.append("\nMethod:")
        report_lines.append("1. Identify top 10 configurations by mean Joint Score")
        report_lines.append("2. BEST: Among top 10, select 3 most stable configs, then choose")
        report_lines.append("   one with minimum total expert quantity (shared + task experts)")
        report_lines.append("3. WORST: Among top 10, select 3 least stable configs, then choose")
        report_lines.append("   one with maximum total expert quantity (shared + task experts)")
        
        report_lines.append("\nStability Metrics:")
        report_lines.append("  - CI Width: Width of 95% confidence interval (smaller = more stable)")
        report_lines.append("  - Variance: Variance across runs (smaller = more stable)")
        report_lines.append("  - Coefficient of Variation: Ratio of std to mean")
        
        # Section 2: Top 10 Summary
        report_lines.append("\n" + "="*80)
        report_lines.append("2. TOP 10 CONFIGURATIONS SUMMARY")
        report_lines.append("="*80)
        
        report_lines.append("\nRanked by Mean Joint Score:\n")
        for idx, (_, row) in enumerate(self.top_10_configs.head(10).iterrows(), 1):
            report_lines.append(f"  {idx:2d}. {row['config']}")
            report_lines.append(f"      Mean JS: {row['js_mean']:.6f} +/- {row['js_std']:.6f}")
            report_lines.append(f"      95% CI: [{row['js_ci_lower']:.6f}, {row['js_ci_upper']:.6f}]")
            report_lines.append(f"      CI Width: {row['js_ci_width']:.6f}")
            report_lines.append(f"      Variance: {row['js_var']:.6f}")
            report_lines.append(f"      Total Experts: {row['total_experts']} ({row['shared_experts']} shared + {row['task_experts']} task)")
            report_lines.append("")
        
        # Section 3: Best Configuration Details
        report_lines.append("\n" + "="*80)
        report_lines.append("3. RECOMMENDED BEST CONFIGURATION")
        report_lines.append("="*80)
        
        best = self.best_config
        report_lines.append(f"\nConfiguration: {best['config']}")
        report_lines.append(f"  Architecture:")
        report_lines.append(f"    - Shared Experts: {best['shared_experts']}")
        report_lines.append(f"    - Task Experts: {best['task_experts']}")
        report_lines.append(f"    - Total Experts: {best['total_experts']}")
        report_lines.append(f"\n  Performance Metrics:")
        report_lines.append(f"    - Mean Joint Score: {best['js_mean']:.6f}")
        report_lines.append(f"    - JS Std Dev: {best['js_std']:.6f}")
        report_lines.append(f"    - 95% CI: [{best['js_ci_lower']:.6f}, {best['js_ci_upper']:.6f}]")
        report_lines.append(f"    - CI Width: {best['js_ci_width']:.6f}")
        report_lines.append(f"    - Mean Accuracy: {best['acc_mean']:.6f}")
        report_lines.append(f"    - Mean AUC: {best['auc_mean']:.6f}")
        report_lines.append(f"    - Number of Runs: {best['n_runs']}")
        
        report_lines.append(f"\n  Selection Rationale:")
        report_lines.append(f"    - Ranked 3rd most stable by {best['stability_metric']}")
        report_lines.append(f"    - {best['stability_metric']} value: {best['stability_value']:.6f}")
        report_lines.append(f"    - Among top 3 stable configs, has MINIMUM total expert quantity")
        report_lines.append(f"    - Provides best balance: good performance + stability + efficiency")
        
        # Section 4: Worst Configuration Details
        report_lines.append("\n" + "="*80)
        report_lines.append("4. IDENTIFIED WORST CONFIGURATION (To Avoid)")
        report_lines.append("="*80)
        
        worst = self.worst_config
        report_lines.append(f"\nConfiguration: {worst['config']}")
        report_lines.append(f"  Architecture:")
        report_lines.append(f"    - Shared Experts: {worst['shared_experts']}")
        report_lines.append(f"    - Task Experts: {worst['task_experts']}")
        report_lines.append(f"    - Total Experts: {worst['total_experts']}")
        report_lines.append(f"\n  Performance Metrics:")
        report_lines.append(f"    - Mean Joint Score: {worst['js_mean']:.6f}")
        report_lines.append(f"    - JS Std Dev: {worst['js_std']:.6f}")
        report_lines.append(f"    - 95% CI: [{worst['js_ci_lower']:.6f}, {worst['js_ci_upper']:.6f}]")
        report_lines.append(f"    - CI Width: {worst['js_ci_width']:.6f}")
        report_lines.append(f"    - Mean Accuracy: {worst['acc_mean']:.6f}")
        report_lines.append(f"    - Mean AUC: {worst['auc_mean']:.6f}")
        report_lines.append(f"    - Number of Runs: {worst['n_runs']}")
        
        report_lines.append(f"\n  Why It's Worst:")
        report_lines.append(f"    - Ranked 3rd LEAST stable by {worst['stability_metric']}")
        report_lines.append(f"    - {worst['stability_metric']} value: {worst['instability_value']:.6f}")
        report_lines.append(f"    - Among bottom 3 unstable configs, has MAXIMUM total expert quantity")
        report_lines.append(f"    - Represents worst case: poor stability + maximum complexity")
        
        # Section 5: Comparative Analysis
        report_lines.append("\n" + "="*80)
        report_lines.append("5. COMPARATIVE ANALYSIS: BEST VS WORST")
        report_lines.append("="*80)
        
        js_diff = best['js_mean'] - worst['js_mean']
        expert_diff = worst['total_experts'] - best['total_experts']
        ci_ratio = worst['js_ci_width'] / best['js_ci_width']
        
        report_lines.append(f"\nPerformance Comparison:")
        report_lines.append(f"  - Best JS Mean: {best['js_mean']:.6f}")
        report_lines.append(f"  - Worst JS Mean: {worst['js_mean']:.6f}")
        report_lines.append(f"  - Difference: {js_diff:.6f} ({js_diff/worst['js_mean']*100:.2f}%)")
        
        report_lines.append(f"\nStability Comparison:")
        report_lines.append(f"  - Best CI Width: {best['js_ci_width']:.6f}")
        report_lines.append(f"  - Worst CI Width: {worst['js_ci_width']:.6f}")
        report_lines.append(f"  - Ratio (Worst/Best): {ci_ratio:.2f}x")
        
        report_lines.append(f"\nComplexity Comparison:")
        report_lines.append(f"  - Best Total Experts: {best['total_experts']}")
        report_lines.append(f"  - Worst Total Experts: {worst['total_experts']}")
        report_lines.append(f"  - Difference: {expert_diff} experts")
        
        # Section 6: Recommendations
        report_lines.append("\n" + "="*80)
        report_lines.append("6. RECOMMENDATIONS")
        report_lines.append("="*80)
        
        report_lines.append(f"\nCHECKMARK RECOMMENDED CONFIGURATION: {best['config']}")
        report_lines.append(f"  - Use this configuration for production deployment")
        report_lines.append(f"  - Provides stable performance with minimal model complexity")
        report_lines.append(f"  - Expected Joint Score: {best['js_mean']:.6f}")
        report_lines.append(f"  - Confidence: With 95% confidence, will achieve at least {best['js_ci_lower']:.6f}")
        
        report_lines.append(f"\nX AVOID CONFIGURATION: {worst['config']}")
        report_lines.append(f"  - This configuration shows high instability across runs")
        report_lines.append(f"  - Despite being in top 10 by mean, has poor reproducibility")
        report_lines.append(f"  - Contains {expert_diff} more experts than best config (unnecessary complexity)")
        
        report_lines.append(f"\nAdditional Notes:")
        report_lines.append(f"  - The best configuration achieves comparable performance")
        report_lines.append(f"  - With {expert_diff}x fewer experts, reducing model complexity")
        report_lines.append(f"  - Better stability across runs ({ci_ratio:.2f}x narrower CI)")
        report_lines.append(f"  - More suitable for practical deployment and maintenance")
        
        report_text = "\n".join(report_lines)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\n✅ Report saved to: {output_file}")
        
        return report_text
    
    def plot_analysis(self, output_dir='best_worst_plots'):
        """Create all visualization plots."""
        os.makedirs(output_dir, exist_ok=True)
        
        if self.top_10_configs is None:
            self.get_top_10_configurations()
        
        print("\n" + "="*80)
        print("GENERATING VISUALIZATION PLOTS")
        print("="*80 + "\n")
        
        # Plot 1: Joint Score Distributions
        self.plot_joint_score_distributions(top_k=10, 
                                           save_path=os.path.join(output_dir, 'joint_score_distributions.png'))
        
        # Plot 2: Top 10 Mean with CI
        self.plot_top10_mean_with_ci(output_dir=output_dir)
        
        # Plot 3: Stability Comparison
        self.plot_stability_comparison(output_dir=output_dir)
        
        # Plot 4: Performance vs Complexity
        self.plot_performance_vs_complexity(output_dir=output_dir)
        
        print(f"\n✅ All plots saved to: {output_dir}")


# Main function
def analyze_best_worst_configuration(csv_file, 
                                    stability_metric_best='ci_width',
                                    stability_metric_worst='ci_width',
                                    output_dir='best_worst_analysis_results'):
    """
    Run complete best/worst configuration analysis.
    
    Parameters:
        csv_file: Path to multi-run results CSV
        stability_metric_best: 'ci_width' or 'variance' for best config selection
        stability_metric_worst: 'ci_width' or 'variance' for worst config selection
        output_dir: Output directory for reports and plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("BEST AND WORST CONFIGURATION ANALYSIS")
    print("="*80 + "\n")
    
    # Initialize and run analysis
    analyzer = ConfigurationAnalyzer(csv_file)
    
    # Load data
    analyzer.load_data()
    
    # Compute statistics
    analyzer.compute_configuration_statistics(ci_method='t_distribution')
    
    # Get top 10
    analyzer.get_top_10_configurations()
    
    # Find best configuration
    best_config = analyzer.get_best_configuration(
        stability_metric=stability_metric_best,
        expert_efficiency='min'
    )
    
    # Find worst configuration
    worst_config = analyzer.get_worst_configuration(
        stability_metric=stability_metric_worst,
        expert_penalty='max'
    )
    
    # Generate report
    report_path = os.path.join(output_dir, 'best_worst_configuration_report.txt')
    analyzer.generate_comprehensive_report(output_file=report_path)
    
    # Create plots
    analyzer.plot_analysis(output_dir=os.path.join(output_dir, 'plots'))
    
    # Save config stats
    stats_path = os.path.join(output_dir, 'all_config_statistics.csv')
    analyzer.config_stats.to_csv(stats_path, index=False)
    print(f"✅ Configuration statistics saved to: {stats_path}")
    
    print(f"\n✅ Analysis complete! Results saved to: {output_dir}")
    
    return analyzer, best_config, worst_config


if __name__ == '__main__':
    # Run analysis
    csv_file = 'multirun_results/multi_run_results.csv'
    
    analyzer, best_config, worst_config = analyze_best_worst_configuration(
        csv_file=csv_file,
        stability_metric_best='ci_width',    # Use narrowest CI for best config
        stability_metric_worst='ci_width',   # Use broadest CI for worst config
        output_dir='best_worst_analysis_results'
    )
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n✓ Best Configuration: {best_config['config']}")
    print(f"  - Total Experts: {best_config['total_experts']}")
    print(f"  - Mean JS: {best_config['js_mean']:.6f}")
    print(f"  - 95% CI Width: {best_config['js_ci_width']:.6f}")
    
    print(f"\n✗ Worst Configuration (to avoid): {worst_config['config']}")
    print(f"  - Total Experts: {worst_config['total_experts']}")
    print(f"  - Mean JS: {worst_config['js_mean']:.6f}")
    print(f"  - 95% CI Width: {worst_config['js_ci_width']:.6f}")
    
    print("\n✅ Analysis Complete!")