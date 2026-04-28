'''
Comprehensive Analysis of Validation and Test Performance
================================================================================
This script analyzes the relationship between validation and test set performance
across all configurations, determining whether validation Joint Score can
reliably predict test performance.

Features:
  - Rank configurations by validation and test JS with 95% CI visualization
  - Identify representative configurations in performance ranges
  - Annotate all configurations on x-axis
  - Detailed analysis plots for representative configurations
  - New plot showing representative configs with 95% CI on val and test sets
  - Correlation analysis between validation and test JS
  - Generalization gap analysis (overfitting/underfitting detection)
'''

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import spearmanr, kendalltau, ttest_rel, pearsonr
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings('ignore')


class PerformanceAnalyzer:
    """Analyzer for validation and test performance relationships."""
    
    def __init__(self, results_dir='multirun_1/multirun_validation_results',
                 output_dir=None, confidence_level=0.95):
        """
        Initialize analyzer.
        
        Parameters:
            results_dir: Directory containing validation and test CSV files
            output_dir: Directory for analysis results (default: same as results_dir)
            confidence_level: Confidence level for CI (default: 0.95 = 95%)
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir) if output_dir else self.results_dir / 'analysis'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level  # For CI calculation
        
        # Load data
        self.val_df = self._load_csv('multi_run_validation_results.csv')
        self.test_df = self._load_csv('multi_run_test_results.csv')
        
        # Compute per-configuration statistics
        self.val_stats = self._compute_statistics(self.val_df)
        self.test_stats = self._compute_statistics(self.test_df)
        
        # Task mapping
        self.dict_classes = ['1_missing', '2_trend', '3_drift']
    
    def _load_csv(self, filename):
        """Load CSV file from results directory."""
        filepath = self.results_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        df = pd.read_csv(filepath)
        print(f"✓ Loaded {filename}: {df.shape}")
        return df
    
    def _compute_statistics(self, df):
        """Compute mean, std, and CI for each configuration."""
        stats_data = []
        
        for (shared, task), group in df.groupby(['shared_experts', 'task_experts']):
            js_values = group['joint_score'].values
            acc_values = group['avg_accuracy'].values
            auc_values = group['avg_auc'].values
            
            # Compute statistics
            js_mean = np.mean(js_values)
            js_std = np.std(js_values)
            js_se = js_std / np.sqrt(len(js_values))
            
            # 95% CI using t-distribution
            t_crit = stats.t.ppf(1 - self.alpha / 2, len(js_values) - 1)
            js_ci_lower = js_mean - t_crit * js_se
            js_ci_upper = js_mean + t_crit * js_se
            
            # Accuracy and AUC
            acc_mean = np.mean(acc_values)
            acc_std = np.std(acc_values)
            auc_mean = np.mean(auc_values)
            auc_std = np.std(auc_values)
            
            # Coefficient of variation (stability metric)
            js_cv = js_std / js_mean if js_mean > 0 else 0
            
            stats_data.append({
                'shared_experts': shared,
                'task_experts': task,
                'config': f'S{shared}_T{task}',
                'total_experts': shared + task,
                'num_runs': len(js_values),
                'js_mean': js_mean,
                'js_std': js_std,
                'js_se': js_se,
                'js_ci_lower': js_ci_lower,
                'js_ci_upper': js_ci_upper,
                'js_ci_width': js_ci_upper - js_ci_lower,
                'js_cv': js_cv,
                'acc_mean': acc_mean,
                'acc_std': acc_std,
                'auc_mean': auc_mean,
                'auc_std': auc_std,
            })
        
        return pd.DataFrame(stats_data).sort_values('js_mean', ascending=False)
    
    def _compute_ci_t(self, data, confidence_level=0.95):
        """Compute confidence interval using t-distribution."""
        alpha = 1 - confidence_level
        mean = np.mean(data)
        se = stats.sem(data)
        t_crit = stats.t.ppf(1 - alpha / 2, len(data) - 1)
        ci_lower = mean - t_crit * se
        ci_upper = mean + t_crit * se
        return ci_lower, ci_upper
    
    def _identify_representative_configs(self, top_k=100, interval=10):
        """
        Identify representative configurations for each performance range.
        
        Splits top_k configurations into ranges and selects the one with least std
        (most stable) in each range.
        
        Parameters:
            top_k: Number of top configurations to analyze
            interval: Size of each range (default: 10)
        
        Returns:
            List of representative configurations with their range info
        """
        top_configs = self.val_stats.head(top_k).reset_index(drop=True)
        
        representatives = []
        ranges = []
        
        for range_idx in range(0, len(top_configs), interval):
            range_end = min(range_idx + interval, len(top_configs))
            range_configs = top_configs.iloc[range_idx:range_end]
            
            # Find configuration with least std (most stable)
            most_stable_idx = range_configs['js_std'].idxmin()
            most_stable = range_configs.loc[most_stable_idx]
            
            range_info = {
                'range_num': range_idx // interval + 1,
                'range_start': range_idx + 1,
                'range_end': range_end,
                'config': most_stable['config'],
                'rank': most_stable_idx + 1,
                'js_mean': most_stable['js_mean'],
                'js_std': most_stable['js_std'],
                'js_cv': most_stable['js_cv'],
            }
            
            ranges.append(range_info)
            representatives.append(most_stable)
        
        return ranges, representatives
    
    def plot_val_test_comparison_with_ci_refined(self, top_k=100, interval=10):
        """
        Refined Plot 1: Rank configurations by mean JS with 95% CI for both val and test.
        
        - Shows ALL configurations in x-axis (using small font)
        - Identifies representative configurations for performance ranges
        - Annotates ranges and representative configurations with config names
        - Provides clear visualization of performance patterns
        """
        print("\n" + "="*80)
        print(f"Plot 1 (Refined): Validation vs Test JS Comparison (Top {top_k})")
        print("="*80)
        
        # Get top configurations by validation JS
        top_configs = self.val_stats.head(top_k)
        
        # Identify representative configurations
        ranges, representatives = self._identify_representative_configs(top_k, interval)
        
        fig = plt.figure(figsize=(32, 10))
        
        positions = np.arange(len(top_configs))
        x_labels = [row['config'] for _, row in top_configs.iterrows()]
        
        # ===== Validation Set Performance =====
        val_means = top_configs['js_mean'].values
        val_ci_lowers = top_configs['js_ci_lower'].values
        val_ci_uppers = top_configs['js_ci_upper'].values
        val_errors_lower = val_means - val_ci_lowers
        val_errors_upper = val_ci_uppers - val_means
        
        plt.errorbar(positions, val_means,
                    yerr=[val_errors_lower, val_errors_upper],
                    fmt='o', markersize=5, capsize=3, capthick=1,
                    color='steelblue', alpha=0.6, elinewidth=1,
                    markeredgecolor='darkblue', markeredgewidth=0.5,
                    label='Validation JS', linewidth=1.5)
        
        # ===== Test Set Performance (aligned with val ranking) =====
        test_means = []
        test_ci_lowers = []
        test_ci_uppers = []
        
        for _, val_row in top_configs.iterrows():
            config = val_row['config']
            test_row = self.test_stats[self.test_stats['config'] == config]
            
            if not test_row.empty:
                test_means.append(test_row['js_mean'].values[0])
                test_ci_lowers.append(test_row['js_ci_lower'].values[0])
                test_ci_uppers.append(test_row['js_ci_upper'].values[0])
            else:
                test_means.append(np.nan)
                test_ci_lowers.append(np.nan)
                test_ci_uppers.append(np.nan)
        
        test_means = np.array(test_means)
        test_ci_lowers = np.array(test_ci_lowers)
        test_ci_uppers = np.array(test_ci_uppers)
        test_errors_lower = test_means - test_ci_lowers
        test_errors_upper = test_ci_uppers - test_means
        
        # Plot test
        plt.errorbar(positions, test_means,
                    yerr=[test_errors_lower, test_errors_upper],
                    fmt='s', markersize=5, capsize=3, capthick=1,
                    color='coral', alpha=0.6, elinewidth=1,
                    markeredgecolor='darkred', markeredgewidth=0.5,
                    label='Test JS', linewidth=1.5)
        
        # ===== Annotate ranges and representative configurations =====
        range_colors = plt.cm.Set3(np.linspace(0, 1, len(ranges)))
        
        for range_info, color in zip(ranges, range_colors):
            range_start = range_info['range_start'] - 1
            range_end = range_info['range_end']
            
            # Add background color for range
            plt.axvspan(range_start - 0.5, range_end - 0.5, alpha=0.15, color=color)
            
            # Add range label with configuration name
            range_mid = (range_start + range_end - 1) / 2
            label_text = f"Range {range_info['range_num']}\n({range_info['range_start']}-{range_info['range_end']})\n{range_info['config']}"
            plt.text(range_mid, plt.ylim()[1] * 0.98,
                    label_text,
                    ha='center', va='top', fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.8, edgecolor='black', linewidth=1))
            
            # Highlight representative configuration with larger star
            rep_idx = range_info['rank'] - 1
            if rep_idx < len(positions):
                plt.scatter(rep_idx, val_means[rep_idx], s=400, marker='*',
                          color=color, edgecolors='black', linewidth=1.5,
                          zorder=10)
        
        # ===== Configure axes =====
        plt.xticks(positions, x_labels, rotation=90, fontsize=5)
        plt.ylabel('Joint Score', fontsize=12, fontweight='bold')
        plt.xlabel('Configuration (Ranked by Validation JS) - All 100 Configurations', fontsize=12, fontweight='bold')
        plt.title(f'Validation vs Test JS Comparison with Representative Configurations (Top {top_k})',
                 fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, axis='y')
        plt.legend(loc='best', fontsize=11, ncol=2)
        
        # Add average lines
        plt.axhline(y=val_means.mean(), color='steelblue', linestyle='--', alpha=0.4,
                   linewidth=1.5)
        plt.axhline(y=np.nanmean(test_means), color='coral', linestyle='--', alpha=0.4,
                   linewidth=1.5)
        
        plt.tight_layout()
        plot_path = self.output_dir / '01_val_test_comparison_with_ci_refined.png'
        plt.savefig(str(plot_path), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {plot_path}")
        
        return ranges
    
    def plot_representative_configs_analysis(self, ranges):
        """
        Refined Plot: Detailed analysis of representative configurations.
        
        Shows:
        - Representative configuration performance with 95% CI (like plot_representative_configs_with_ci)
        - Stability (std) comparison (with annotated values)
        - Generalization gap for each representative
        - Summary table with configuration names and all metrics
        """
        print("\n" + "="*80)
        print("Plot (Refined): Representative Configurations Analysis")
        print("="*80)
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        
        # Extract metrics for representatives
        configs = [r['config'] for r in ranges]
        range_nums = [r['range_num'] for r in ranges]
        x_pos = np.arange(len(ranges))
        
        # Get validation metrics for representatives
        val_metrics = []
        for config in configs:
            val_row = self.val_stats[self.val_stats['config'] == config]
            if not val_row.empty:
                val_metrics.append({
                    'js_mean': val_row['js_mean'].values[0],
                    'js_ci_lower': val_row['js_ci_lower'].values[0],
                    'js_ci_upper': val_row['js_ci_upper'].values[0],
                    'js_std': val_row['js_std'].values[0],
                })
        
        # Get test metrics for representatives
        test_metrics = []
        for config in configs:
            test_row = self.test_stats[self.test_stats['config'] == config]
            if not test_row.empty:
                test_metrics.append({
                    'js_mean': test_row['js_mean'].values[0],
                    'js_ci_lower': test_row['js_ci_lower'].values[0],
                    'js_ci_upper': test_row['js_ci_upper'].values[0],
                    'js_std': test_row['js_std'].values[0],
                })
        
        # ===== Plot 1: Mean JS with 95% CI for Validation and Test =====
        val_means = np.array([m['js_mean'] for m in val_metrics])
        val_ci_lowers = np.array([m['js_ci_lower'] for m in val_metrics])
        val_ci_uppers = np.array([m['js_ci_upper'] for m in val_metrics])
        val_errors_lower = val_means - val_ci_lowers
        val_errors_upper = val_ci_uppers - val_means
        
        test_means = np.array([m['js_mean'] for m in test_metrics])
        test_ci_lowers = np.array([m['js_ci_lower'] for m in test_metrics])
        test_ci_uppers = np.array([m['js_ci_upper'] for m in test_metrics])
        test_errors_lower = test_means - test_ci_lowers
        test_errors_upper = test_ci_uppers - test_means
        
        width = 0.35
        
        # Plot validation with CI
        axes[0, 0].errorbar(x_pos - width/2, val_means,
                           yerr=[val_errors_lower, val_errors_upper],
                           fmt='o', markersize=12, capsize=8, capthick=2.5,
                           color='steelblue', alpha=0.8, elinewidth=2.5,
                           markeredgecolor='darkblue', markeredgewidth=2,
                           label='Validation JS', linewidth=2.5)
        
        # Plot test with CI
        axes[0, 0].errorbar(x_pos + width/2, test_means,
                           yerr=[test_errors_lower, test_errors_upper],
                           fmt='s', markersize=12, capsize=8, capthick=2.5,
                           color='coral', alpha=0.8, elinewidth=2.5,
                           markeredgecolor='darkred', markeredgewidth=2,
                           label='Test JS', linewidth=2.5)
        
        # Annotate mean values
        for i, (config, v_mean, t_mean) in enumerate(zip(configs, val_means, test_means)):
            # Validation annotation
            axes[0, 0].annotate(f'{v_mean:.4f}', (i - width/2, v_mean),
                               textcoords="offset points", xytext=(0, 15), ha='center',
                               fontsize=9, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.4', facecolor='steelblue', alpha=0.4))
            
            # Test annotation
            axes[0, 0].annotate(f'{t_mean:.4f}', (i + width/2, t_mean),
                               textcoords="offset points", xytext=(0, -20), ha='center',
                               fontsize=9, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.4', facecolor='coral', alpha=0.4))
        
        # Configure x-axis with range and config annotations
        x_labels = [f"Range {r['range_num']}\n{r['config']}" for r in ranges]
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(x_labels, fontsize=10, fontweight='bold')
        
        axes[0, 0].set_xlabel('Performance Range - Representative Configuration', fontsize=11, fontweight='bold')
        axes[0, 0].set_ylabel('Joint Score', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('Representative Configurations: Mean JS with 95% Confidence Intervals',
                            fontsize=12, fontweight='bold')
        axes[0, 0].legend(fontsize=10, loc='best')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # ===== Plot 2: Stability (Std Dev) Comparison (with annotated values) =====
        val_stds = np.array([m['js_std'] for m in val_metrics])
        test_stds = np.array([m['js_std'] for m in test_metrics])
        
        axes[0, 1].plot(range_nums, val_stds, 'o-', linewidth=3, markersize=11,
                       label='Validation Std', color='steelblue', markeredgecolor='darkblue',
                       markeredgewidth=1.5)
        axes[0, 1].plot(range_nums, test_stds, 's-', linewidth=3, markersize=11,
                       label='Test Std', color='coral', markeredgecolor='darkred',
                       markeredgewidth=1.5)
        
        # Annotate stability values
        for i, (config, v_std, t_std, range_num) in enumerate(zip(configs, val_stds, test_stds, range_nums)):
            axes[0, 1].annotate(f'{v_std:.4f}', (range_num, v_std), textcoords="offset points",
                               xytext=(0, 10), ha='center', fontsize=9, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='steelblue', alpha=0.4))
            axes[0, 1].annotate(f'{t_std:.4f}', (range_num, t_std), textcoords="offset points",
                               xytext=(0, -15), ha='center', fontsize=9, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='coral', alpha=0.4))
        
        axes[0, 1].set_xlabel('Range', fontsize=11, fontweight='bold')
        axes[0, 1].set_ylabel('Joint Score Std Dev', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('Stability of Representative Configurations (Lower is More Stable)',
                            fontsize=12, fontweight='bold')
        axes[0, 1].set_xticks(range_nums)
        axes[0, 1].set_xticklabels([f"Range {n}" for n in range_nums], fontsize=10)
        axes[0, 1].legend(fontsize=10, loc='best')
        axes[0, 1].grid(True, alpha=0.3)
        
        # ===== Plot 3: Generalization Gap =====
        gap = np.abs(val_means - test_means)
        colors_gap = ['green' if g < 0.01 else 'orange' if g < 0.02 else 'red' for g in gap]
        
        bars = axes[1, 0].bar(x_pos, gap, color=colors_gap, alpha=0.8, edgecolor='black', linewidth=1.5)
        axes[1, 0].axhline(y=gap.mean(), color='red', linestyle='--', linewidth=2,
                          label=f'Mean Gap: {gap.mean():.6f}')
        
        # Annotate gap values with config names
        for i, (g, config, range_num) in enumerate(zip(gap, configs, range_nums)):
            axes[1, 0].annotate(f'{g:.4f}\nRange {range_num}\n{config}', (i, g), textcoords="offset points",
                               xytext=(0, 5), ha='center', fontsize=8, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
        
        axes[1, 0].set_xlabel('Representative Configuration', fontsize=11, fontweight='bold')
        axes[1, 0].set_ylabel('|Val JS - Test JS|', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('Generalization Gap (Lower is Better)',
                            fontsize=12, fontweight='bold')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels([f"R{r['range_num']}\n{r['config']}" for r in ranges], fontsize=9)
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # ===== Plot 4: Configuration Details Table =====
        axes[1, 1].axis('off')
        
        # Create table data with comprehensive metrics
        table_data = []
        table_data.append(['Range', 'Config', 'Val JS', 'Val CI', 'Test JS', 'Test CI', 'Gap', 'Val Std', 'Test Std'])
        
        for i, r in enumerate(ranges):
            # Format CI as [lower, upper]
            val_ci = f"[{val_ci_lowers[i]:.4f}, {val_ci_uppers[i]:.4f}]"
            test_ci = f"[{test_ci_lowers[i]:.4f}, {test_ci_uppers[i]:.4f}]"
            
            row = [
                f"Range {r['range_num']}",
                r['config'],
                f"{val_means[i]:.4f}",
                val_ci,
                f"{test_means[i]:.4f}",
                test_ci,
                f"{gap[i]:.4f}",
                f"{val_stds[i]:.4f}",
                f"{test_stds[i]:.4f}"
            ]
            table_data.append(row)
        
        table = axes[1, 1].table(cellText=table_data, cellLoc='center', loc='center',
                                colWidths=[0.08, 0.10, 0.10, 0.15, 0.10, 0.15, 0.08, 0.10, 0.10])
        
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 2.2)
        
        # Color header
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white', fontsize=8)
        
        # Alternate row colors
        for i in range(1, len(table_data)):
            color = '#f0f0f0' if i % 2 == 0 else 'white'
            for j in range(len(table_data[0])):
                table[(i, j)].set_facecolor(color)
                if j == 1:  # Config column
                    table[(i, j)].set_text_props(weight='bold', fontsize=8)
        
        axes[1, 1].set_title('Representative Configuration Summary (All Metrics)',
                            fontsize=12, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plot_path = self.output_dir / '01b_representative_configs_analysis.png'
        plt.savefig(str(plot_path), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {plot_path}")
    
    def plot_representative_configs_with_ci(self, ranges):
        """
        NEW PLOT: Show representative configurations with 95% CI on val and test sets.
        
        Displays mean JS and 95% CI for each representative configuration,
        with range and configuration name annotations on x-axis.
        """
        print("\n" + "="*80)
        print("Plot (New): Representative Configurations with 95% CI")
        print("="*80)
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        configs = [r['config'] for r in ranges]
        range_nums = [r['range_num'] for r in ranges]
        
        # Get validation metrics for representatives
        val_metrics = []
        for config in configs:
            val_row = self.val_stats[self.val_stats['config'] == config]
            if not val_row.empty:
                val_metrics.append({
                    'js_mean': val_row['js_mean'].values[0],
                    'js_ci_lower': val_row['js_ci_lower'].values[0],
                    'js_ci_upper': val_row['js_ci_upper'].values[0],
                })
        
        # Get test metrics for representatives
        test_metrics = []
        for config in configs:
            test_row = self.test_stats[self.test_stats['config'] == config]
            if not test_row.empty:
                test_metrics.append({
                    'js_mean': test_row['js_mean'].values[0],
                    'js_ci_lower': test_row['js_ci_lower'].values[0],
                    'js_ci_upper': test_row['js_ci_upper'].values[0],
                })
        
        # Prepare data
        x_pos = np.arange(len(configs))
        width = 0.35
        
        val_means = np.array([m['js_mean'] for m in val_metrics])
        val_ci_lowers = np.array([m['js_ci_lower'] for m in val_metrics])
        val_ci_uppers = np.array([m['js_ci_upper'] for m in val_metrics])
        val_errors_lower = val_means - val_ci_lowers
        val_errors_upper = val_ci_uppers - val_means
        
        test_means = np.array([m['js_mean'] for m in test_metrics])
        test_ci_lowers = np.array([m['js_ci_lower'] for m in test_metrics])
        test_ci_uppers = np.array([m['js_ci_upper'] for m in test_metrics])
        test_errors_lower = test_means - test_ci_lowers
        test_errors_upper = test_ci_uppers - test_means
        
        # Plot validation
        ax.errorbar(x_pos - width/2, val_means, 
                   yerr=[val_errors_lower, val_errors_upper],
                   fmt='o', markersize=12, capsize=8, capthick=2,
                   color='steelblue', alpha=0.8, elinewidth=2.5,
                   markeredgecolor='darkblue', markeredgewidth=2,
                   label='Validation JS', linewidth=2.5)
        
        # Plot test
        ax.errorbar(x_pos + width/2, test_means,
                   yerr=[test_errors_lower, test_errors_upper],
                   fmt='s', markersize=12, capsize=8, capthick=2,
                   color='coral', alpha=0.8, elinewidth=2.5,
                   markeredgecolor='darkred', markeredgewidth=2,
                   label='Test JS', linewidth=2.5)
        
        # Annotate values
        for i, (config, range_num) in enumerate(zip(configs, range_nums)):
            # Validation annotation
            ax.annotate(f'{val_means[i]:.4f}', (i - width/2, val_means[i]),
                       textcoords="offset points", xytext=(0, 15), ha='center',
                       fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='steelblue', alpha=0.4))
            
            # Test annotation
            ax.annotate(f'{test_means[i]:.4f}', (i + width/2, test_means[i]),
                       textcoords="offset points", xytext=(0, -20), ha='center',
                       fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='coral', alpha=0.4))
        
        # Set x-axis labels
        x_labels = [f"Range {r['range_num']}\n{r['config']}" for r in ranges]
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Joint Score', fontsize=12, fontweight='bold')
        ax.set_xlabel('Performance Range - Representative Configuration', fontsize=12, fontweight='bold')
        ax.set_title('Representative Configurations: Mean JS with 95% Confidence Intervals',
                    fontsize=14, fontweight='bold', pad=20)
        
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=12, loc='best', ncol=2)
        
        # Add reference lines
        ax.axhline(y=val_means.mean(), color='steelblue', linestyle='--', alpha=0.4,
                  linewidth=2, label=f'Val Mean: {val_means.mean():.4f}')
        ax.axhline(y=test_means.mean(), color='coral', linestyle='--', alpha=0.4,
                  linewidth=2, label=f'Test Mean: {test_means.mean():.4f}')
        
        plt.tight_layout()
        plot_path = self.output_dir / '01c_representative_configs_with_ci.png'
        plt.savefig(str(plot_path), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {plot_path}")
    
    def print_representative_configs_summary(self, ranges):
        """Print summary of representative configurations to terminal."""
        print("\n" + "="*80)
        print("REPRESENTATIVE CONFIGURATIONS SUMMARY")
        print("="*80)
        
        print(f"\nTotal Ranges: {len(ranges)}")
        print(f"Range Interval: 10 configurations per range\n")
        
        print("-" * 120)
        print(f"{'Range':<10} {'Config Positions':<20} {'Rep. Config':<15} {'Val JS':<12} {'Val Std':<12} {'Test JS':<12} {'Test Std':<12} {'Gap':<12}")
        print("-" * 120)
        
        for r in ranges:
            val_row = self.val_stats[self.val_stats['config'] == r['config']].iloc[0]
            test_row = self.test_stats[self.test_stats['config'] == r['config']]
            
            test_js = test_row['js_mean'].values[0] if not test_row.empty else np.nan
            test_std = test_row['js_std'].values[0] if not test_row.empty else np.nan
            gap = abs(r['js_mean'] - test_js) if not np.isnan(test_js) else np.nan
            
            print(f"{r['range_num']:<10} "
                  f"#{r['range_start']:<8} - #{r['range_end']:<8} "
                  f"{r['config']:<15} "
                  f"{r['js_mean']:<12.6f} "
                  f"{r['js_std']:<12.6f} "
                  f"{test_js:<12.6f} "
                  f"{test_std:<12.6f} "
                  f"{gap:<12.6f}")
        
        print("-" * 120)
        print(f"\nKey Findings:")
        print(f"  • Total configurations analyzed: {len(self.val_stats)}")
        print(f"  • Number of ranges: {len(ranges)}")
        print(f"  • Best range (highest mean JS): Range {ranges[0]['range_num']} ({ranges[0]['config']})")
        
        # Find most stable
        most_stable = min(ranges, key=lambda x: x['js_cv'])
        print(f"  • Most stable representative: {most_stable['config']} "
              f"(Range {most_stable['range_num']}, CV={most_stable['js_cv']:.6f})")
        
        # Find smallest gap
        gaps = []
        for r in ranges:
            test_row = self.test_stats[self.test_stats['config'] == r['config']]
            if not test_row.empty:
                test_js = test_row['js_mean'].values[0]
                gap = abs(r['js_mean'] - test_js)
                gaps.append((r['config'], gap, r['range_num']))
        
        if gaps:
            best_gap_config = min(gaps, key=lambda x: x[1])
            print(f"  • Best generalization: {best_gap_config[0]} "
                  f"(Range {best_gap_config[2]}, Gap={best_gap_config[1]:.6f})")
        
        print("\n" + "="*80 + "\n")
    
    def plot_correlation_scatter(self):
        """
        Plot 2: Scatter plot of validation vs test JS with regression line.
        Shows how well validation performance predicts test performance.
        """
        print("\n" + "="*80)
        print("Plot 2: Correlation Analysis (Validation vs Test)")
        print("="*80)
        
        # Merge val and test statistics
        merged = pd.merge(
            self.val_stats[['config', 'js_mean']].rename(columns={'js_mean': 'val_js'}),
            self.test_stats[['config', 'js_mean']].rename(columns={'js_mean': 'test_js'}),
            on='config'
        )
        
        # Compute correlations
        pearson_r, pearson_p = pearsonr(merged['val_js'], merged['test_js'])
        spearman_r, spearman_p = spearmanr(merged['val_js'], merged['test_js'])
        kendall_r, kendall_p = kendalltau(merged['val_js'], merged['test_js'])
        
        print(f"\nCorrelation Analysis:")
        print(f"  Pearson r:  {pearson_r:.6f} (p={pearson_p:.6f})")
        print(f"  Spearman ρ: {spearman_r:.6f} (p={spearman_p:.6f})")
        print(f"  Kendall τ:  {kendall_r:.6f} (p={kendall_p:.6f})")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # ===== Scatter plot =====
        colors = merged['config'].str.extract(r'S(\d+)_T(\d+)').astype(int).sum(axis=1)
        
        scatter = axes[0].scatter(merged['val_js'], merged['test_js'],
                                 c=colors, cmap='viridis', s=200,
                                 alpha=0.6, edgecolors='black', linewidth=1.5)
        
        # Add regression line
        z = np.polyfit(merged['val_js'], merged['test_js'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(merged['val_js'].min(), merged['val_js'].max(), 100)
        axes[0].plot(x_line, p(x_line), "r--", linewidth=2.5, label=f'Linear fit')
        
        # Perfect prediction line
        axes[0].plot([merged['val_js'].min(), merged['val_js'].max()],
                    [merged['val_js'].min(), merged['val_js'].max()],
                    'g--', linewidth=2, alpha=0.5, label='Perfect prediction')
        
        axes[0].set_xlabel('Validation Joint Score', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Test Joint Score', fontsize=12, fontweight='bold')
        axes[0].set_title(f'Validation vs Test Performance\n(Pearson r={pearson_r:.4f}, p={pearson_p:.4f})',
                         fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=10)
        cbar = plt.colorbar(scatter, ax=axes[0])
        cbar.set_label('Total Experts (S+T)', fontsize=11)
        
        # ===== Generalization gap distribution =====
        gap = np.abs(merged['val_js'] - merged['test_js'])
        
        axes[1].hist(gap, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        axes[1].axvline(gap.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'Mean Gap: {gap.mean():.6f}')
        axes[1].axvline(gap.median(), color='green', linestyle='--', linewidth=2,
                       label=f'Median Gap: {gap.median():.6f}')
        axes[1].set_xlabel('|Validation JS - Test JS|', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[1].set_title('Generalization Gap Distribution\n(Lower is better)',
                         fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plot_path = self.output_dir / '02_correlation_scatter.png'
        plt.savefig(str(plot_path), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {plot_path}")
    
    def run_complete_analysis(self, top_k=100):
        """Run all analysis plots and reports."""
        print("\n" + "="*80)
        print("STARTING COMPREHENSIVE VALIDATION-TEST PERFORMANCE ANALYSIS")
        print("="*80)
        
        ranges = self.plot_val_test_comparison_with_ci_refined(top_k=top_k)
        self.plot_representative_configs_analysis(ranges)
        self.plot_representative_configs_with_ci(ranges)
        self.print_representative_configs_summary(ranges)
        self.plot_correlation_scatter()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"✓ All results saved to: {self.output_dir}")
        print("\nGenerated files:")
        print("  1. 01_val_test_comparison_with_ci_refined.png")
        print("     → Shows all 100 configurations with ranges and representatives")
        print("  2. 01b_representative_configs_analysis.png")
        print("     → Detailed line plots and analysis of representative configs")
        print("  3. 01c_representative_configs_with_ci.png")
        print("     → Mean JS with 95% CI for representative configurations")
        print("  4. 02_correlation_scatter.png")
        print("     → Correlation analysis between validation and test")
        print("="*80 + "\n")


# Main execution
if __name__ == '__main__':
    analyzer = PerformanceAnalyzer(
        results_dir='multirun/multirun_validation_results',
        output_dir='multirun/multirun_validation_results/analysis_v2'
    )
    
    analyzer.run_complete_analysis(top_k=100)