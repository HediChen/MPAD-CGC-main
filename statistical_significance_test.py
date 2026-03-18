'''
Statistical Significance Testing for MPAD-CGC Expert Configuration Optimization
================================================================================
This script performs comprehensive statistical significance testing to validate
the difference between expert configurations (shared and task experts), addressing
peer-review comments about:
  1. Running experiments multiple times (5+ runs) with different random seeds
  2. Reporting confidence intervals (95% CI)
  3. Performing statistical significance tests (paired t-test, ANOVA)
  4. Computing effect sizes (Cohen's d)
  5. Bootstrap analysis for robustness

The script generates:
  - Multi-run Joint Score distributions
  - Paired t-test results between top configurations
  - ANOVA test across top configurations
  - 95% confidence intervals
  - Effect size analysis
  - Bootstrap confidence intervals
  - Comprehensive statistical report
  - Visualization plots
'''

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import t, f_oneway, shapiro, levene
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types."""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class MultiRunStatisticalAnalyzer:
    """
    Comprehensive statistical analyzer for multi-run expert configuration experiments.
    
    Attributes:
        results_dict: {config_tuple: {run_id: metrics_dict}}
        confidence_level: Confidence level for CI (default: 0.95 for 95% CI)
        n_bootstrap_samples: Number of bootstrap samples (default: 10000)
    """
    
    def __init__(self, confidence_level=0.95, n_bootstrap_samples=10000):
        self.results_dict = defaultdict(lambda: defaultdict(dict))
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.n_bootstrap_samples = n_bootstrap_samples
        self.statistical_results = {}
        self.confidence_intervals = {}
        self.effect_sizes = {}
        self.pairwise_comparisons = []
        self.anova_results = {}
        
    def add_run_result(self, config_tuple, run_id, joint_score, avg_accuracy, avg_auc, 
                       std_accuracy, std_auc):
        """
        Add results from a single run of a specific configuration.
        
        Parameters:
            config_tuple: (num_shared_experts, num_task_experts)
            run_id: Run number (1, 2, 3, ...)
            joint_score: Joint Score value
            avg_accuracy: Average accuracy across tasks
            avg_auc: Average AUC across tasks
            std_accuracy: Std dev of accuracy across tasks
            std_auc: Std dev of AUC across tasks
        """
        self.results_dict[config_tuple][run_id] = {
            'joint_score': float(joint_score),
            'avg_accuracy': float(avg_accuracy),
            'avg_auc': float(avg_auc),
            'std_accuracy': float(std_accuracy),
            'std_auc': float(std_auc)
        }
    
    def load_from_csv(self, csv_file):
        """
        Load multi-run results from CSV file.
        Expected CSV columns: shared_experts, task_experts, run_id, joint_score, 
                             avg_accuracy, avg_auc, std_accuracy, std_auc
        """
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            config = (int(row['shared_experts']), int(row['task_experts']))
            run_id = int(row['run_id'])
            self.add_run_result(
                config, run_id,
                float(row['joint_score']),
                float(row['avg_accuracy']),
                float(row['avg_auc']),
                float(row['std_accuracy']),
                float(row['std_auc'])
            )
    
    def get_config_statistics(self):
        """
        Compute summary statistics for each configuration across all runs.
        
        Returns:
            pd.DataFrame with columns: shared_experts, task_experts, 
                     mean_js, std_js, median_js, min_js, max_js, cv_js,
                     ci_lower_95, ci_upper_95, sem, n_runs
        """
        stats_list = []
        
        for config, runs_data in sorted(self.results_dict.items()):
            shared, task = config
            joint_scores = [data['joint_score'] for data in runs_data.values()]
            
            if len(joint_scores) == 0:
                continue
            
            joint_scores_arr = np.array(joint_scores)
            n_runs = len(joint_scores)
            mean_js = np.mean(joint_scores_arr)
            std_js = np.std(joint_scores_arr, ddof=1)
            median_js = np.median(joint_scores_arr)
            min_js = np.min(joint_scores_arr)
            max_js = np.max(joint_scores_arr)
            cv_js = std_js / mean_js if mean_js != 0 else 0  # Coefficient of variation
            sem = std_js / np.sqrt(n_runs)  # Standard error of mean
            
            # 95% Confidence interval using t-distribution
            ci_lower, ci_upper = self._compute_ci_t(joint_scores_arr, self.confidence_level)
            
            stats_list.append({
                'shared_experts': int(shared),
                'task_experts': int(task),
                'config': f"S{shared}_T{task}",
                'mean_js': float(mean_js),
                'std_js': float(std_js),
                'median_js': float(median_js),
                'min_js': float(min_js),
                'max_js': float(max_js),
                'cv_js': float(cv_js),
                'ci_lower_95': float(ci_lower),
                'ci_upper_95': float(ci_upper),
                'ci_width': float(ci_upper - ci_lower),
                'sem': float(sem),
                'n_runs': int(n_runs)
            })
        
        df_stats = pd.DataFrame(stats_list)
        df_stats = df_stats.sort_values('mean_js', ascending=False)
        self.config_stats = df_stats
        return df_stats
    
    def _compute_ci_t(self, data, confidence_level):
        """Compute confidence interval using t-distribution."""
        n = len(data)
        mean = np.mean(data)
        std_err = stats.sem(data)
        dof = n - 1
        t_critical = t.ppf((1 + confidence_level) / 2, dof)
        margin = t_critical * std_err
        return float(mean - margin), float(mean + margin)
    
    def _compute_ci_bootstrap(self, data, confidence_level, n_samples=None):
        """Compute confidence interval using bootstrap method."""
        if n_samples is None:
            n_samples = self.n_bootstrap_samples
        
        bootstrap_means = []
        for _ in range(n_samples):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        alpha = 1 - confidence_level
        ci_lower = np.percentile(bootstrap_means, alpha/2 * 100)
        ci_upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
        
        return float(ci_lower), float(ci_upper), np.array(bootstrap_means)
    
    def perform_paired_ttest(self, config1, config2, metric='joint_score'):
        """
        Perform paired t-test between two configurations.
        
        Parameters:
            config1, config2: Configuration tuples (shared, task)
            metric: Metric to compare ('joint_score', 'avg_accuracy', 'avg_auc')
        
        Returns:
            dict with test results including t-statistic, p-value, mean difference, effect size
        """
        data1 = [data[metric] for data in self.results_dict[config1].values()]
        data2 = [data[metric] for data in self.results_dict[config2].values()]
        
        if len(data1) != len(data2):
            print(f"Warning: Different number of runs ({len(data1)} vs {len(data2)})")
        
        # Perform paired t-test (matched samples)
        # If different number of runs, use minimum length
        min_len = min(len(data1), len(data2))
        data1 = np.array(data1[:min_len])
        data2 = np.array(data2[:min_len])
        
        t_stat, p_value = stats.ttest_rel(data1, data2)
        
        # Compute mean difference
        diff = data1 - data2
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)
        
        # Compute Cohen's d
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0
        
        # Compute 95% CI for mean difference
        ci_lower, ci_upper = self._compute_ci_t(diff, self.confidence_level)
        
        # Determine statistical significance
        is_significant = bool(p_value < self.alpha)
        
        result = {
            'config1': config1,
            'config2': config2,
            'metric': metric,
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'mean_config1': float(np.mean(data1)),
            'mean_config2': float(np.mean(data2)),
            'mean_difference': float(mean_diff),
            'std_difference': float(std_diff),
            'cohens_d': float(cohens_d),
            'cohens_d_interpretation': self._interpret_cohens_d(cohens_d),
            'ci_lower_diff': float(ci_lower),
            'ci_upper_diff': float(ci_upper),
            'is_significant': is_significant,
            'significance_level': float(self.alpha),
            'n_pairs': int(len(data1)),
            'effect_size_category': self._interpret_cohens_d(cohens_d)
        }
        
        return result
    
    def perform_anova(self, configs_list, metric='joint_score'):
        """
        Perform one-way ANOVA across multiple configurations.
        
        Parameters:
            configs_list: List of configuration tuples to compare
            metric: Metric to compare
        
        Returns:
            dict with ANOVA results including F-statistic, p-value, effect size
        """
        groups = []
        for config in configs_list:
            data = [data[metric] for data in self.results_dict[config].values()]
            groups.append(np.array(data))
        
        # Perform ANOVA
        f_stat, p_value = f_oneway(*groups)
        
        # Compute effect size (eta-squared)
        all_data = np.concatenate(groups)
        grand_mean = np.mean(all_data)
        
        ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 for group in groups)
        ss_total = sum((x - grand_mean)**2 for x in all_data)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        # Determine significance
        is_significant = bool(p_value < self.alpha)
        
        result = {
            'metric': metric,
            'configs': configs_list,
            'n_configs': len(configs_list),
            'f_statistic': float(f_stat),
            'p_value': float(p_value),
            'eta_squared': float(eta_squared),
            'eta_squared_interpretation': self._interpret_eta_squared(eta_squared),
            'is_significant': is_significant,
            'significance_level': float(self.alpha),
            'group_means': {str(config): float(np.mean(groups[i])) for i, config in enumerate(configs_list)},
            'group_stds': {str(config): float(np.std(groups[i], ddof=1)) for i, config in enumerate(configs_list)}
        }
        
        return result
    
    def perform_all_pairwise_tests(self, top_k=5, metric='joint_score'):
        """
        Perform paired t-tests between all pairs of top configurations.
        
        Parameters:
            top_k: Number of top configurations to compare
            metric: Metric to compare
        
        Returns:
            List of pairwise comparison results
        """
        if not hasattr(self, 'config_stats'):
            self.get_config_statistics()
        
        top_configs = self.config_stats.head(top_k)[['shared_experts', 'task_experts']].values.tolist()
        top_configs = [tuple(row) for row in top_configs]
        
        pairwise_results = []
        for i, config1 in enumerate(top_configs):
            for config2 in top_configs[i+1:]:
                result = self.perform_paired_ttest(config1, config2, metric)
                pairwise_results.append(result)
        
        self.pairwise_comparisons = pairwise_results
        return pairwise_results
    
    def compute_bootstrap_ci_for_configs(self, configs_list, metric='joint_score'):
        """
        Compute bootstrap confidence intervals for specified configurations.
        
        Parameters:
            configs_list: List of configuration tuples
            metric: Metric to analyze
        
        Returns:
            dict mapping config -> bootstrap CI results
        """
        bootstrap_results = {}
        
        for config in configs_list:
            data = np.array([d[metric] for d in self.results_dict[config].values()])
            ci_lower, ci_upper, bootstrap_dist = self._compute_ci_bootstrap(
                data, self.confidence_level
            )
            
            bootstrap_results[config] = {
                'mean': float(np.mean(data)),
                'std': float(np.std(data, ddof=1)),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper),
                'bootstrap_distribution': bootstrap_dist.tolist(),
                'n_samples': int(len(data))
            }
        
        return bootstrap_results
    
    def _interpret_cohens_d(self, d):
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return 'negligible'
        elif abs_d < 0.5:
            return 'small'
        elif abs_d < 0.8:
            return 'medium'
        else:
            return 'large'
    
    def _interpret_eta_squared(self, eta_sq):
        """Interpret eta-squared effect size."""
        if eta_sq < 0.01:
            return 'negligible'
        elif eta_sq < 0.06:
            return 'small'
        elif eta_sq < 0.14:
            return 'medium'
        else:
            return 'large'
    
    def generate_statistical_report(self, output_file='statistical_analysis_report.txt'):
        """
        Generate comprehensive statistical analysis report.
        
        Parameters:
            output_file: Path to save the report
        """
        report_lines = []
        
        report_lines.append("="*80)
        report_lines.append("STATISTICAL SIGNIFICANCE TESTING FOR EXPERT CONFIGURATION OPTIMIZATION")
        report_lines.append("="*80)
        report_lines.append(f"\nReport Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Confidence Level: {self.confidence_level*100:.0f}%")
        report_lines.append(f"Significance Level (alpha): {self.alpha}")
        report_lines.append(f"Bootstrap Samples: {self.n_bootstrap_samples}")
        
        # Section 1: Configuration Statistics
        report_lines.append("\n" + "="*80)
        report_lines.append("1. CONFIGURATION STATISTICS (Multi-Run Summary)")
        report_lines.append("="*80)
        
        if hasattr(self, 'config_stats'):
            config_stats = self.config_stats
            report_lines.append(f"\nTop 10 Configurations by Mean Joint Score:\n")
            for idx, row in config_stats.head(10).iterrows():
                report_lines.append(
                    f"  Rank {idx+1}: S{int(row['shared_experts'])}_T{int(row['task_experts'])} "
                    f"-> Mean JS: {row['mean_js']:.6f} +/- {row['std_js']:.6f} "
                    f"[95% CI: {row['ci_lower_95']:.6f}, {row['ci_upper_95']:.6f}] "
                    f"(n={int(row['n_runs'])}, CV={row['cv_js']:.4f})"
                )
        
        # Section 2: Pairwise Comparisons
        report_lines.append("\n" + "="*80)
        report_lines.append("2. PAIRWISE T-TEST COMPARISONS (Top Configurations)")
        report_lines.append("="*80)
        
        if self.pairwise_comparisons:
            for result in self.pairwise_comparisons:
                config1 = f"S{result['config1'][0]}_T{result['config1'][1]}"
                config2 = f"S{result['config2'][0]}_T{result['config2'][1]}"
                report_lines.append(f"\n{config1} vs {config2}:")
                report_lines.append(f"  Mean {config1}: {result['mean_config1']:.6f}")
                report_lines.append(f"  Mean {config2}: {result['mean_config2']:.6f}")
                report_lines.append(f"  Mean Difference: {result['mean_difference']:.6f}")
                report_lines.append(f"  95% CI (difference): [{result['ci_lower_diff']:.6f}, {result['ci_upper_diff']:.6f}]")
                report_lines.append(f"  t-statistic: {result['t_statistic']:.6f}")
                sig_marker = '***' if result['p_value'] < 0.001 else '**' if result['p_value'] < 0.01 else '*' if result['p_value'] < 0.05 else '(NS)'
                report_lines.append(f"  p-value: {result['p_value']:.6f} {sig_marker}")
                report_lines.append(f"  Cohen's d: {result['cohens_d']:.4f} ({result['cohens_d_interpretation']} effect)")
                report_lines.append(f"  Statistically Significant: {'Yes' if result['is_significant'] else 'No'}")
                report_lines.append(f"  Number of pairs: {result['n_pairs']}")
        
        # Section 3: ANOVA Results
        report_lines.append("\n" + "="*80)
        report_lines.append("3. ONE-WAY ANOVA (Top 5 Configurations)")
        report_lines.append("="*80)
        
        if 'top_5_anova' in self.statistical_results:
            anova = self.statistical_results['top_5_anova']
            report_lines.append(f"\nComparing {anova['n_configs']} configurations:")
            report_lines.append(f"  F-statistic: {anova['f_statistic']:.6f}")
            sig_marker = '***' if anova['p_value'] < 0.001 else '**' if anova['p_value'] < 0.01 else '*' if anova['p_value'] < 0.05 else '(NS)'
            report_lines.append(f"  p-value: {anova['p_value']:.6f} {sig_marker}")
            report_lines.append(f"  Eta-squared (effect size): {anova['eta_squared']:.6f} ({anova['eta_squared_interpretation']} effect)")
            report_lines.append(f"  Statistically Significant: {'Yes' if anova['is_significant'] else 'No'}")
            report_lines.append(f"\n  Group Means:")
            for config, mean in anova['group_means'].items():
                std = anova['group_stds'][config]
                report_lines.append(f"    {config}: {mean:.6f} +/- {std:.6f}")
        
        # Section 4: Key Findings
        report_lines.append("\n" + "="*80)
        report_lines.append("4. KEY FINDINGS & RECOMMENDATIONS")
        report_lines.append("="*80)
        
        if hasattr(self, 'config_stats'):
            best_config = self.config_stats.iloc[0]
            second_config = self.config_stats.iloc[1] if len(self.config_stats) > 1 else None
            
            report_lines.append(f"\nBest Configuration: S{int(best_config['shared_experts'])}_T{int(best_config['task_experts'])}")
            report_lines.append(f"  - Mean Joint Score: {best_config['mean_js']:.6f} +/- {best_config['std_js']:.6f}")
            report_lines.append(f"  - 95% CI: [{best_config['ci_lower_95']:.6f}, {best_config['ci_upper_95']:.6f}]")
            report_lines.append(f"  - Coefficient of Variation: {best_config['cv_js']:.4f}")
            report_lines.append(f"  - Runs: {int(best_config['n_runs'])}")
            
            if second_config is not None:
                report_lines.append(f"\nSecond-Best Configuration: S{int(second_config['shared_experts'])}_T{int(second_config['task_experts'])}")
                report_lines.append(f"  - Mean Joint Score: {second_config['mean_js']:.6f} +/- {second_config['std_js']:.6f}")
                report_lines.append(f"  - 95% CI: [{second_config['ci_lower_95']:.6f}, {second_config['ci_upper_95']:.6f}]")
                
                # Check if CIs overlap
                ci_overlap = not (best_config['ci_lower_95'] > second_config['ci_upper_95'] or 
                                 second_config['ci_lower_95'] > best_config['ci_upper_95'])
                report_lines.append(f"\nConfidence Interval Overlap: {'Yes (overlapping)' if ci_overlap else 'No (non-overlapping)'}")
                
                if self.pairwise_comparisons:
                    for result in self.pairwise_comparisons:
                        if ((result['config1'][0] == best_config['shared_experts'] and 
                             result['config1'][1] == best_config['task_experts']) and
                            (result['config2'][0] == second_config['shared_experts'] and 
                             result['config2'][1] == second_config['task_experts'])):
                            report_lines.append(f"\nPairwise Comparison (Best vs Second-Best):")
                            report_lines.append(f"  - t-statistic: {result['t_statistic']:.6f}")
                            report_lines.append(f"  - p-value: {result['p_value']:.6f}")
                            report_lines.append(f"  - Cohen's d: {result['cohens_d']:.4f} ({result['cohens_d_interpretation']} effect)")
                            report_lines.append(f"  - Statistically Significant: {'YES (p < 0.05)' if result['is_significant'] else 'NO (p >= 0.05)'}")
        
        report_lines.append("\n" + "="*80)
        report_lines.append("INTERPRETATION GUIDELINES")
        report_lines.append("="*80)
        report_lines.append("\n- Cohen's d Effect Size:")
        report_lines.append("  * Negligible: |d| < 0.2")
        report_lines.append("  * Small: 0.2 <= |d| < 0.5")
        report_lines.append("  * Medium: 0.5 <= |d| < 0.8")
        report_lines.append("  * Large: |d| >= 0.8")
        report_lines.append("\n- Eta-squared (eta^2) Effect Size:")
        report_lines.append("  * Negligible: eta^2 < 0.01")
        report_lines.append("  * Small: 0.01 <= eta^2 < 0.06")
        report_lines.append("  * Medium: 0.06 <= eta^2 < 0.14")
        report_lines.append("  * Large: eta^2 >= 0.14")
        report_lines.append("\n- Statistical Significance:")
        report_lines.append("  * p < 0.001: Highly significant ***")
        report_lines.append("  * p < 0.01: Very significant **")
        report_lines.append("  * p < 0.05: Significant *")
        report_lines.append("  * p >= 0.05: Not significant (NS)")
        report_lines.append("\n- Confidence Interval Interpretation:")
        report_lines.append("  * Non-overlapping CIs suggest statistically significant difference")
        report_lines.append("  * Overlapping CIs may indicate non-significant difference")
        
        report_text = "\n".join(report_lines)
        
        # Use UTF-8 encoding
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\n✅ Report saved to: {output_file}")
        
        return report_text
    
    def plot_joint_score_distributions(self, top_k=5, save_path=None):
        """
        Plot Joint Score distributions for top configurations across runs.
        """
        if not hasattr(self, 'config_stats'):
            self.get_config_statistics()
        
        top_configs = self.config_stats.head(top_k)[['shared_experts', 'task_experts']].values.tolist()
        top_configs = [tuple(row) for row in top_configs]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Box plot
        data_for_box = []
        labels_for_box = []
        for config in top_configs:
            data = [d['joint_score'] for d in self.results_dict[config].values()]
            data_for_box.append(data)
            labels_for_box.append(f"S{config[0]}_T{config[1]}")
        
        bp = axes[0].boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
        for patch, color in zip(bp['boxes'], sns.color_palette("husl", top_k)):
            patch.set_facecolor(color)
        axes[0].set_ylabel('Joint Score')
        axes[0].set_title(f'Joint Score Distribution - Top {top_k} Configurations')
        axes[0].grid(True, alpha=0.3)
        
        # Violin plot with confidence intervals
        positions = range(len(top_configs))
        for i, config in enumerate(top_configs):
            data = np.array([d['joint_score'] for d in self.results_dict[config].values()])
            mean = np.mean(data)
            ci_lower, ci_upper = self._compute_ci_t(data, self.confidence_level)
            
            # Plot point estimate and CI
            axes[1].errorbar(i, mean, yerr=[[mean - ci_lower], [ci_upper - mean]], 
                           fmt='o', markersize=8, capsize=5, capthick=2, alpha=0.7)
            axes[1].scatter(i, mean, s=100, zorder=5)
        
        axes[1].set_xticks(positions)
        axes[1].set_xticklabels(labels_for_box)
        axes[1].set_ylabel('Joint Score')
        axes[1].set_title(f'Mean Joint Score with 95% CI - Top {top_k} Configurations')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plot saved to: {save_path}")
        plt.show()
    
    def plot_pairwise_comparison(self, save_path=None):
        """
        Plot pairwise comparison results (p-values and effect sizes).
        """
        if not self.pairwise_comparisons:
            print("No pairwise comparisons available. Run perform_all_pairwise_tests() first.")
            return
        
        comparisons = []
        p_values = []
        cohens_ds = []
        
        for result in self.pairwise_comparisons:
            config1 = f"S{result['config1'][0]}_T{result['config1'][1]}"
            config2 = f"S{result['config2'][0]}_T{result['config2'][1]}"
            comparisons.append(f"{config1} vs\n{config2}")
            p_values.append(result['p_value'])
            cohens_ds.append(abs(result['cohens_d']))
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # p-values plot
        colors_p = ['red' if p < 0.05 else 'green' for p in p_values]
        axes[0].barh(range(len(comparisons)), p_values, color=colors_p, alpha=0.7)
        axes[0].axvline(x=0.05, color='black', linestyle='--', linewidth=2, label='alpha = 0.05')
        axes[0].set_yticks(range(len(comparisons)))
        axes[0].set_yticklabels(comparisons, fontsize=9)
        axes[0].set_xlabel('p-value')
        axes[0].set_title('Pairwise t-test p-values\n(Red: significant, Green: non-significant)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # Effect sizes plot
        colors_d = ['green' if d < 0.2 else 'blue' if d < 0.5 else 'orange' if d < 0.8 else 'red' 
                   for d in cohens_ds]
        axes[1].barh(range(len(comparisons)), cohens_ds, color=colors_d, alpha=0.7)
        axes[1].axvline(x=0.2, color='black', linestyle='--', linewidth=1, alpha=0.5)
        axes[1].axvline(x=0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
        axes[1].axvline(x=0.8, color='black', linestyle='--', linewidth=1, alpha=0.5)
        axes[1].set_yticks(range(len(comparisons)))
        axes[1].set_yticklabels(comparisons, fontsize=9)
        axes[1].set_xlabel("|Cohen's d|")
        axes[1].set_title("Effect Sizes (Cohen's d)\nGreen: negligible, Blue: small, Orange: medium, Red: large")
        axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plot saved to: {save_path}")
        plt.show()
    
    def plot_bootstrap_ci(self, configs_list, save_path=None):
        """
        Plot bootstrap confidence intervals for specified configurations.
        """
        bootstrap_results = self.compute_bootstrap_ci_for_configs(configs_list, 'joint_score')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, config in enumerate(configs_list):
            result = bootstrap_results[config]
            mean = result['mean']
            ci_lower = result['ci_lower']
            ci_upper = result['ci_upper']
            
            # Plot bootstrap distribution as scatter
            ax.scatter([i]*len(result['bootstrap_distribution']), 
                      result['bootstrap_distribution'], alpha=0.1, s=10)
            
            # Plot CI
            ax.errorbar(i, mean, yerr=[[mean - ci_lower], [ci_upper - mean]], 
                       fmt='o', markersize=10, capsize=5, capthick=2, color='red', 
                       label='95% Bootstrap CI' if i == 0 else '')
            ax.scatter(i, mean, s=100, color='red', zorder=5)
        
        ax.set_xticks(range(len(configs_list)))
        ax.set_xticklabels([f"S{c[0]}_T{c[1]}" for c in configs_list])
        ax.set_ylabel('Joint Score')
        ax.set_title('Bootstrap Confidence Intervals (10,000 samples)')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plot saved to: {save_path}")
        plt.show()
    
    def save_statistical_results_to_json(self, output_file='statistical_results.json'):
        """Save all statistical results to JSON file."""
        # Convert config_stats to serializable format
        config_stats_dict = None
        if hasattr(self, 'config_stats'):
            config_stats_dict = self.config_stats.to_dict('records')
        
        results_to_save = {
            'config_statistics': config_stats_dict,
            'pairwise_comparisons': self.pairwise_comparisons,
            'anova_results': self.statistical_results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Use custom encoder for numpy types
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        print(f"✅ Results saved to: {output_file}")


# Example usage function
def run_comprehensive_statistical_analysis(csv_file, output_dir='statistical_analysis_results'):
    """
    Run comprehensive statistical analysis on multi-run results.
    
    Parameters:
        csv_file: Path to CSV file with multi-run results
        output_dir: Directory to save all results and plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize analyzer
    analyzer = MultiRunStatisticalAnalyzer(confidence_level=0.95, n_bootstrap_samples=10000)
    
    # Load results
    print("Loading multi-run results...")
    analyzer.load_from_csv(csv_file)
    
    # Compute statistics
    print("Computing configuration statistics...")
    config_stats = analyzer.get_config_statistics()
    print("\nTop 10 Configurations:")
    print(config_stats.head(10)[['shared_experts', 'task_experts', 'mean_js', 'std_js', 'ci_lower_95', 'ci_upper_95']])
    
    # Perform pairwise tests
    print("\nPerforming pairwise t-tests on top 5 configurations...")
    pairwise_results = analyzer.perform_all_pairwise_tests(top_k=5)
    
    # Perform ANOVA
    print("\nPerforming ANOVA on top 5 configurations...")
    top_5_configs = config_stats.head(5)[['shared_experts', 'task_experts']].values.tolist()
    top_5_configs = [tuple(row) for row in top_5_configs]
    anova_results = analyzer.perform_anova(top_5_configs)
    analyzer.statistical_results['top_5_anova'] = anova_results
    
    # Generate report
    print("\nGenerating statistical report...")
    analyzer.generate_statistical_report(
        output_file=os.path.join(output_dir, 'statistical_analysis_report.txt')
    )
    
    # Create plots
    print("\nCreating visualization plots...")
    analyzer.plot_joint_score_distributions(
        top_k=5,
        save_path=os.path.join(output_dir, 'joint_score_distributions.png')
    )
    
    analyzer.plot_pairwise_comparison(
        save_path=os.path.join(output_dir, 'pairwise_comparisons.png')
    )
    
    analyzer.plot_bootstrap_ci(
        configs_list=top_5_configs,
        save_path=os.path.join(output_dir, 'bootstrap_confidence_intervals.png')
    )
    
    # Save results
    config_stats.to_csv(os.path.join(output_dir, 'configuration_statistics.csv'), index=False)
    analyzer.save_statistical_results_to_json(
        output_file=os.path.join(output_dir, 'statistical_results.json')
    )
    
    print(f"\n✅ All results saved to: {output_dir}")
    
    return analyzer


if __name__ == '__main__':
    from run_multirun_experiments import MultiRunExperimentRunner

    # runner = MultiRunExperimentRunner(num_runs=5)
    # csv_path, df_results = runner.run_full_pipeline()
    # Example: Run analysis on multi-run CSV file
    csv_path = './multirun_results/multi_run_results.csv'
    analyzer = run_comprehensive_statistical_analysis(csv_path)
    
    print("Statistical Significance Testing Module for MPAD-CGC Configuration Optimization")
    print("Run with: analyzer = run_comprehensive_statistical_analysis('your_csv_file.csv')")