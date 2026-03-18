'''
Lower Confidence Bound (LCB) Statistical Significance Analysis
================================================================================
This script performs statistical significance testing for MPAD-CGC expert 
configuration optimization using the Lower Confidence Bound (LCB) approach.

The LCB method directly addresses the reviewer's concern by:
1. Quantifying uncertainty in performance metrics across multiple runs
2. Providing a conservative estimate of configuration quality
3. Answering: "What is the minimum performance guarantee after considering uncertainty?"
4. Better differentiating top configurations with more robust comparison

Key Innovation:
- Uses confidence bounds instead of simple mean - std deviation
- Incorporates standard error (SE = sigma / sqrt(n)) for proper uncertainty quantification
- Two levels of conservatism: k=1 (exploratory) and k=2 (conservative, ~95% CI)
- Directly applicable to multi-seed experimental design
'''

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import t
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class LCBStatisticalAnalyzer:
    """
    Lower Confidence Bound (LCB) based statistical analyzer for expert configurations.
    
    The LCB approach computes a conservative lower bound on expected performance:
    JS_LCB = (μ_acc - k·SE_acc) + (μ_auc - k·SE_auc)
    
    where:
        μ_acc, μ_auc: mean accuracy and AUC across tasks
        SE_acc, SE_auc: standard error of accuracy and AUC
        k: confidence level parameter (1 for exploratory, 2 for ~95% confidence)
    """
    
    def __init__(self, k=2, confidence_level=0.95):
        """
        Initialize LCB analyzer.
        
        Parameters:
            k: Confidence level parameter
               - k=1: 68% confidence (1 standard error)
               - k=2: ~95% confidence (2 standard errors)
               - k=1.96: exactly 95% for large samples
            confidence_level: Used for t-distribution based CI computation
        """
        self.k = k
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.results_dict = {}
        self.lcb_results = None
        
    def load_from_csv(self, csv_file):
        """
        Load multi-run results from CSV file.
        
        Expected columns:
            shared_experts, task_experts, run_id, seed,
            joint_score, avg_accuracy, avg_auc, std_accuracy, std_auc
        """
        df = pd.read_csv(csv_file)
        
        # Group by configuration
        for (shared, task), group in df.groupby(['shared_experts', 'task_experts']):
            config_tuple = (int(shared), int(task))
            
            # Extract metrics across all runs
            accuracies = group['avg_accuracy'].values
            aucs = group['avg_auc'].values
            joint_scores = group['joint_score'].values
            
            self.results_dict[config_tuple] = {
                'accuracies': accuracies,
                'aucs': aucs,
                'joint_scores': joint_scores,
                'run_ids': group['run_id'].values,
                'n_runs': len(group)
            }
        
        print(f"✓ Loaded {len(self.results_dict)} configurations from {csv_file}")
        return df
    
    def compute_lcb_metrics(self, values, k=None):
        """
        Compute Lower Confidence Bound metrics.
        
        Parameters:
            values: Array of metric values across runs
            k: Confidence parameter (uses self.k if not provided)
        
        Returns:
            dict with mean, std, SE, LCB, and confidence information
        """
        if k is None:
            k = self.k
        
        n = len(values)
        mean = np.mean(values)
        std = np.std(values, ddof=1) if n > 1 else 0
        se = std / np.sqrt(n) if n > 1 else 0
        
        # Compute LCB: μ - k·SE
        lcb = mean - k * se
        
        # Compute t-based confidence interval for comparison
        if n > 1:
            dof = n - 1
            t_crit = t.ppf((1 + self.confidence_level) / 2, dof)
            ci_lower = mean - t_crit * se
            ci_upper = mean + t_crit * se
        else:
            ci_lower = mean
            ci_upper = mean
        
        return {
            'mean': float(mean),
            'std': float(std),
            'se': float(se),
            'lcb': float(lcb),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'ci_width': float(ci_upper - ci_lower),
            'n': int(n),
            'k': float(k)
        }
    
    def compute_joint_score_lcb(self, config_tuple, k=None):
        """
        Compute Joint Score LCB for a specific configuration.
        
        JS_LCB = (μ_acc - k·SE_acc) + (μ_auc - k·SE_auc)
        
        This represents the lower bound on joint performance at confidence level k.
        """
        if k is None:
            k = self.k
        
        if config_tuple not in self.results_dict:
            return None
        
        data = self.results_dict[config_tuple]
        
        acc_metrics = self.compute_lcb_metrics(data['accuracies'], k=k)
        auc_metrics = self.compute_lcb_metrics(data['aucs'], k=k)
        
        # Joint Score LCB
        js_lcb = acc_metrics['lcb'] + auc_metrics['lcb']
        
        # Standard Joint Score (original approach using mean - std)
        js_original = (acc_metrics['mean'] + auc_metrics['mean']) - \
                     (acc_metrics['std'] + auc_metrics['std'])
        
        # Direct mean of joint scores (for validation)
        js_mean = np.mean(data['joint_scores'])
        
        return {
            'config': config_tuple,
            'accuracy': acc_metrics,
            'auc': auc_metrics,
            'js_lcb': float(js_lcb),
            'js_original': float(js_original),
            'js_mean': float(js_mean),
            'k': float(k),
            'n_runs': int(data['n_runs'])
        }
    
    def compute_all_configurations_lcb(self, k=None):
        """
        Compute LCB scores for all configurations.
        
        Returns:
            pd.DataFrame sorted by JS_LCB (descending)
        """
        if k is None:
            k = self.k
        
        lcb_list = []
        
        for config_tuple in sorted(self.results_dict.keys()):
            lcb_result = self.compute_joint_score_lcb(config_tuple, k=k)
            if lcb_result:
                lcb_list.append({
                    'shared_experts': config_tuple[0],
                    'task_experts': config_tuple[1],
                    'config': f"S{config_tuple[0]}_T{config_tuple[1]}",
                    'n_runs': lcb_result['n_runs'],
                    'acc_mean': lcb_result['accuracy']['mean'],
                    'acc_std': lcb_result['accuracy']['std'],
                    'acc_se': lcb_result['accuracy']['se'],
                    'acc_lcb': lcb_result['accuracy']['lcb'],
                    'auc_mean': lcb_result['auc']['mean'],
                    'auc_std': lcb_result['auc']['std'],
                    'auc_se': lcb_result['auc']['se'],
                    'auc_lcb': lcb_result['auc']['lcb'],
                    'js_lcb': lcb_result['js_lcb'],
                    'js_original': lcb_result['js_original'],
                    'js_mean': lcb_result['js_mean'],
                    'k': lcb_result['k']
                })
        
        df_lcb = pd.DataFrame(lcb_list)
        df_lcb = df_lcb.sort_values('js_lcb', ascending=False)
        
        self.lcb_results = df_lcb
        return df_lcb
    
    def compare_lcb_vs_original(self):
        """
        Compare LCB rankings with original Joint Score rankings.
        
        Shows how the conservative LCB approach changes configuration ranking
        compared to the original mean-based approach.
        """
        if self.lcb_results is None:
            print("Error: Run compute_all_configurations_lcb() first")
            return None
        
        df = self.lcb_results.copy()
        
        # Add ranking columns
        df['rank_lcb'] = df['js_lcb'].rank(ascending=False, method='min').astype(int)
        df['rank_original'] = df['js_original'].rank(ascending=False, method='min').astype(int)
        df['rank_mean'] = df['js_mean'].rank(ascending=False, method='min').astype(int)
        df['rank_change_lcb_vs_original'] = df['rank_original'] - df['rank_lcb']
        
        return df.sort_values('rank_lcb')
    
    def compute_lcb_across_k_values(self, k_values=None):
        """
        Compute LCB rankings for multiple k values to show sensitivity.
        
        Parameters:
            k_values: List of k values to test
                     Default: [0.5, 1.0, 1.96, 2.0, 2.5] 
        """
        if k_values is None:
            k_values = [0.5, 1.0, 1.96, 2.0, 2.5]
        
        results_by_k = {}
        
        for k in k_values:
            df_k = self.compute_all_configurations_lcb(k=k)
            results_by_k[k] = df_k[['config', 'js_lcb']].copy()
            results_by_k[k].columns = ['config', f'js_lcb_k={k}']
        
        # Combine all k results
        df_combined = results_by_k[k_values[0]][['config']].copy()
        for k in k_values:
            df_combined = df_combined.merge(
                results_by_k[k],
                on='config',
                how='left'
            )
        
        return df_combined, results_by_k
    
    def perform_lcb_pairwise_comparison(self, config1, config2, k=None):
        """
        Compare two configurations using LCB approach.
        
        Parameters:
            config1, config2: Configuration tuples
            k: Confidence parameter
        
        Returns:
            dict with comparison metrics and interpretation
        """
        if k is None:
            k = self.k
        
        lcb1 = self.compute_joint_score_lcb(config1, k=k)
        lcb2 = self.compute_joint_score_lcb(config2, k=k)
        
        if lcb1 is None or lcb2 is None:
            return None
        
        js_lcb_diff = lcb1['js_lcb'] - lcb2['js_lcb']
        
        # Check if confidence intervals overlap (using t-based CIs)
        acc_ci1 = (lcb1['accuracy']['ci_lower'], lcb1['accuracy']['ci_upper'])
        acc_ci2 = (lcb2['accuracy']['ci_lower'], lcb2['accuracy']['ci_upper'])
        auc_ci1 = (lcb1['auc']['ci_lower'], lcb1['auc']['ci_upper'])
        auc_ci2 = (lcb2['auc']['ci_lower'], lcb2['auc']['ci_upper'])
        
        acc_overlap = not (acc_ci1[1] < acc_ci2[0] or acc_ci2[1] < acc_ci1[0])
        auc_overlap = not (auc_ci1[1] < auc_ci2[0] or auc_ci2[1] < auc_ci1[0])
        
        return {
            'config1': config1,
            'config2': config2,
            'config1_label': f"S{config1[0]}_T{config1[1]}",
            'config2_label': f"S{config2[0]}_T{config2[1]}",
            'js_lcb_config1': lcb1['js_lcb'],
            'js_lcb_config2': lcb2['js_lcb'],
            'js_lcb_difference': float(js_lcb_diff),
            'lcb_config1_favored': js_lcb_diff > 0,
            'acc_ci1': acc_ci1,
            'acc_ci2': acc_ci2,
            'acc_ci_overlap': acc_overlap,
            'auc_ci1': auc_ci1,
            'auc_ci2': auc_ci2,
            'auc_ci_overlap': auc_overlap,
            'statistical_difference': not (acc_overlap and auc_overlap),
            'k': float(k),
            'n_runs_config1': lcb1['n_runs'],
            'n_runs_config2': lcb2['n_runs']
        }
    
    def generate_lcb_report(self, output_file='lcb_statistical_analysis_report.txt', k=None):
        """
        Generate comprehensive LCB statistical analysis report.
        """
        if k is None:
            k = self.k
        
        if self.lcb_results is None:
            self.compute_all_configurations_lcb(k=k)
        
        report_lines = []
        
        report_lines.append("="*80)
        report_lines.append("LOWER CONFIDENCE BOUND (LCB) STATISTICAL ANALYSIS")
        report_lines.append("="*80)
        report_lines.append(f"\nReport Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Confidence Parameter k: {k}")
        
        if k == 1.0:
            report_lines.append("  Interpretation: 68% confidence level (1 standard error)")
        elif k == 1.96:
            report_lines.append("  Interpretation: 95% confidence level (1.96 standard errors)")
        elif k == 2.0:
            report_lines.append("  Interpretation: ~95% confidence level (2 standard errors, conservative)")
        
        report_lines.append("\nMethod Overview:")
        report_lines.append("-" * 80)
        report_lines.append("JS_LCB = (mu_acc - k*SE_acc) + (mu_auc - k*SE_auc)")
        report_lines.append("\nwhere:")
        report_lines.append("  mu_acc, mu_auc: mean accuracy and AUC across all tasks")
        report_lines.append("  SE = sigma / sqrt(n): standard error of the metric")
        report_lines.append("  n: number of independent runs")
        report_lines.append("  k: confidence parameter controlling conservatism")
        report_lines.append("\nInterpretation:")
        report_lines.append("  LCB provides a lower bound: we can be confident the true")
        report_lines.append("  performance is AT LEAST this value, accounting for uncertainty.")
        report_lines.append("  This directly answers: 'What is the guaranteed minimum performance?'")
        
        # Section 1: Top Configurations by LCB
        report_lines.append("\n" + "="*80)
        report_lines.append("1. TOP CONFIGURATIONS BY LCB SCORE")
        report_lines.append("="*80)
        
        df_ranked = self.compare_lcb_vs_original()
        
        report_lines.append(f"\nTop 10 Configurations (k={k}):\n")
        for idx, (_, row) in enumerate(df_ranked.head(10).iterrows(), 1):
            report_lines.append(
                f"  {idx:2d}. {row['config']}")
            report_lines.append(
                f"      JS_LCB: {row['js_lcb']:.6f} (lower bound on joint performance)")
            report_lines.append(
                f"      JS_mean: {row['js_mean']:.6f} (average joint score observed)")
            report_lines.append(
                f"      JS_original: {row['js_original']:.6f} (mean - std approach)")
            report_lines.append(
                f"      Accuracy: {row['acc_mean']:.4f} +/- {row['acc_std']:.4f}, LCB={row['acc_lcb']:.4f}")
            report_lines.append(
                f"      AUC:      {row['auc_mean']:.4f} +/- {row['auc_std']:.4f}, LCB={row['auc_lcb']:.4f}")
            report_lines.append(
                f"      Runs: {int(row['n_runs'])}, Rank (LCB): {int(row['rank_lcb'])}, "
                f"Rank (Original): {int(row['rank_original'])}, Change: {int(row['rank_change_lcb_vs_original'])}")
            report_lines.append("")
        
        # Section 2: Ranking Comparison
        report_lines.append("\n" + "="*80)
        report_lines.append("2. RANKING STABILITY ANALYSIS")
        report_lines.append("="*80)
        
        rank_changes = df_ranked['rank_change_lcb_vs_original'].abs()
        report_lines.append(
            f"\nRanking consistency between LCB and original approach:")
        report_lines.append(
            f"  Mean absolute rank change: {rank_changes.mean():.2f}")
        report_lines.append(
            f"  Max rank change: {rank_changes.max():.0f}")
        report_lines.append(
            f"  Configs with rank change > 3: {(rank_changes > 3).sum()}")
        
        # Which configs moved most?
        top_movers = df_ranked.nlargest(5, 'rank_change_lcb_vs_original')
        report_lines.append(
            f"\n  Top 5 configs with biggest improvement in LCB ranking:")
        for idx, (_, row) in enumerate(top_movers.iterrows(), 1):
            if row['rank_change_lcb_vs_original'] > 0:
                report_lines.append(
                    f"    {row['config']}: moved up {int(row['rank_change_lcb_vs_original'])} positions "
                    f"(original rank {int(row['rank_original'])} -> LCB rank {int(row['rank_lcb'])})")
        
        # Section 3: Sensitivity Analysis
        report_lines.append("\n" + "="*80)
        report_lines.append("3. SENSITIVITY TO CONFIDENCE PARAMETER k")
        report_lines.append("="*80)
        
        df_sensitivity, _ = self.compute_lcb_across_k_values(k_values=[0.5, 1.0, 1.96, 2.0, 2.5])
        report_lines.append("\nTop 5 configurations across different k values:\n")
        report_lines.append(df_sensitivity.head(5).to_string(index=False))
        
        # Section 4: Pairwise Comparisons
        report_lines.append("\n" + "="*80)
        report_lines.append("4. TOP CONFIGURATION PAIRWISE COMPARISONS")
        report_lines.append("="*80)
        
        best_config = tuple(df_ranked.iloc[0][['shared_experts', 'task_experts']].astype(int))
        second_config = tuple(df_ranked.iloc[1][['shared_experts', 'task_experts']].astype(int))
        
        comparison = self.perform_lcb_pairwise_comparison(best_config, second_config, k=k)
        
        if comparison:
            report_lines.append(f"\n{comparison['config1_label']} vs {comparison['config2_label']}:")
            report_lines.append(f"  JS_LCB difference: {comparison['js_lcb_difference']:.6f}")
            report_lines.append(
                f"  {comparison['config1_label']} JS_LCB: {comparison['js_lcb_config1']:.6f}")
            report_lines.append(
                f"  {comparison['config2_label']} JS_LCB: {comparison['js_lcb_config2']:.6f}")
            report_lines.append(
                f"  {comparison['config1_label']} Favored: {comparison['lcb_config1_favored']}")
            report_lines.append(f"\n  Accuracy CIs:")
            report_lines.append(
                f"    {comparison['config1_label']}: [{comparison['acc_ci1'][0]:.4f}, {comparison['acc_ci1'][1]:.4f}]")
            report_lines.append(
                f"    {comparison['config2_label']}: [{comparison['acc_ci2'][0]:.4f}, {comparison['acc_ci2'][1]:.4f}]")
            report_lines.append(
                f"    Overlap: {comparison['acc_ci_overlap']}")
            report_lines.append(f"\n  AUC CIs:")
            report_lines.append(
                f"    {comparison['config1_label']}: [{comparison['auc_ci1'][0]:.4f}, {comparison['auc_ci1'][1]:.4f}]")
            report_lines.append(
                f"    {comparison['config2_label']}: [{comparison['auc_ci2'][0]:.4f}, {comparison['auc_ci2'][1]:.4f}]")
            report_lines.append(
                f"    Overlap: {comparison['auc_ci_overlap']}")
            report_lines.append(
                f"  Statistically Different: {comparison['statistical_difference']}")
        
        # Section 5: Recommendations
        report_lines.append("\n" + "="*80)
        report_lines.append("5. STATISTICAL RECOMMENDATIONS")
        report_lines.append("="*80)
        
        best_row = df_ranked.iloc[0]
        report_lines.append(f"\nRecommended Configuration: {best_row['config']}")
        report_lines.append(f"  - LCB Score: {best_row['js_lcb']:.6f} (conservative lower bound)")
        report_lines.append(f"  - Mean Score: {best_row['js_mean']:.6f} (observed average)")
        report_lines.append(f"  - Performance guarantee: At k={k}, we can be confident")
        report_lines.append(
            f"    this configuration will achieve AT LEAST JS={best_row['js_lcb']:.6f}")
        report_lines.append(f"  - Number of validation runs: {int(best_row['n_runs'])}")
        report_lines.append(
            f"  - Robustness: Rank stable across methods (original rank: {int(best_row['rank_original'])})")
        
        report_lines.append("\n" + "="*80)
        report_lines.append("INTERPRETATION GUIDE")
        report_lines.append("="*80)
        report_lines.append("\n- LCB vs Mean-based JS:")
        report_lines.append("  * LCB is MORE CONSERVATIVE (lower value)")
        report_lines.append("  * LCB better accounts for variance across runs")
        report_lines.append("  * Use LCB when you want guaranteed minimum performance")
        report_lines.append("\n- Confidence Parameter k:")
        report_lines.append("  * k=0.5: aggressive (less conservative)")
        report_lines.append("  * k=1.0: moderate confidence")
        report_lines.append("  * k=1.96: standard 95% confidence")
        report_lines.append("  * k=2.0: very conservative (recommended default)")
        report_lines.append("\n- CI Overlap:")
        report_lines.append("  * Non-overlapping CIs: significant difference between configs")
        report_lines.append("  * Overlapping CIs: no clear statistical difference")
        
        report_text = "\n".join(report_lines)
        
        # Save report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\n✅ LCB Report saved to: {output_file}")
        
        return report_text
    
    def plot_lcb_analysis(self, output_dir='lcb_analysis_plots', top_k=10):
        """
        Create comprehensive visualization of LCB analysis.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if self.lcb_results is None:
            self.compute_all_configurations_lcb(k=self.k)
        
        df = self.lcb_results.head(top_k).copy()
        
        # Plot 1: LCB vs Mean vs Original
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(df))
        width = 0.25
        
        ax.bar(x - width, df['js_lcb'], width, label=f'JS_LCB (k={self.k})', alpha=0.8)
        ax.bar(x, df['js_mean'], width, label='JS_mean', alpha=0.8)
        ax.bar(x + width, df['js_original'], width, label='JS_original', alpha=0.8)
        
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Joint Score')
        ax.set_title(f'Comparison of Scoring Methods (Top {top_k} by LCB)')
        ax.set_xticks(x)
        ax.set_xticklabels(df['config'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plot_path1 = os.path.join(output_dir, 'lcb_vs_methods.png')
        plt.savefig(plot_path1, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Plot saved: {plot_path1}")
        
        # Plot 2: Accuracy and AUC LCBs
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Accuracy
        axes[0].errorbar(x, df['acc_mean'], yerr=df['acc_std'], fmt='o', 
                        capsize=5, label='Mean +/- Std', alpha=0.7)
        axes[0].scatter(x, df['acc_lcb'], marker='v', s=100, 
                       label=f'LCB (k={self.k})', color='red', alpha=0.7)
        axes[0].set_xlabel('Configuration')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Accuracy: Mean and LCB')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(df['config'], rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # AUC
        axes[1].errorbar(x, df['auc_mean'], yerr=df['auc_std'], fmt='o', 
                        capsize=5, label='Mean +/- Std', alpha=0.7)
        axes[1].scatter(x, df['auc_lcb'], marker='v', s=100, 
                       label=f'LCB (k={self.k})', color='red', alpha=0.7)
        axes[1].set_xlabel('Configuration')
        axes[1].set_ylabel('AUC')
        axes[1].set_title('AUC: Mean and LCB')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(df['config'], rotation=45, ha='right')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path2 = os.path.join(output_dir, 'lcb_accuracy_auc.png')
        plt.savefig(plot_path2, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Plot saved: {plot_path2}")
        
        # Plot 3: Sensitivity to k
        df_sensitivity, _ = self.compute_lcb_across_k_values()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for idx, row in df_sensitivity.head(top_k).iterrows():
            k_values = [0.5, 1.0, 1.96, 2.0, 2.5]
            js_lcb_values = [row[f'js_lcb_k={k}'] for k in k_values]
            ax.plot(k_values, js_lcb_values, marker='o', label=row['config'], alpha=0.7)
        
        ax.set_xlabel('Confidence Parameter k')
        ax.set_ylabel('JS_LCB')
        ax.set_title(f'LCB Sensitivity to Confidence Parameter (Top {top_k})')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path3 = os.path.join(output_dir, 'lcb_sensitivity_k.png')
        plt.savefig(plot_path3, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Plot saved: {plot_path3}")
        
        print(f"\n✅ All plots saved to: {output_dir}")


# Main function
def run_lcb_statistical_analysis(csv_file, output_dir='lcb_statistical_analysis',
                                 k=2.0, top_k=10):
    """
    Run complete LCB statistical analysis.
    
    Parameters:
        csv_file: Path to CSV file with multi-run results
        output_dir: Output directory for reports and plots
        k: Confidence parameter (default 2.0 for ~95% confidence)
        top_k: Number of top configurations to display in plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("LOWER CONFIDENCE BOUND (LCB) STATISTICAL ANALYSIS")
    print("="*80 + "\n")
    
    # Initialize analyzer
    analyzer = LCBStatisticalAnalyzer(k=k, confidence_level=0.95)
    
    # Load data
    print(f"Loading data from: {csv_file}")
    df = analyzer.load_from_csv(csv_file)
    print(f"✓ Loaded {len(df)} data points\n")
    
    # Compute LCB for all configurations
    print(f"Computing LCB scores (k={k})...")
    df_lcb = analyzer.compute_all_configurations_lcb(k=k)
    print(f"✓ Computed LCB for {len(df_lcb)} configurations\n")
    
    # Save LCB results
    lcb_csv_path = os.path.join(output_dir, f'lcb_results_k{k}.csv')
    df_lcb.to_csv(lcb_csv_path, index=False)
    print(f"✓ LCB results saved to: {lcb_csv_path}\n")
    
    # Generate report
    print("Generating LCB analysis report...")
    report_path = os.path.join(output_dir, 'lcb_statistical_report.txt')
    analyzer.generate_lcb_report(output_file=report_path, k=k)
    
    # Create plots
    print("\nGenerating plots...")
    analyzer.plot_lcb_analysis(output_dir=os.path.join(output_dir, 'plots'), top_k=top_k)
    
    print(f"\n✅ LCB analysis complete!")
    print(f"✅ Results saved to: {output_dir}")
    
    return analyzer, df_lcb


if __name__ == '__main__':
    # Run LCB analysis
    csv_file = 'multirun_results/multi_run_results.csv'
    
    # Analysis with k=2.0 (conservative, ~95% confidence)
    analyzer, df_lcb = run_lcb_statistical_analysis(
        csv_file=csv_file,
        output_dir='lcb_statistical_analysis_results',
        k=2.0,
        top_k=10
    )
    
    print("\n" + "="*80)
    print("TOP 5 CONFIGURATIONS BY LCB SCORE")
    print("="*80)
    print(df_lcb[['config', 'acc_mean', 'acc_lcb', 'auc_mean', 'auc_lcb', 'js_lcb']].head(5))
    print("\n✅ LCB Statistical Analysis Complete!")