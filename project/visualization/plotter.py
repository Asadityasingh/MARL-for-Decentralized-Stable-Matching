"""
Visualization Module
Create publication-quality plots for results
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

# Set style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 150


class ResultsVisualizer:
    """Create visualizations for experiment results"""
    
    def __init__(self, save_dir: Optional[Path] = None):
        """
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_training_curve(self, rewards_df: pd.DataFrame, 
                           title: str = "Training Curve",
                           save_name: Optional[str] = None):
        """
        Plot training rewards over episodes.
        
        Args:
            rewards_df: DataFrame with episode rewards
            title: Plot title
            save_name: Filename to save (without extension)
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Average across all agents
        avg_rewards = rewards_df.mean(axis=1)
        episodes = np.arange(len(avg_rewards))
        
        # Plot with smoothing
        window = min(100, len(avg_rewards) // 10)
        if window > 1:
            smoothed = avg_rewards.rolling(window=window, center=True).mean()
            ax.plot(episodes, smoothed, linewidth=2, label='Smoothed')
            ax.plot(episodes, avg_rewards, alpha=0.3, linewidth=0.5, label='Raw')
        else:
            ax.plot(episodes, avg_rewards, linewidth=2)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Reward')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            plt.savefig(self.save_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
            plt.savefig(self.save_dir / f"{save_name}.pdf", bbox_inches='tight')
        
        plt.close()
    
    def plot_stability_vs_agents(self, results_df: pd.DataFrame,
                                save_name: Optional[str] = None):
        """
        Plot stability metrics vs number of agents.
        
        Args:
            results_df: DataFrame with experiment results
            save_name: Filename to save
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Group by n_agents
        grouped = results_df.groupby('n_agents')
        
        # Stability rate
        stability_rate = grouped['is_stable'].mean()
        stability_std = grouped['is_stable'].std()
        
        axes[0, 0].bar(stability_rate.index, stability_rate.values, 
                      yerr=stability_std.values, capsize=5, alpha=0.7)
        axes[0, 0].set_xlabel('Number of Agents')
        axes[0, 0].set_ylabel('Stability Rate')
        axes[0, 0].set_title('Stability Rate vs Number of Agents')
        axes[0, 0].set_ylim([0, 1.1])
        axes[0, 0].grid(True, alpha=0.3)
        
        # DoI
        doi_mean = grouped['doi'].mean()
        doi_std = grouped['doi'].std()
        
        axes[0, 1].errorbar(doi_mean.index, doi_mean.values, 
                           yerr=doi_std.values, marker='o', capsize=5, linewidth=2)
        axes[0, 1].set_xlabel('Number of Agents')
        axes[0, 1].set_ylabel('Degree of Instability (DoI)')
        axes[0, 1].set_title('DoI vs Number of Agents')
        axes[0, 1].grid(True, alpha=0.3)
        
        # RoI
        roi_mean = grouped['roi'].mean()
        roi_std = grouped['roi'].std()
        
        axes[1, 0].errorbar(roi_mean.index, roi_mean.values,
                           yerr=roi_std.values, marker='s', capsize=5, linewidth=2)
        axes[1, 0].set_xlabel('Number of Agents')
        axes[1, 0].set_ylabel('Ratio of Instability (RoI)')
        axes[1, 0].set_title('RoI vs Number of Agents')
        axes[1, 0].grid(True, alpha=0.3)
        
        # MD
        md_mean = grouped['md'].mean()
        md_std = grouped['md'].std()
        
        axes[1, 1].errorbar(md_mean.index, md_mean.values,
                           yerr=md_std.values, marker='^', capsize=5, linewidth=2)
        axes[1, 1].set_xlabel('Number of Agents')
        axes[1, 1].set_ylabel('Maximum Dissatisfaction (MD)')
        axes[1, 1].set_title('MD vs Number of Agents')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            plt.savefig(self.save_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
            plt.savefig(self.save_dir / f"{save_name}.pdf", bbox_inches='tight')
        
        plt.close()
    
    def plot_fairness_metrics(self, results_df: pd.DataFrame,
                             save_name: Optional[str] = None):
        """
        Plot fairness metrics.
        
        Args:
            results_df: DataFrame with experiment results
            save_name: Filename to save
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        grouped = results_df.groupby('n_agents')
        
        # Regret cost
        regret_mean = grouped['regret_cost'].mean()
        regret_std = grouped['regret_cost'].std()
        
        axes[0].errorbar(regret_mean.index, regret_mean.values,
                        yerr=regret_std.values, marker='o', capsize=5, linewidth=2)
        axes[0].set_xlabel('Number of Agents')
        axes[0].set_ylabel('Regret Cost')
        axes[0].set_title('Regret Cost vs Number of Agents')
        axes[0].grid(True, alpha=0.3)
        
        # Egalitarian cost
        egal_mean = grouped['egalitarian_cost'].mean()
        egal_std = grouped['egalitarian_cost'].std()
        
        axes[1].errorbar(egal_mean.index, egal_mean.values,
                        yerr=egal_std.values, marker='s', capsize=5, linewidth=2)
        axes[1].set_xlabel('Number of Agents')
        axes[1].set_ylabel('Egalitarian Cost')
        axes[1].set_title('Egalitarian Cost vs Number of Agents')
        axes[1].grid(True, alpha=0.3)
        
        # Set-equality cost
        seteq_mean = grouped['set_equality_cost'].mean()
        seteq_std = grouped['set_equality_cost'].std()
        
        axes[2].errorbar(seteq_mean.index, seteq_mean.values,
                        yerr=seteq_std.values, marker='^', capsize=5, linewidth=2)
        axes[2].set_xlabel('Number of Agents')
        axes[2].set_ylabel('Set-Equality Cost')
        axes[2].set_title('Set-Equality Cost vs Number of Agents')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            plt.savefig(self.save_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
            plt.savefig(self.save_dir / f"{save_name}.pdf", bbox_inches='tight')
        
        plt.close()
    
    def plot_comparison_by_problem_type(self, results_df: pd.DataFrame,
                                       save_name: Optional[str] = None):
        """
        Compare results across problem types (SM, SMI, SMT).
        
        Args:
            results_df: DataFrame with experiment results
            save_name: Filename to save
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Stability rate by problem type
        stability_by_type = results_df.groupby(['problem_type', 'n_agents'])['is_stable'].mean().unstack()
        
        stability_by_type.T.plot(kind='bar', ax=axes[0, 0], alpha=0.7)
        axes[0, 0].set_xlabel('Number of Agents')
        axes[0, 0].set_ylabel('Stability Rate')
        axes[0, 0].set_title('Stability Rate by Problem Type')
        axes[0, 0].legend(title='Problem Type')
        axes[0, 0].grid(True, alpha=0.3)
        
        # DoI by problem type
        doi_by_type = results_df.groupby(['problem_type', 'n_agents'])['doi'].mean().unstack()
        
        doi_by_type.T.plot(kind='bar', ax=axes[0, 1], alpha=0.7)
        axes[0, 1].set_xlabel('Number of Agents')
        axes[0, 1].set_ylabel('DoI')
        axes[0, 1].set_title('DoI by Problem Type')
        axes[0, 1].legend(title='Problem Type')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Regret cost by problem type
        regret_by_type = results_df.groupby(['problem_type', 'n_agents'])['regret_cost'].mean().unstack()
        
        regret_by_type.T.plot(kind='bar', ax=axes[1, 0], alpha=0.7)
        axes[1, 0].set_xlabel('Number of Agents')
        axes[1, 0].set_ylabel('Regret Cost')
        axes[1, 0].set_title('Regret Cost by Problem Type')
        axes[1, 0].legend(title='Problem Type')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Egalitarian cost by problem type
        egal_by_type = results_df.groupby(['problem_type', 'n_agents'])['egalitarian_cost'].mean().unstack()
        
        egal_by_type.T.plot(kind='bar', ax=axes[1, 1], alpha=0.7)
        axes[1, 1].set_xlabel('Number of Agents')
        axes[1, 1].set_ylabel('Egalitarian Cost')
        axes[1, 1].set_title('Egalitarian Cost by Problem Type')
        axes[1, 1].legend(title='Problem Type')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            plt.savefig(self.save_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
            plt.savefig(self.save_dir / f"{save_name}.pdf", bbox_inches='tight')
        
        plt.close()
    
    def plot_symmetric_vs_asymmetric(self, results_df: pd.DataFrame,
                                    save_name: Optional[str] = None):
        """
        Compare symmetric vs asymmetric preferences.
        
        Args:
            results_df: DataFrame with experiment results
            save_name: Filename to save
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Stability rate
        stability = results_df.groupby(['symmetric', 'n_agents'])['is_stable'].mean().unstack()
        
        stability.T.plot(kind='bar', ax=axes[0], alpha=0.7)
        axes[0].set_xlabel('Number of Agents')
        axes[0].set_ylabel('Stability Rate')
        axes[0].set_title('Stability: Symmetric vs Asymmetric')
        axes[0].legend(['Asymmetric', 'Symmetric'])
        axes[0].grid(True, alpha=0.3)
        
        # DoI
        doi = results_df.groupby(['symmetric', 'n_agents'])['doi'].mean().unstack()
        
        doi.T.plot(kind='bar', ax=axes[1], alpha=0.7)
        axes[1].set_xlabel('Number of Agents')
        axes[1].set_ylabel('DoI')
        axes[1].set_title('DoI: Symmetric vs Asymmetric')
        axes[1].legend(['Asymmetric', 'Symmetric'])
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            plt.savefig(self.save_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
            plt.savefig(self.save_dir / f"{save_name}.pdf", bbox_inches='tight')
        
        plt.close()
    
    def create_summary_report(self, results_df: pd.DataFrame, 
                             report_name: str = "summary_report"):
        """
        Create a comprehensive summary report with all plots.
        
        Args:
            results_df: DataFrame with all experiment results
            report_name: Base name for report files
        """
        print(f"Generating summary report...")
        
        self.plot_stability_vs_agents(results_df, f"{report_name}_stability")
        print(f"  ✓ Stability plots saved")
        
        self.plot_fairness_metrics(results_df, f"{report_name}_fairness")
        print(f"  ✓ Fairness plots saved")
        
        self.plot_comparison_by_problem_type(results_df, f"{report_name}_by_type")
        print(f"  ✓ Problem type comparison saved")
        
        self.plot_symmetric_vs_asymmetric(results_df, f"{report_name}_sym_vs_asym")
        print(f"  ✓ Symmetric vs asymmetric comparison saved")
        
        print(f"\n✓ Summary report complete! Saved to: {self.save_dir}")
