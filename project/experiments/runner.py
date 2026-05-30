"""
Experiment Runner with WandB integration.
Each (config, instance, run) is one WandB run.
Grouped by config name for easy comparison in the WandB dashboard.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import json
from datetime import datetime
import wandb

from env.matching_env import MatchingEnvironment
from agents.sarsa_agent import SARSAAgent
from matching.preferences import PreferenceGenerator
from training.trainer import Trainer
from metrics.stability import StabilityMetrics
from metrics.fairness import FairnessMetrics
from metrics.median import MedianMatchingAnalyzer
from experiments.config import ExperimentConfig


WANDB_PROJECT = 'marl-stable-matching-v2'


class ExperimentRunner:

    def __init__(self, device: str = 'auto', wandb_entity: Optional[str] = None):
        if device == 'auto':
            if torch.backends.mps.is_available():
                self.device = 'mps'
            elif torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        self.wandb_entity = wandb_entity
        print(f"ExperimentRunner initialized | device={self.device}")

    def run_single_experiment(self, config: ExperimentConfig,
                               instance_id: int, run_id: int,
                               save_dir: Path) -> Dict:
        """Run one (instance, run) pair with full WandB logging."""

        run_name = f"{config.name}_i{instance_id}_r{run_id}"
        print(f"\n{'='*60}")
        print(f"  {run_name}")
        print(f"  Episodes: {config.n_episodes} | Steps: {config.steps_per_episode}")
        print(f"  lr={config.lr} | γ={config.gamma} | ε_decay={config.epsilon_decay:.6f}")
        print(f"{'='*60}")

        # --- WandB run ---
        try:
            wandb_run = wandb.init(
                project=WANDB_PROJECT,
                entity=self.wandb_entity,
                name=run_name,
                group=config.name,
                tags=[
                    config.problem_type,
                    'sym' if config.symmetric else 'asym',
                    f'{config.grid_size[0]}x{config.grid_size[1]}',
                    f'{config.n_agents}agents',
                ],
                config=config.to_dict(),
                reinit=True,
                settings=wandb.Settings(init_timeout=30),
            )
        except Exception as e:
            print(f'  WandB init failed (offline?): {e}')
            wandb_run = None

        # --- Setup ---
        seed = instance_id * 1000 + run_id

        n_per_side = config.n_agents // 2

        if config.problem_type == 'SM':
            utilities = PreferenceGenerator.generate_sm(n_per_side, config.symmetric, seed)
        elif config.problem_type == 'SMI':
            utilities = PreferenceGenerator.generate_smi(n_per_side, 0.3, config.symmetric, seed)
        else:
            utilities = PreferenceGenerator.generate_smt(n_per_side, 0.2, config.symmetric, seed)

        env = MatchingEnvironment(config.grid_size, n_per_side, utilities, seed=seed)

        agents = {}
        for i in range(n_per_side):
            for side in ['s1', 's2']:
                aid = f'{side}_{i}'
                agents[aid] = SARSAAgent(
                    env.obs_dim, env.action_dim, aid,
                    lr=config.lr,
                    gamma=config.gamma,
                    epsilon_start=config.epsilon_start,
                    epsilon_min=config.epsilon_min,
                    epsilon_decay=config.epsilon_decay,
                    replay_buffer_size=config.replay_buffer_size,
                    device=self.device,
                )

        trainer_config = {
            'n_episodes': config.n_episodes,
            'steps_per_episode': config.steps_per_episode,
            'batch_size': config.batch_size,
            'log_frequency': config.log_frequency,
            'save_frequency': config.save_frequency,
        }

        exp_save_dir = save_dir / f'instance_{instance_id}_run_{run_id}'
        trainer = Trainer(env, agents, trainer_config, wandb_run=wandb_run)

        # --- Train ---
        training_stats = trainer.train(save_dir=str(exp_save_dir))

        # --- Evaluate (greedy, epsilon=0) ---
        eval_stats = trainer.evaluate(n_episodes=10)
        final_matching = eval_stats['matches'][-1]

        # --- Compute paper metrics ---
        n_per_side = config.n_agents // 2
        stability = StabilityMetrics.compute_all_stability_metrics(
            final_matching, utilities, n_per_side
        )
        fairness = FairnessMetrics.compute_all_fairness_metrics(
            final_matching, utilities, n_per_side
        )
        median = MedianMatchingAnalyzer.analyze_median_properties(
            final_matching, utilities, n_per_side
        )

        # --- Log final metrics to WandB ---
        if wandb_run is not None:
            try:
                wandb_run.log({
                    'final/is_stable': int(stability['is_stable']),
                    'final/doi': stability['doi'],
                    'final/roi': stability['roi'],
                    'final/md': stability['md'],
                    'final/num_blocking_pairs': stability['num_blocking_pairs'],
                    'final/regret_cost': fairness['regret_cost'],
                    'final/egalitarian_cost': fairness['egalitarian_cost'],
                    'final/set_equality_cost': fairness['set_equality_cost'],
                    'final/num_stable_matchings': median['num_stable_matchings'],
                    'final/is_msm': int(median['is_msm']),
                    'final/mm_proportion': median['mm_proportion'],
                    'final/eval_avg_reward': float(np.mean(list(eval_stats['avg_reward'].values()))),
                })
                wandb_run.finish()
            except Exception as e:
                print(f'  WandB log failed: {e}')

        # --- Compile results ---
        results = {
            'config_name': config.name,
            'instance_id': instance_id,
            'run_id': run_id,
            'n_agents': config.n_agents,  # total agents (n_per_side = n_agents//2)
            'grid_size': list(config.grid_size),
            'problem_type': config.problem_type,
            'symmetric': config.symmetric,
            'n_episodes': config.n_episodes,
            'seed': seed,
            'is_stable': bool(stability['is_stable']),
            'doi': int(stability['doi']),
            'roi': float(stability['roi']),
            'md': float(stability['md']),
            'num_blocking_pairs': int(stability['num_blocking_pairs']),
            'regret_cost': int(fairness['regret_cost']),
            'egalitarian_cost': int(fairness['egalitarian_cost']),
            'set_equality_cost': int(fairness['set_equality_cost']),
            'num_stable_matchings': int(median['num_stable_matchings']),
            'is_msm': bool(median['is_msm']),
            'mm_proportion': float(median['mm_proportion']),
            'eval_avg_reward': float(np.mean(list(eval_stats['avg_reward'].values()))),
            'final_avg_reward': float(np.mean(list(training_stats['final_avg_reward'].values()))),
            'timestamp': datetime.now().isoformat(),
        }

        with open(exp_save_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)

        status = '✓ STABLE' if results['is_stable'] else f'✗ DoI={results["doi"]}'
        print(f"  {status} | Regret={results['regret_cost']} | "
              f"Egal={results['egalitarian_cost']} | MM={results['mm_proportion']:.2f}")

        return results

    def run_experiment_config(self, config: ExperimentConfig,
                               base_save_dir: Path) -> pd.DataFrame:
        """Run all instances × runs for one config. Returns results DataFrame."""
        save_dir = base_save_dir / config.name
        save_dir.mkdir(parents=True, exist_ok=True)
        config.save(str(save_dir / 'config.json'))

        all_results = []
        for instance_id in range(config.n_instances):
            for run_id in range(config.n_runs):
                try:
                    result = self.run_single_experiment(config, instance_id, run_id, save_dir)
                    all_results.append(result)
                except Exception as e:
                    import traceback
                    print(f"  ERROR instance={instance_id} run={run_id}: {e}")
                    traceback.print_exc()

        df = pd.DataFrame(all_results)
        df.to_csv(save_dir / 'all_results.csv', index=False)

        summary = self._compute_summary(df, config)
        with open(save_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        # Log config-level summary as a WandB summary run
        self._log_config_summary(config, summary)

        print(f"\n  Config done: {config.name}")
        print(f"  Stability: {summary['stability_rate']*100:.1f}% | "
              f"Avg DoI: {summary['avg_doi']:.2f} | "
              f"Avg Regret: {summary['avg_regret_cost']:.2f}")

        return df

    def _compute_summary(self, df: pd.DataFrame, config: ExperimentConfig) -> Dict:
        def _s(col):
            return {'mean': float(df[col].mean()), 'std': float(df[col].std())}

        return {
            'config_name': config.name,
            'n_experiments': len(df),
            'stability_rate': float(df['is_stable'].mean()),
            'doi': _s('doi'),
            'roi': _s('roi'),
            'md': _s('md'),
            'regret_cost': _s('regret_cost'),
            'egalitarian_cost': _s('egalitarian_cost'),
            'set_equality_cost': _s('set_equality_cost'),
            'mm_proportion': _s('mm_proportion'),
            'eval_avg_reward': _s('eval_avg_reward'),
        }

    def _log_config_summary(self, config: ExperimentConfig, summary: Dict):
        """One WandB run per config that holds the aggregated summary metrics."""
        run = wandb.init(
            project=WANDB_PROJECT,
            entity=self.wandb_entity,
            name=f"{config.name}_SUMMARY",
            group=config.name,
            tags=['summary', config.problem_type,
                  'sym' if config.symmetric else 'asym'],
            config=config.to_dict(),
            reinit=True,
        )
        run.log({
            'summary/stability_rate': summary['stability_rate'],
            'summary/avg_doi': summary['doi']['mean'],
            'summary/std_doi': summary['doi']['std'],
            'summary/avg_roi': summary['roi']['mean'],
            'summary/avg_md': summary['md']['mean'],
            'summary/avg_regret_cost': summary['regret_cost']['mean'],
            'summary/avg_egalitarian_cost': summary['egalitarian_cost']['mean'],
            'summary/avg_set_equality_cost': summary['set_equality_cost']['mean'],
            'summary/avg_mm_proportion': summary['mm_proportion']['mean'],
            'summary/avg_eval_reward': summary['eval_avg_reward']['mean'],
        })
        run.finish()
