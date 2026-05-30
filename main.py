# -*- coding: utf-8 -*-
"""
Main entry point for MARL Stable Matching experiments.

Modes:
  validate  - 1 config (3x3, 8 agents, SM-sym), 2 instances x 2 runs
              Paper expects 100% stability. Use this to confirm everything works.
  full      - All 54 paper configs, 10 instances x 5 runs each (~2700 total runs)

Usage:
  python main.py --mode validate
  python main.py --mode full
  python main.py --mode full --resume
  python main.py --mode full --config 3x3_8agents_SM_sym
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'project'))

import argparse
import json
from pathlib import Path
import wandb

from experiments.config import get_all_experiment_configs, get_validation_config
from experiments.runner import ExperimentRunner


def get_completed_configs(output_dir: Path) -> set:
    """Return set of config names that already have a summary.json."""
    completed = set()
    if output_dir.exists():
        for summary_file in output_dir.glob('*/summary.json'):
            with open(summary_file) as f:
                data = json.load(f)
            name = data.get('config_name', summary_file.parent.name)
            completed.add(name)
    return completed


def main():
    parser = argparse.ArgumentParser(
        description='MARL for Decentralized Stable Matching — Paper Reproduction'
    )
    parser.add_argument('--mode', choices=['validate', 'full'], default='validate',
                        help='validate: single sanity-check config | full: all 54 paper configs')
    parser.add_argument('--output', default='results',
                        help='Output directory for results (default: results/)')
    parser.add_argument('--device', default='auto',
                        choices=['auto', 'mps', 'cuda', 'cpu'])
    parser.add_argument('--wandb-entity', default=None,
                        help='WandB entity/username (optional)')
    parser.add_argument('--resume', action='store_true',
                        help='Skip configs that already have a summary.json')
    parser.add_argument('--config', default=None,
                        help='Run a single config by name (full mode only)')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('  MARL for Decentralized Stable Matching')
    print('  Taywade, Goldsmith & Harrison (2021)')
    print('=' * 60)
    print(f'  Mode   : {args.mode}')
    print(f'  Device : {args.device}')
    print(f'  Output : {output_dir}')
    print(f'  WandB  : project=marl-stable-matching')
    print('=' * 60)

    runner = ExperimentRunner(device=args.device, wandb_entity=args.wandb_entity)

    if args.mode == 'validate':
        config = get_validation_config()
        print(f'\nValidation config: {config.name}')
        print(f'  Grid={config.grid_size} | Agents={config.n_agents} | '
              f'Episodes={config.n_episodes} | ε_decay={config.epsilon_decay:.6f}')
        print(f'  Expected result: 100% stability (paper Table 1)\n')
        runner.run_experiment_config(config, output_dir)
        print('\nValidation complete. Check WandB for training curves.')
        print('If stability_rate=1.0 → ready for full run.')

    else:  # full
        all_configs = get_all_experiment_configs()

        if args.config:
            all_configs = [c for c in all_configs if c.name == args.config]
            if not all_configs:
                print(f'ERROR: config "{args.config}" not found.')
                print('Available configs:')
                for c in get_all_experiment_configs():
                    print(f'  {c.name}')
                sys.exit(1)

        if args.resume:
            completed = get_completed_configs(output_dir)
            skipped = [c for c in all_configs if c.name in completed]
            all_configs = [c for c in all_configs if c.name not in completed]
            if skipped:
                print(f'Resuming: skipping {len(skipped)} completed configs.')

        total = len(all_configs)
        total_runs = sum(c.n_instances * c.n_runs for c in all_configs)
        print(f'\nConfigs to run : {total}')
        print(f'Total runs     : {total_runs}')
        print(f'Estimated time : see README for time estimates\n')

        if not args.config:
            confirm = input('Start full experiment run? (yes/no): ')
            if confirm.strip().lower() != 'yes':
                print('Aborted.')
                sys.exit(0)

        for i, config in enumerate(all_configs):
            print(f'\n[{i+1}/{total}] {config.name}')
            try:
                runner.run_experiment_config(config, output_dir)
            except KeyboardInterrupt:
                print('\nInterrupted. Results saved so far. Use --resume to continue.')
                sys.exit(0)
            except Exception as e:
                import traceback
                print(f'ERROR in {config.name}: {e}')
                traceback.print_exc()
                continue

        print('\n' + '=' * 60)
        print('  All experiments complete!')
        print(f'  Results: {output_dir}')
        print('=' * 60)


if __name__ == '__main__':
    main()
