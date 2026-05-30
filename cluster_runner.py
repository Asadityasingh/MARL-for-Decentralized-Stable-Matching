# -*- coding: utf-8 -*-
"""
cluster_runner.py
Runs 50 runs for one config using multiple CPU workers in parallel.
CPU is 6x faster than GPU for tiny SARSA networks (3435 params).
With 8 CPU workers: 50 runs split across 8 parallel processes.
Fully resumable — skips already-completed runs.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'project'))

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import torch
import numpy as np

from experiments.config import get_all_experiment_configs
from experiments.runner import ExperimentRunner


def is_completed(save_dir, instance_id, run_id):
    f = Path(save_dir) / f'instance_{instance_id}_run_{run_id}' / 'results.json'
    return f.exists()


def worker(worker_id, config_name, run_pairs, output_dir, wandb_entity):
    """CPU worker process — runs assigned runs sequentially."""
    # Force CPU — GPU kernel overhead dominates for tiny 3435-param networks
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    device = 'cpu'
    # Limit threads per worker to avoid oversubscription
    torch.set_num_threads(2)
    os.environ['OMP_NUM_THREADS'] = '2'

    print(f'[Worker {worker_id}] CPU | {len(run_pairs)} runs assigned')

    all_configs = get_all_experiment_configs()
    config = next((c for c in all_configs if c.name == config_name), None)
    if config is None:
        print(f'[Worker {worker_id}] ERROR: config {config_name} not found')
        return

    save_dir = Path(output_dir) / config.name
    save_dir.mkdir(parents=True, exist_ok=True)

    runner = ExperimentRunner(device=device, wandb_entity=wandb_entity)

    for instance_id, run_id in run_pairs:
        if is_completed(save_dir, instance_id, run_id):
            print(f'[Worker {worker_id}] Skip i={instance_id} r={run_id} (done)')
            continue
        try:
            print(f'[Worker {worker_id}] Starting i={instance_id} r={run_id}')
            runner.run_single_experiment(config, instance_id, run_id, save_dir)
        except Exception as e:
            import traceback
            print(f'[Worker {worker_id}] ERROR i={instance_id} r={run_id}: {e}')
            traceback.print_exc()


def write_summary_if_complete(config, output_dir):
    import pandas as pd
    save_dir = Path(output_dir) / config.name
    done = list(save_dir.glob('instance_*_run_*/results.json'))
    total = config.n_instances * config.n_runs

    if len(done) < total:
        print(f'  {len(done)}/{total} runs complete — summary pending')
        return

    print(f'  All {total} runs complete — writing summary')
    results = [json.load(open(f)) for f in done]
    df = pd.DataFrame(results)
    df.to_csv(save_dir / 'all_results.csv', index=False)

    summary = {
        'config_name': config.name,
        'n_experiments': len(df),
        'stability_rate': float(df['is_stable'].mean()),
        'doi':              {'mean': float(df['doi'].mean()),              'std': float(df['doi'].std())},
        'roi':              {'mean': float(df['roi'].mean()),              'std': float(df['roi'].std())},
        'md':               {'mean': float(df['md'].mean()),               'std': float(df['md'].std())},
        'regret_cost':      {'mean': float(df['regret_cost'].mean()),      'std': float(df['regret_cost'].std())},
        'egalitarian_cost': {'mean': float(df['egalitarian_cost'].mean()), 'std': float(df['egalitarian_cost'].std())},
        'set_equality_cost':{'mean': float(df['set_equality_cost'].mean()),'std': float(df['set_equality_cost'].std())},
        'mm_proportion':    {'mean': float(df['mm_proportion'].mean()),    'std': float(df['mm_proportion'].std())},
    }
    with open(save_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'  Stability: {summary["stability_rate"]*100:.1f}%')
    print(f'  Avg DoI:   {summary["doi"]["mean"]:.2f} +/- {summary["doi"]["std"]:.2f}')
    print(f'  Avg RoI:   {summary["roi"]["mean"]:.4f}')
    print(f'  Regret:    {summary["regret_cost"]["mean"]:.2f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', default='results')
    parser.add_argument('--nworkers', type=int, default=8,
                        help='Number of parallel CPU workers (default: 8)')
    parser.add_argument('--wandb-entity', default=None)
    args = parser.parse_args()

    all_configs = get_all_experiment_configs()
    config = next((c for c in all_configs if c.name == args.config), None)
    if config is None:
        print(f'ERROR: config "{args.config}" not found')
        sys.exit(1)

    all_pairs = [(i, r) for i in range(config.n_instances) for r in range(config.n_runs)]
    save_dir = Path(args.output) / config.name
    pending = [(i, r) for i, r in all_pairs if not is_completed(save_dir, i, r)]

    print('=' * 60)
    print(f'  Config   : {config.name}')
    print(f'  Total    : {len(all_pairs)} runs')
    print(f'  Pending  : {len(pending)} runs')
    print(f'  Workers  : {args.nworkers} CPU processes')
    print(f'  Device   : CPU (6x faster than GPU for small networks)')
    print('=' * 60)

    if not pending:
        print('All runs already complete.')
        write_summary_if_complete(config, args.output)
        return

    # Round-robin split across workers
    chunks = [pending[i::args.nworkers] for i in range(args.nworkers)]
    for wid, chunk in enumerate(chunks):
        if chunk:
            print(f'  Worker {wid}: {len(chunk)} runs')

    processes = []
    for wid, chunk in enumerate(chunks):
        if not chunk:
            continue
        p = mp.Process(
            target=worker,
            args=(wid, config.name, chunk, args.output, args.wandb_entity)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print('\nAll workers finished.')
    write_summary_if_complete(config, args.output)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
