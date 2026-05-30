# -*- coding: utf-8 -*-
"""
submit_jobs.py
Creates exactly 2 PBS jobs (cluster limit: max_run=2 per user).
Splits all 54 configs between the 2 jobs.
Each job runs its configs sequentially, using 8 CPU workers per config.

Strategy:
  Job 0 (batch_0): configs 0-26  (27 configs)
  Job 1 (batch_1): configs 27-53 (27 configs)

Each config takes ~15h on cluster CPU.
27 configs x 15h = ~405h sequential per job.
With 47h walltime limit + auto-resubmit: ~9 resubmissions per job.
Total wall clock: ~405h = ~17 days.

To speed up further: ask admin to increase max_run limit.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'project'))

import argparse
import subprocess
import json
from pathlib import Path

from experiments.config import get_all_experiment_configs

OUTPUT_DIR = 'results'
PBS_LOGS = 'pbs_logs'
PBS_SCRIPTS = 'pbs_scripts'


def is_config_complete(config_name):
    return (Path(OUTPUT_DIR) / config_name / 'summary.json').exists()


def get_pending_configs(configs, resume):
    if resume:
        pending = [c for c in configs if not is_config_complete(c.name)]
        skipped = len(configs) - len(pending)
        if skipped:
            print(f'  Skipping {skipped} completed configs')
        return pending
    return configs


def generate_batch_script(batch_name, config_names):
    config_list = ' '.join(config_names)
    with open('job.cmd') as f:
        script = f.read()
    script = script.replace('BATCH_NAME', batch_name)
    script = script.replace('BATCH_CONFIGS', config_list)
    return script


def submit(script_content, batch_name, dry_run):
    script_path = Path(PBS_SCRIPTS) / f'{batch_name}.cmd'
    with open(script_path, 'w') as f:
        f.write(script_content)

    if dry_run:
        print(f'  [DRY RUN] {batch_name} ({script_path})')
        return 'DRY_RUN'

    result = subprocess.run(['/opt/pbs/bin/qsub', str(script_path)],
                            capture_output=True, text=True)
    if result.returncode == 0:
        job_id = result.stdout.strip()
        print(f'  Submitted {batch_name} -> {job_id}')
        return job_id
    else:
        print(f'  ERROR {batch_name}: {result.stderr.strip()}')
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true',
                        help='Skip configs that already have summary.json')
    parser.add_argument('--config', default=None,
                        help='Run only this specific config (puts it in batch_0)')
    parser.add_argument('--nbatches', type=int, default=2,
                        help='Number of parallel jobs (default: 2, cluster limit)')
    args = parser.parse_args()

    Path(PBS_LOGS).mkdir(exist_ok=True)
    Path(PBS_SCRIPTS).mkdir(exist_ok=True)
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    all_configs = get_all_experiment_configs()

    if args.config:
        configs = [c for c in all_configs if c.name == args.config]
        if not configs:
            print(f'ERROR: "{args.config}" not found')
            sys.exit(1)
    else:
        configs = get_pending_configs(all_configs, args.resume)

    if not configs:
        print('All configs complete!')
        return

    # Split configs across batches
    n = args.nbatches
    batches = [configs[i::n] for i in range(n)]

    print('=' * 65)
    print('  MARL Stable Matching — PBS Submission')
    print(f'  Total configs  : {len(configs)}')
    print(f'  Parallel jobs  : {n} (cluster max_run limit)')
    print(f'  Configs/job    : ~{len(configs)//n}')
    print(f'  Resources/job  : select=1:ncpus=32:mem=128gb')
    print(f'  Strategy       : 16 CPU workers per config')
    print('=' * 65)

    log = []
    for batch_idx, batch in enumerate(batches):
        if not batch:
            continue
        batch_name = f'batch_{batch_idx}'
        config_names = [c.name for c in batch]

        print(f'\n  {batch_name}: {len(batch)} configs')
        for name in config_names[:3]:
            print(f'    {name}')
        if len(config_names) > 3:
            print(f'    ... and {len(config_names)-3} more')

        script = generate_batch_script(batch_name, config_names)
        job_id = submit(script, batch_name, args.dry_run)
        log.append({'batch': batch_name, 'job_id': job_id,
                    'configs': config_names, 'n_configs': len(batch)})

    with open(Path(PBS_LOGS) / 'submission_log.json', 'w') as f:
        json.dump(log, f, indent=2)

    submitted = len([x for x in log if x['job_id'] not in (None, 'DRY_RUN')])
    print(f'\n{"="*65}')
    print(f'  Jobs submitted : {submitted}/{n}')
    print(f'  Each job runs  : ~{len(configs)//n} configs sequentially')
    print(f'  Monitor        : qstat -u $USER')
    print(f'  Progress       : ./run.sh progress')
    print(f'  WandB          : https://wandb.ai/ns26z139-iit-madras/marl-stable-matching')
    print(f'{"="*65}')
    print()
    print('NOTE: To run faster, ask cluster admin to increase max_run limit.')
    print('      Current limit: max_run = [u:PBS_GENERIC=2]')
    print('      Request: max_run = [u:PBS_GENERIC=10] or higher')


if __name__ == '__main__':
    main()
