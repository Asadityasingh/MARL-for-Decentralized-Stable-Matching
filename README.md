# Multi-Agent Reinforcement Learning for Decentralized Stable Matching

Full reproduction of the paper: **"Multi-agent Reinforcement Learning for Decentralized Stable Matching"**
by Taywade, Goldsmith, and Harrison (2021) — arXiv:2005.01117v3

---

## Paper Reference

```
Kshitija Taywade, Judy Goldsmith, and Brent Harrison
"Multi-agent Reinforcement Learning for Decentralized Stable Matching"
arXiv:2005.01117v3 [cs.LG] 4 Dec 2021
```

---

## Project Structure

```
Stable_matching/
├── project/
│   ├── env/                  # Grid-based matching environment
│   ├── agents/               # SARSA agent with neural network
│   ├── matching/             # Preference generation (SM, SMI, SMT)
│   ├── training/             # Training loop with WandB logging
│   ├── metrics/              # Stability (DoI, RoI, MD) and fairness metrics
│   ├── experiments/          # Config definitions and experiment runner
│   └── visualization/        # Result plotting
├── main.py                   # Local run entry point
├── cluster_runner.py         # PBS job worker (runs on GPU node)
├── submit_jobs.py            # Submits all 54 jobs to PBS cluster
├── job.cmd                   # PBS job template
├── run.sh                    # Unified control script
├── requirements.txt
└── 2005.01117v3-2.txt        # Original paper
```

---

## Algorithm (Paper Specs)

| Parameter | Value |
|---|---|
| Algorithm | SARSA (on-policy TD) |
| Network | Input -> 50 -> 25 -> Output (MLP) |
| Optimizer | Adam, lr=1e-4 |
| Discount | gamma=0.9 |
| Epsilon | 1.0 -> 0.05 (hits min at 80% of training) |
| Replay buffer | Last 10 episodes |
| Batch size | 32 |
| Reward (matched) | U_ij * N(1, 0.1) |
| Reward (unmatched) | -1 per step |

---

## Experiments (54 Configurations)

| Grid | Agents | Episodes | Steps/ep | Configs |
|---|---|---|---|---|
| 3x3 | 8 | 100k | 300 | 6 |
| 4x4 | 8 | 100k | 400 | 6 |
| 4x4 | 10 | 200k | 400 | 6 |
| 4x4 | 12 | 300k | 500 | 6 |
| 4x4 | 14 | 300k | 500 | 6 |
| 5x5 | 8 | 100k | 500 | 6 |
| 5x5 | 10 | 200k | 500 | 6 |
| 5x5 | 12 | 400k | 700 | 6 |
| 5x5 | 14 | 400k | 700 | 6 |

> **Note on steps/episode and episodes:** The paper states *"steps per episode varies between 300–700 and training can take between 100k to 400k episodes to converge"* without specifying exact values per config. Our assignments scale linearly with grid size and agent count within those ranges, which is the logical interpretation.

Each config: 3 problem types (SM, SMI, SMT) x 2 preference types (sym, asym)
Each config: 10 instances x 5 runs = **50 runs per config**
**Total: 54 configs x 50 runs = 2,700 training runs**

---

## Hardware

| Machine | Use |
|---|---|
| Mac Mini M4 (MPS) | Development, validation |
| IITM Aqua HPC Cluster | Full experiments (54 jobs in parallel) |
| Cluster GPU | 2x Tesla V100-PCIE-32GB per job |
| Cluster specs | 80 CPU cores, 188GB RAM, 264TB storage |

---

## Local Setup (Mac Mini)

```bash
cd Stable_matching
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run validation (sanity check before full run)
```bash
./run.sh validate
```

### Watch live log
```bash
./run.sh log
```

### Check status
```bash
./run.sh status
```

### Stop
```bash
./run.sh stop
```

---

## Cluster Setup (IITM Aqua HPC)

### First time setup on cluster
```bash
ssh ns26z139@aqua.iitm.ac.in
cd Stable_matching
module load anaconda3_2024.10
module load gcc13.3.0
/lfs/sware/anaconda3_2023/envs/torch_gpu/bin/pip install gymnasium wandb
/lfs/sware/anaconda3_2023/envs/torch_gpu/bin/wandb login
```

### Sync code from Mac Mini to cluster
```bash
rsync -avz --exclude='venv/' --exclude='wandb/' --exclude='__pycache__/' --exclude='*.pyc' --exclude='.DS_Store' . ns26z139@aqua.iitm.ac.in:~/Stable_matching
```

### Submit all 54 jobs
```bash
/lfs/sware/anaconda3_2023/envs/torch_gpu/bin/python submit_jobs.py
```

### Dry run (preview without submitting)
```bash
/lfs/sware/anaconda3_2023/envs/torch_gpu/bin/python submit_jobs.py --dry-run
```

### Resume after interruption (skips completed configs)
```bash
/lfs/sware/anaconda3_2023/envs/torch_gpu/bin/python submit_jobs.py --resume
```

### Submit single config
```bash
/lfs/sware/anaconda3_2023/envs/torch_gpu/bin/python submit_jobs.py --config 3x3_8agents_SM_sym
```

---

## Monitoring

### Check from Mac Mini (single command)
```bash
ssh ns26z139@aqua.iitm.ac.in "cd Stable_matching && qstat -u ns26z139 && echo '---' && ./run.sh progress"
```

### Check PBS queue on cluster
```bash
qstat -u ns26z139
```

### Count completed runs
```bash
./run.sh progress
```

### Check live job output
```bash
ssh gpu009 "tail -50 /var/spool/pbs/spool/JOBID.hn1.OU"
```

### Check GPU usage on compute node
```bash
ssh gpu009 "nvidia-smi"
ssh gpu009 "ps aux | grep python | grep -v grep"
```

### Check PBS job logs (written after job finishes)
```bash
cat pbs_logs/output_3x3_8agents_SM_sym.log
cat pbs_logs/error_3x3_8agents_SM_sym.log
```

### WandB dashboard
```
https://wandb.ai/ns26z139-iit-madras/marl-stable-matching
```

---

## How Cluster Jobs Work

- **1 job per config** — 54 jobs submitted simultaneously
- **2 GPUs per job** — GPU 0 runs instances 0-4, GPU 1 runs instances 5-9
- **25 runs per GPU** — parallel via Python multiprocessing
- **47h walltime** — cluster hard limit
- **Auto-resubmit** — job resubmits itself if not all 50 runs done
- **Resumable** — skips already-completed runs on restart
- **WandB streaming** — every episode logged in real time

### Job lifecycle for large configs (e.g. 5x5 14-agents)
```
Day 0:  Submit -> runs 50 runs across 2 GPUs
Day 2:  47h limit -> saves progress -> auto-resubmits
Day 4:  Resumes from where it stopped
...
Day 10: All 50 runs done -> job ends
```

---

## Expected Timeline

| Time | Event |
|---|---|
| +12h | 3x3 configs complete (6 configs, 300 runs) |
| +15h | 4x4 8-agent configs complete |
| +35h | 4x4 10-agent + 5x5 8-agent complete |
| +Day 4 | 4x4 12-agent configs complete |
| +Day 5.5 | 4x4 14-agent configs complete |
| +Day 7 | 5x5 12-agent configs complete |
| +Day 10 | ALL 2,700 runs complete |

---

## Metrics Computed

### Stability
- **is_stable** — no blocking pairs
- **DoI** — degree of instability (number of blocking agents)
- **RoI** — ratio of instability (blocking pairs / n^2)
- **MD** — maximum dissatisfaction

### Fairness
- **Regret cost** — max rank among matched pairs
- **Egalitarian cost** — sum of all ranks
- **Set-equality cost** — |sum_S1_ranks - sum_S2_ranks|
- **MM proportion** — % agents matched to median stable partner
- **is_MSM** — whether matching is median stable matching

---

## Results Structure

```
results/
├── 3x3_8agents_SM_sym/
│   ├── config.json
│   ├── instance_0_run_0/
│   │   ├── results.json        # metrics for this run
│   │   ├── final_models/       # saved agent weights
│   │   └── logs/               # episode rewards, losses
│   ├── ...
│   ├── all_results.csv         # all 50 runs combined
│   └── summary.json            # aggregated stats
├── 3x3_8agents_SM_asym/
│   └── ...
└── ...
```

---

## References

1. Taywade, K., Goldsmith, J., & Harrison, B. (2021). Multi-agent Reinforcement Learning for Decentralized Stable Matching. arXiv:2005.01117v3.
2. Gale, D., & Shapley, L. S. (1962). College admissions and the stability of marriage. The American Mathematical Monthly, 69(1), 9-15.
3. Zhao, D., et al. (2016). Deep reinforcement learning with experience replay based on SARSA. IEEE SSCI.

---

## Authors

Implementation: Aditya Singh (ns26z139), IIT Madras

Original Paper: Kshitija Taywade, Judy Goldsmith, Brent Harrison — University of Kentucky
