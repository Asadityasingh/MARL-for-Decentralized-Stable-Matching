#!/bin/bash
#PBS -N MARL_BATCH
#PBS -q verylong
#PBS -l select=1:ncpus=32:mem=128gb
#PBS -l walltime=47:00:00
#PBS -o pbs_logs/output_BATCH_NAME.log
#PBS -e pbs_logs/error_BATCH_NAME.log

# Full path - no module load needed on remote nodes
PYTHON3=/lfs/sware/anaconda3_2023/envs/torch_gpu/bin/python
WORKDIR=/lfs/usrhome/oth/ns26z139/Stable_matching

timestamp() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

timestamp "Job started: BATCH_NAME"
timestamp "PBS Job ID: $PBS_JOBID"
timestamp "Node: $(hostname)"

export OMP_NUM_THREADS=2
cd $WORKDIR
module load anaconda3_2024.10

timestamp "Python: $($PYTHON3 --version)"

CONFIGS=(BATCH_CONFIGS)
N_CONFIGS=${#CONFIGS[@]}

timestamp "Node: $(hostname) | Configs: $N_CONFIGS"

# Run configs sequentially on this single node
for CONFIG in "${CONFIGS[@]}"; do
    DONE=$(find $WORKDIR/results/$CONFIG -name "results.json" 2>/dev/null | wc -l)
    if [ "$DONE" -ge 50 ]; then
        timestamp "Skip $CONFIG (complete)"
        continue
    fi

    timestamp "Launch $CONFIG ($DONE/50 done)"
    cd $WORKDIR && \
        CUDA_VISIBLE_DEVICES='' \
        OMP_NUM_THREADS=2 \
        $PYTHON3 cluster_runner.py \
        --config $CONFIG \
        --output $WORKDIR/results \
        --nworkers 16 \
        --wandb-entity ns26z139-iit-madras \
        > $WORKDIR/pbs_logs/${CONFIG}.log 2>&1
    timestamp "Done $CONFIG"
done

# Resubmit if pending
PENDING=0
for CONFIG in BATCH_CONFIGS; do
    DONE=$(find $WORKDIR/results/$CONFIG -name "results.json" 2>/dev/null | wc -l)
    [ "$DONE" -lt 50 ] && PENDING=$((PENDING+1))
done

if [ "$PENDING" -gt 0 ]; then
    timestamp "$PENDING configs pending - resubmitting"
    qsub $WORKDIR/pbs_scripts/BATCH_NAME.cmd
else
    timestamp "ALL configs in BATCH_NAME complete!"
fi

timestamp "Job finished: BATCH_NAME"
