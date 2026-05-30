#!/bin/bash
echo "=== Python versions ==="
find /lfs/sware /usr/local /opt -name "python3*" -type f 2>/dev/null | grep -v ".pyc" | head -20

echo ""
echo "=== Module system ==="
module avail 2>&1 | grep -i python | head -20

echo ""
echo "=== GPU info ==="
nvidia-smi 2>/dev/null || echo "nvidia-smi not found in PATH"

echo ""
echo "=== CPU info ==="
nproc
cat /proc/cpuinfo | grep "model name" | head -1

echo ""
echo "=== Memory ==="
free -h

echo ""
echo "=== PBS/SLURM scheduler ==="
which qsub 2>/dev/null && echo "PBS available" || echo "no qsub"
which sbatch 2>/dev/null && echo "SLURM available" || echo "no sbatch"
qstat 2>/dev/null | head -5

echo ""
echo "=== Disk space ==="
df -h /lfs/usrhome/oth/ns26z139/ 2>/dev/null || df -h ~
