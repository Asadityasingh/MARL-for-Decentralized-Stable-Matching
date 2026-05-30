#!/bin/bash
# run.sh — unified control script
# Local:   ./run.sh validate | full | status | log | stop
# Cluster: ./run.sh submit [--dry-run] [--resume] [--config NAME]
#          ./run.sh jobs        (check PBS queue)
#          ./run.sh progress    (count completed runs)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$SCRIPT_DIR/venv/bin/activate"
LOG="$SCRIPT_DIR/train.log"
PID_FILE="$SCRIPT_DIR/train.pid"
PYTHON="/lfs/usrhome/oth/ns26z139/Stable_matching/venv/bin/python"

case "$1" in
  validate|full)
    if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
      echo "Training already running (PID $(cat $PID_FILE)). Use './run.sh stop' first."
      exit 1
    fi
    echo "Starting training: mode=$1"
    echo "Log: $LOG"
    nohup bash -c "
      source $VENV 2>/dev/null || true
      cd $SCRIPT_DIR
      $PYTHON main.py --mode $1 ${2:-} ${3:-}
    " > "$LOG" 2>&1 &
    echo $! > "$PID_FILE"
    echo "PID $(cat $PID_FILE) - safe to close terminal."
    echo "Watch: ./run.sh log"
    ;;

  submit)
    # Cluster: generate and submit all PBS jobs
    source $VENV 2>/dev/null || true
    cd $SCRIPT_DIR
    shift
    $PYTHON submit_jobs.py "$@"
    ;;

  jobs)
    # Check PBS queue
    echo "=== Your PBS jobs ==="
    qstat -u $USER 2>/dev/null || echo "qstat not available"
    ;;

  progress)
    # Count completed runs across all configs
    echo "=== Completed runs ==="
    total=0
    done=0
    for dir in results/*/; do
      config=$(basename $dir)
      n=$(find $dir -name "results.json" 2>/dev/null | wc -l)
      echo "  $config: $n/50"
      done=$((done + n))
      total=$((total + 50))
    done
    echo ""
    echo "Total: $done / $total runs complete"
    ;;

  status)
    if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
      echo "Training RUNNING (PID $(cat $PID_FILE))"
      tail -3 "$LOG" 2>/dev/null
    else
      echo "No local training running"
    fi
    ;;

  log)
    echo "Tailing $LOG (Ctrl+C to stop, training continues)"
    tail -f "$LOG"
    ;;

  stop)
    if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
      kill $(cat "$PID_FILE") && rm -f "$PID_FILE"
      echo "Stopped."
    else
      echo "Nothing running."
    fi
    ;;

  *)
    echo "Usage:"
    echo "  Local training:"
    echo "    ./run.sh validate          start validation run"
    echo "    ./run.sh full              start full run"
    echo "    ./run.sh status            check if running"
    echo "    ./run.sh log               tail live log"
    echo "    ./run.sh stop              stop training"
    echo ""
    echo "  Cluster (PBS):"
    echo "    ./run.sh submit            submit all 498 jobs to gpuq"
    echo "    ./run.sh submit --dry-run  preview without submitting"
    echo "    ./run.sh submit --resume   skip completed runs"
    echo "    ./run.sh submit --config 3x3_8agents_SM_sym"
    echo "    ./run.sh jobs              check PBS queue"
    echo "    ./run.sh progress          count completed runs"
    ;;
esac
