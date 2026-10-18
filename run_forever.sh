#!/usr/bin/env bash
# Keeps the pipeline alive: if the Python process exits for any reason, wait 30 s and start it again.
# Start in the background:   nohup bash run_forever.sh > /dev/null 2>&1 &
mkdir -p /workspace/logs
exec 9>/workspace/run_forever.lock
flock -n 9 || { echo "pipeline already running"; exit 1; }   # never two copies on one GPU
set -a; source /workspace/wm3.env; set +a
export MPLBACKEND=Agg
source /workspace/venv/bin/activate
cd /workspace/wm3-pipeline
while true; do
  echo "=== start $(date -u) ===" >> /workspace/logs/service.log
  python -u -m wm3.service >> /workspace/logs/service.log 2>&1
  echo "=== exited code $? at $(date -u), restarting in 30 s ===" >> /workspace/logs/service.log
  sleep 30
done
