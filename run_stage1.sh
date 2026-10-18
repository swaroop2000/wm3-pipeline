#!/usr/bin/env bash
# Stage 1 in one go. Puts the .pt in weights/, unzips data.zip into sample/, inspects both,
# runs the model smoke test, then pulls one real GFS cycle. Everything is also saved to /workspace/stage1.log
set -uo pipefail
source /workspace/venv/bin/activate
cd /workspace/wm3-pipeline
{
echo "##### $(date -u) #####"
PT=$(ls -1 /workspace/weights/*.pt /workspace/*.pt 2>/dev/null | head -1)
[ -z "$PT" ] && { echo "NO .pt FILE FOUND in /workspace"; exit 1; }
case "$PT" in /workspace/weights/*) ;; *) mv "$PT" /workspace/weights/ && PT=/workspace/weights/$(basename "$PT");; esac
echo "weights: $PT"
if [ -f /workspace/data.zip ]; then unzip -o -q /workspace/data.zip -d /workspace/sample && echo "unzipped data.zip"; fi
echo "== sample folder contents =="; find /workspace/sample -type f -exec ls -lh {} \; | awk '{print $5, $9}'

echo; echo "########## 1. INSPECT WEIGHTS + SAMPLE ##########"
python scripts/inspect_inputs.py --weights "$PT" --sample /workspace/sample
echo; echo "########## 2. SMOKE TEST ##########"
python scripts/smoke_test.py --weights "$PT"
echo; echo "########## 3. GFS FETCH ##########"
python scripts/gfs_fetch.py
echo "##### done $(date -u) #####"
} 2>&1 | tee /workspace/stage1.log
