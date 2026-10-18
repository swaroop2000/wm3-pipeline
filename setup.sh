#!/usr/bin/env bash
# One-time setup on the RunPod pod. Run from /workspace/wm3-pipeline:  bash setup.sh
# Uses a venv so we control torch exactly (the WM-3 repo pins clash: torch 2.5.1 vs a NATTEN wheel built for torch 2.4.0+cu121).
set -euo pipefail
WS=/workspace
mkdir -p "$WS"   # no network volume needed; container disk is fine (outputs live in S3)
cd "$WS"
echo "== python ==" ; python3 --version

echo "== GPU =="; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

if [ ! -d "$WS/venv" ]; then python3 -m venv "$WS/venv"; fi
source "$WS/venv/bin/activate"
python -m pip install -q --upgrade pip wheel

echo "== torch 2.4.0 + CUDA 12.1 =="
pip install -q torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
echo "== NATTEN 0.17.3 (matched to torch 2.4.0 cu121) =="
pip install -q "natten==0.17.3+torch240cu121" -f https://shi-labs.com/natten/wheels/ --trusted-host shi-labs.com   # their SSL cert expired (Oct 2026)
echo "== matepoint (from test.pypi, no deps so it can't pull a different torch) =="
pip install -q --no-deps --index-url https://test.pypi.org/simple/ matepoint
echo "== data / IO libs =="
pip install -q "numpy<2" xarray cfgrib eccodes eccodeslib netCDF4 boto3 requests matplotlib gdown

if [ ! -d "$WS/WeatherMesh-3" ]; then git clone -q https://github.com/windborne/WeatherMesh-3.git "$WS/WeatherMesh-3"; fi
mkdir -p "$WS/weights" "$WS/sample" "$WS/data" "$WS/out"

echo "== version check =="
python - <<'PY'
import torch, natten, numpy, matepoint, cfgrib, boto3
print("torch", torch.__version__, "| cuda ok:", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0))
print("natten", natten.__version__, "| numpy", numpy.__version__, "| cfgrib", cfgrib.__version__)
PY
echo "SETUP OK"
