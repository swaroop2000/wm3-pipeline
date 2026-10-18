"""Stage 1b: look inside the weights file and the sample input, and print what the model expects.
Run:  python scripts/inspect_inputs.py --weights /workspace/weights/<file> --sample /workspace/sample/<file>
Paste the whole output back to Claude."""
import argparse, os, sys, json, pickle, types
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="/workspace/WeatherMesh-3")
ap.add_argument("--weights", required=True)
ap.add_argument("--sample", required=True)
a = ap.parse_args()
os.chdir(a.repo); sys.path.insert(0, a.repo)

def shim():  # same trick the repo uses so the checkpoint unpickles
    import utils
    m = types.ModuleType("utils_lite")
    for k in dir(utils):
        if not k.startswith("_"): setattr(m, k, getattr(utils, k))
    for c in ["WeatherTrainerConfig", "LRScheduleConfig", "DataConfig", "OptimConfig"]:
        setattr(m, c, type(c, (), {"__init__": lambda self, *x, **y: None}))
    sys.modules["utils_lite"] = m

print("=" * 70, "\nWEIGHTS:", a.weights, f"({os.path.getsize(a.weights)/1e9:.2f} GB)")
shim()
ck = torch.load(a.weights, map_location="cpu", weights_only=False)
sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
if isinstance(ck, dict):
    print("top-level keys:", [k for k in ck.keys()][:20])
    for k, v in ck.items():
        if k != "model_state_dict" and not torch.is_tensor(v):
            print(f"  {k}: {str(v)[:300]}")
dtypes = {}
for v in sd.values(): dtypes[str(v.dtype)] = dtypes.get(str(v.dtype), 0) + 1
print("tensors:", len(sd), "| dtypes:", dtypes, "| params: %.1fM" % (sum(v.numel() for v in sd.values()) / 1e6))
prefixes = sorted({".".join(k.split(".")[:2]) for k in sd})
print("module groups:", prefixes)

print("=" * 70, "\nWHAT THE MODEL EXPECTS")
from meshes import LatLonGrid
from utils import levels_gfs, levels_hres, levels_medium
extra_in = ['45_tcc', '168_2d', '246_100u', '247_100v']
for name, src, lev in [("GFS encoder", "neogfs-25", levels_gfs), ("HRES encoder", "neohres-20", levels_hres)]:
    m = LatLonGrid(source=src, extra_sfc_vars=extra_in, extra_sfc_pad=9, input_levels=lev, levels=levels_medium)
    print(f"{name}: shape {tuple(m.shape())}, n_pr={m.n_pr}, sfc={m.sfc_vars}")
    print("   first/last channels:", m.full_varlist[:3], "...", m.full_varlist[-14:])
norm = json.load(open("constants/normalization.json"))
print("normalization.json keys:", list(norm.keys()))
print("  example 167_2t:", str(norm.get("167_2t"))[:200])
print("  example 129_z :", str(norm.get("129_z"))[:200])

def load(p):
    if p.endswith(".npy"): return np.load(p, allow_pickle=True)
    if p.endswith(".npz"): return dict(np.load(p, allow_pickle=True))
    if p.endswith((".pt", ".pth")): return torch.load(p, map_location="cpu", weights_only=False)
    if p.endswith((".nc", ".nc4")):
        import xarray as xr; d = xr.open_dataset(p); print(d); return {k: d[k].values for k in d.data_vars}
    if p.endswith((".pkl", ".pickle")): return pickle.load(open(p, "rb"))
    if p.endswith((".json", ".txt", ".md", ".csv")): print(open(p).read()[:1500]); return None
    print("  (unknown type) first bytes:", open(p, "rb").read(64)); return None

paths = [a.sample] if os.path.isfile(a.sample) else sorted(
    os.path.join(r, f) for r, _, fs in os.walk(a.sample) for f in fs if not f.startswith("."))
def show(name, x):
    if torch.is_tensor(x): x = x.float().numpy()
    if isinstance(x, np.ndarray) and x.dtype != object:
        print(f"{name}: shape {x.shape} dtype {x.dtype} | nan {np.isnan(x).sum() if x.dtype.kind=='f' else 0} "
              f"| min {np.nanmin(x):.4g} mean {np.nanmean(x):.4g} max {np.nanmax(x):.4g}")
        if x.ndim >= 3 and x.shape[-1] < 400:
            c = x.reshape(-1, x.shape[-1]).astype(np.float64)
            mu, sd_ = np.nanmean(c, 0), np.nanstd(c, 0)
            idx = list(range(min(6, len(mu)))) + list(range(max(6, len(mu) - 14), len(mu)))
            for i in idx: print(f"   ch{i:3d}: mean {mu[i]:10.4g}  std {sd_[i]:10.4g}")
    elif isinstance(x, dict):
        for k, v in x.items(): show(f"{name}.{k}", v)
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x): show(f"{name}[{i}]", v)
    else:
        print(f"{name}: {type(x)} {str(x)[:200]}")
for p in paths:
    print("=" * 70, "\nSAMPLE FILE:", p, f"({os.path.getsize(p)/1e6:.1f} MB)")
    obj = load(p)
    if obj is not None: show(os.path.basename(p), obj)
