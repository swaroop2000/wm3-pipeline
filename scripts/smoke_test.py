"""Stage 1c: build WM-3, load weights (and SHOW what didn't load), push a fake input through, time it.
Run:  python scripts/smoke_test.py --weights /workspace/weights/<file>
Paste the whole output back to Claude."""
import argparse, os, sys, time, types, traceback
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="/workspace/WeatherMesh-3")
ap.add_argument("--weights", required=True)
ap.add_argument("--hours", type=int, default=240)
a = ap.parse_args()
weights = os.path.abspath(a.weights)
os.chdir(a.repo); sys.path.insert(0, a.repo)   # the repo loads constants/ by relative path

import model as wm
from utils import levels_gfs  # noqa

# Build with no weights first (repo's loader swallows mismatches), then load ourselves and report.
wm.load_weights = lambda m, p: None
net = wm.get_WeatherMesh3(weights)
import utils
m = types.ModuleType("utils_lite")
for k in dir(utils):
    if not k.startswith("_"): setattr(m, k, getattr(utils, k))
for c in ["WeatherTrainerConfig", "LRScheduleConfig", "DataConfig", "OptimConfig"]:
    setattr(m, c, type(c, (), {"__init__": lambda self, *x, **y: None}))
sys.modules["utils_lite"] = m
ck = torch.load(weights, map_location="cpu", weights_only=False)
sd = ck.get("model_state_dict", ck)
res = net.load_state_dict(sd, strict=False)
print(f"MISSING keys ({len(res.missing_keys)}):", res.missing_keys[:25])
print(f"UNEXPECTED keys ({len(res.unexpected_keys)}):", res.unexpected_keys[:25])
print("processors in model:", list(net.processors.keys()))
print("params: %.1fM" % (sum(p.numel() for p in net.parameters()) / 1e6))

dev = "cuda"
net = net.to(dev).eval()   # fp32 weights, fp16 math via autocast (a full .half() breaks instance norm)
g_mesh, h_mesh = net.encoders[0].mesh, net.encoders[1].mesh
print("GFS input shape", tuple(g_mesh.shape()), "| HRES input shape", tuple(h_mesh.shape()))

# Normalized zeros = "average atmosphere". Not a real forecast, just proves the plumbing.
x_g = torch.zeros((1, *g_mesh.shape()), dtype=torch.float32, device=dev)
x_h = torch.zeros((1, *h_mesh.shape()), dtype=torch.float32, device=dev)
t0 = torch.tensor([int(time.time()) // 21600 * 21600], device=dev)

stats = {}
def cb(dt, out):
    y = out[0].float()
    stats[dt] = (y.isnan().sum().item(), y.mean().item(), y.std().item(), tuple(y.shape))

try:
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        net([x_g, x_h, t0], [6], callback=cb); torch.cuda.synchronize()      # warm-up
        torch.cuda.reset_peak_memory_stats()
        dts = list(range(6, a.hours + 1, 6))
        t = time.time()
        net([x_g, x_h, t0], dts, callback=cb); torch.cuda.synchronize()
        el = time.time() - t
    print(f"\nINFERENCE OK: {len(dts)} lead times to {a.hours} h in {el:.2f} s "
          f"({el/len(dts):.3f} s per 6 h step) | peak VRAM {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
    for dt in [6, 24, 72, 120, a.hours]:
        if dt in stats:
            n, mu, s, sh = stats[dt]
            print(f"  +{dt:3d}h  shape {sh}  nan {n}  mean {mu:.4f}  std {s:.4f}")
except Exception:
    traceback.print_exc()
    print("\nINFERENCE FAILED - paste everything above")
