"""CLI: fetch one GFS cycle and print a report.  python scripts/gfs_fetch.py [--cycle YYYYMMDDHH]"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wm3.gfs import fetch_cycle, newest_cycle, PR_VARS, SFC

ap = argparse.ArgumentParser(); ap.add_argument("--cycle"); ap.add_argument("--out", default="/workspace/data")
a = ap.parse_args()
cyc = a.cycle or newest_cycle()
path, info = fetch_cycle(cyc, a.out)
print("cycle", cyc, "->", path, info)
d = np.load(path)
for k in [f"{v}|{l} mb" for v in PR_VARS for l in (10, 500, 850, 1000)] + [f"{v}|{l}" for v, l in SFC]:
    x = d[k]; print(f"{k:28s} {x.min():10.4g} {x.mean():10.4g} {x.max():10.4g}")
