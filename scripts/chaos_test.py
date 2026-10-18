"""Fault-injection tests: prove the pipeline survives / rejects the things that go wrong in real life.
Runs in a temp folder; does NOT touch the live service, S3 or the GPU.   python scripts/chaos_test.py"""
import os, shutil, sys, tempfile, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wm3 import gfs, prep, checks

results = []
def report(name, ok, detail=""):
    results.append((name, ok)); print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)

tmp = tempfile.mkdtemp(dir="/workspace" if os.path.isdir("/workspace") else None)
cyc = gfs.newest_cycle(); print("using cycle", cyc)
REAL = list(gfs.SOURCES)

# 1. each mirror on its own can deliver a complete cycle
for base in REAL:
    gfs.SOURCES[:] = [base]
    d = os.path.join(tmp, base.split("/")[2]); t = time.time()
    try:
        p, info = gfs.fetch_cycle(cyc, d, log=lambda m: None)
        report(f"mirror {base.split('/')[2]} delivers all {len(gfs.WANT)} fields", True, f"{info.get('mb')} MB in {time.time()-t:.1f}s")
    except Exception as e:
        report(f"mirror {base.split('/')[2]} delivers all fields", False, str(e)[:120])

# 2. primary mirror dead -> automatic failover
gfs.SOURCES[:] = ["https://no-such-host.invalid"] + REAL
d = os.path.join(tmp, "failover")
try:
    p, info = gfs.fetch_cycle(cyc, d, log=lambda m: None)
    report("dead primary mirror -> falls over to the next one", info["source"] != gfs.SOURCES[0], f"used {info['source'].split('/')[2]}")
except Exception as e:
    report("dead primary mirror -> failover", False, str(e)[:120])
gfs.SOURCES[:] = REAL

# 3. corrupted cached file -> detected and downloaded again
open(p, "wb").write(b"garbage"); t = time.time()
p2, info2 = gfs.fetch_cycle(cyc, d, log=lambda m: None)
report("corrupted cached input is detected and re-downloaded", not info2.get("cached") and np.load(p2)["TMP|2 m above ground"].shape == (721, 1440))

# 4. truncated (half-written) cache -> detected
np.savez(p2, **{"TMP|2 m above ground": np.zeros((721, 1440), np.float32)})
p3, info3 = gfs.fetch_cycle(cyc, d, log=lambda m: None)
report("incomplete cached input (1 of 138 fields) is re-downloaded", not info3.get("cached"))

# 5. a cycle that does not exist -> clear error, no crash loop
try:
    gfs.fetch_cycle("2099010100", d, log=lambda m: None); report("missing cycle raises a clear error", False)
except RuntimeError as e:
    report("missing cycle raises a clear error", "every mirror" in str(e))

# 6. bad INPUT values are rejected before the model runs
raw = dict(np.load(p3)); xg, xh = prep.inputs_from_gfs(p3)
cs = checks.CheckSet(); checks.check_inputs(cs, raw, xg, xh)
report("real GFS input passes all input checks", not cs.summary()["failed"], f"{cs.summary()['passed']}/{cs.summary()['total']}")
for label, field, val in [("NaN in 2 m temperature", "TMP|2 m above ground", np.nan),
                          ("2 m temperature in Celsius (unit bug)", "TMP|2 m above ground", None),
                          ("pressure in hPa instead of Pa", "PRMSL|mean sea level", None)]:
    bad = dict(raw); a = bad[field].copy()
    if val is not None: a[100:110, 200:210] = val
    elif "Celsius" in label: a = a - 273.15
    else: a = a / 100
    bad[field] = a
    cs = checks.CheckSet(); checks.check_inputs(cs, bad, xg, xh)
    report(f"bad input caught: {label}", bool(cs.summary()["failed"]), "; ".join(cs.summary()["failed"])[:110])

# 7. bad OUTPUT values fail the cycle
good = {"2t": raw["TMP|2 m above ground"][:720], "2d": raw["DPT|2 m above ground"][:720], "msl": raw["PRMSL|mean sea level"][:720],
        "z500": raw["HGT|500 mb"][:720] * 9.80665, "t850": raw["TMP|850 mb"][:720], "t250": raw["TMP|250 mb"][:720],
        "tcc": raw["TCDC|entire atmosphere"][:720] / 100, "q850": raw["SPFH|850 mb"][:720], "q500": raw["SPFH|500 mb"][:720],
        "q250": raw["SPFH|250 mb"][:720], "10u": raw["UGRD|10 m above ground"][:720], "10v": raw["VGRD|10 m above ground"][:720],
        "100u": raw["UGRD|100 m above ground"][:720], "100v": raw["VGRD|100 m above ground"][:720],
        "u250": raw["UGRD|250 mb"][:720], "v250": raw["VGRD|250 mb"][:720], "u850": raw["UGRD|850 mb"][:720], "v850": raw["VGRD|850 mb"][:720]}
cs = checks.CheckSet(); checks.check_lead(cs, 6, good, 0)
report("a physically real field set passes all output checks", not cs.summary()["failed"], f"{cs.summary()['passed']}/{cs.summary()['total']}")
for label, mutate, nbad in [("NaN values from the model", lambda f: f, 5),
                            ("exploding temperature (+80 K)", lambda f: {**f, "2t": f["2t"] + 80}, 0),
                            ("wind blow-up at 250 hPa", lambda f: {**f, "u250": f["u250"] * 4}, 0),
                            ("cloud cover way out of range", lambda f: {**f, "tcc": f["tcc"] * 5}, 0),
                            ("dewpoint above temperature", lambda f: {**f, "2d": f["2t"] + 5}, 0)]:
    cs = checks.CheckSet(); checks.check_lead(cs, 6, mutate(good), nbad)
    report(f"bad output caught: {label}", bool(cs.summary()["failed"]), "; ".join(cs.summary()["failed"])[:110])

shutil.rmtree(tmp, ignore_errors=True)       # the test downloads ~600 MB into a temp folder; remove it
n_ok = sum(ok for _, ok in results)
print(f"\n{n_ok}/{len(results)} fault tests passed")
