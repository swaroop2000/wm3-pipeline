"""Stage 2 checks. Paste the whole output back.
 A. Our GFS preprocessing vs WindBorne's sample for the same time (2025-03-07 00Z): must match closely.
 B. Model run on the sample inputs and on our own GFS inputs for that date: forecasts should be close.
 C. Skill: our 24 h / 72 h forecast vs the later GFS analyses, compared with doing nothing (persistence).
 D. Model run on the newest real-time GFS cycle + timing.
"""
import calendar, glob, os, subprocess, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wm3 import prep
from wm3.runner import WM3

DATA = "/workspace/data"; SAMPLE = "/workspace/sample"
W = "/workspace/weights/WeatherMesh3.pt"
PY = sys.executable; HERE = os.path.dirname(os.path.abspath(__file__))
G = 9.80665
lat = np.arange(90, -90.01, -0.25)[:-1]; wlat = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, 1440))
wlat /= wlat.mean()

def fetch(cyc):
    p = f"{DATA}/gfs_{cyc}_raw.npz"
    if not os.path.exists(p):
        subprocess.run([PY, f"{HERE}/gfs_fetch.py", "--cycle", cyc], check=True, stdout=subprocess.DEVNULL)
    return p

def rmse(a, b): return float(np.sqrt(np.mean(wlat * (a - b) ** 2)))
def gmean(a): return float(np.mean(wlat * a))

def analysis_fields(p):
    r = np.load(p)
    return {"z500": r["HGT|500 mb"][:720] * G, "t850": r["TMP|850 mb"][:720], "2t": r["TMP|2 m above ground"][:720],
            "msl": r["PRMSL|mean sea level"][:720]}

print("#" * 20, "A. preprocessing vs WindBorne sample (2025-03-07 00Z)")
smp_dir = glob.glob(f"{SAMPLE}/**/wm3/data", recursive=True)[0]
f = glob.glob(f"{smp_dir}/neogfs/f000/**/*.npz", recursive=True)[0]
S = np.load(f); spr = S["pr"].astype(np.float32); ssf = S["sfc"].astype(np.float32)
t = time.time(); p25 = fetch("2025030700"); print(f"fetched GFS 2025030700 in {time.time()-t:.1f}s")
opr, osf, oex = prep.gfs_25_normalized(p25)
print("var        level   our-vs-sample RMSE (normalized units)   correlation")
for i, v in enumerate(prep.PR_VARS):
    for l in (50, 250, 500, 850, 1000):
        j = prep.LEVELS_GFS.index(l); a, b = opr[..., i, j], spr[..., i, j]
        print(f"{v:9s} {l:5d}   {np.sqrt(np.mean((a-b)**2)):8.4f}   {np.corrcoef(a.ravel(), b.ravel())[0,1]:.5f}")
for i, (n, _) in enumerate(prep.CORE_SFC):
    a, b = osf[..., i], ssf[..., i]
    print(f"{n:9s}   sfc   {np.sqrt(np.mean((a-b)**2)):8.4f}   {np.corrcoef(a.ravel(), b.ravel())[0,1]:.5f}")
for n, _ in prep.EXTRA_SFC:
    b = np.load(glob.glob(f"{smp_dir}/neogfs/extra/{n}/**/*.npz", recursive=True)[0])["x"].astype(np.float32)
    a = oex[n]
    print(f"{n:9s}   sfc   {np.sqrt(np.mean((a-b)**2)):8.4f}   {np.corrcoef(a.ravel(), b.ravel())[0,1]:.5f}")

print("\n" + "#" * 20, "B/C. model runs for 2025-03-07 00Z")
t = time.time(); m = WM3(W); print(f"model loaded in {time.time()-t:.1f}s; outputs {len(m.out_vars)} channels")
K = {"z500": m.idx("129_z_500"), "t850": m.idx("130_t_850"), "2t": m.idx("167_2t"), "msl": m.idx("151_msl"),
     "u250": m.idx("131_u_250"), "v250": m.idx("132_v_250"), "tcc": m.idx("45_tcc"), "q850": m.idx("133_q_850")}
LEADS = [6, 24, 72, 120, 240]

def run(xg, xh, t0, label):
    keep = {}
    def on(dt, y, n_bad): keep[dt] = {k: y[..., i].copy() for k, i in K.items()} | {"nan": int(np.isnan(y).sum())}
    info = m.forecast(xg, xh, t0, LEADS, on)
    print(f"\n[{label}] {len(LEADS)} leads to {max(LEADS)} h in {info['total_s']:.2f}s, peak VRAM {info['peak_vram_gb']:.1f} GB")
    print(" lead  nan   2t mean/min/max (K)        msl mean/min/max (hPa)     z500 mean (m)  max wind250  tcc mean")
    for dt in LEADS:
        o = keep[dt]; ws = np.hypot(o["u250"], o["v250"])
        print(f" {dt:4d} {o['nan']:4d}  {gmean(o['2t']):6.1f} {o['2t'].min():6.1f} {o['2t'].max():6.1f}   "
              f"{gmean(o['msl'])/100:7.1f} {o['msl'].min()/100:7.1f} {o['msl'].max()/100:7.1f}   "
              f"{gmean(o['z500'])/G:8.1f}   {ws.max():6.1f}    {gmean(o['tcc']):.3f}")
    return keep

xs_g, xs_h, ts = prep.inputs_from_sample(smp_dir)
print("sample time", ts, time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(ts)))
F_s = run(xs_g, xs_h, ts, "WindBorne sample inputs (GFS + real HRES)")
xg, xh = prep.inputs_from_gfs(p25)
F_g = run(xg, xh, ts, "OUR GFS-only inputs, same time")
print("\nour-GFS forecast vs sample-input forecast (RMSE):")
for dt in (24, 120):
    print(f"  +{dt}h  z500 {rmse(F_g[dt]['z500'], F_s[dt]['z500'])/G:.1f} m   t850 {rmse(F_g[dt]['t850'], F_s[dt]['t850']):.2f} K   2t {rmse(F_g[dt]['2t'], F_s[dt]['2t']):.2f} K")

print("\nskill vs later GFS analyses (lower is better; 'persist' = assume weather never changes):")
A0 = analysis_fields(p25)
for dt, cyc in [(24, "2025030800"), (72, "2025031000"), (120, "2025031200")]:
    A = analysis_fields(fetch(cyc))
    print(f"  +{dt:3d}h  z500 model {rmse(F_g[dt]['z500'], A['z500'])/G:6.1f} m | persist {rmse(A0['z500'], A['z500'])/G:6.1f} m"
          f"    t850 model {rmse(F_g[dt]['t850'], A['t850']):5.2f} K | persist {rmse(A0['t850'], A['t850']):5.2f} K"
          f"    msl model {rmse(F_g[dt]['msl'], A['msl'])/100:5.2f} hPa | persist {rmse(A0['msl'], A['msl'])/100:5.2f} hPa")

print("\n" + "#" * 20, "D. newest real-time cycle")
newest = sorted(glob.glob(f"{DATA}/gfs_2026*_raw.npz"))[-1]
cyc = os.path.basename(newest)[4:14]
t0 = calendar.timegm(time.strptime(cyc, "%Y%m%d%H"))
t = time.time(); xg, xh = prep.inputs_from_gfs(newest); print(f"{cyc}: preprocess {time.time()-t:.1f}s")
run(xg, xh, t0, f"real-time GFS {cyc}")
print("\nDONE")
