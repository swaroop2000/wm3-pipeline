"""The real-time pipeline. One long-running process:

  every POLL_SEC: is there a newer complete GFS cycle?  ->  download -> preprocess -> WM-3 (to 240 h)
  -> checks -> NetCDF + API grids + maps -> upload to S3 -> update runs.json / status.json

A background thread rewrites status.json every few minutes (heartbeat), so the status page can tell
"running", "waiting for data", "failed" and "dead" apart. Errors are retried (3 tries per cycle)
and pushed to the phone via ntfy. run_forever.sh restarts this process if it ever exits.

Run:  python -u -m wm3.service
"""
import glob, os, shutil, time, traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import numpy as np

from wm3 import gfs, prep, checks, outputs
from wm3.status import Status, s3, BUCKET, key

WEIGHTS = os.environ.get("WM3_WEIGHTS", "/workspace/weights/WeatherMesh3.pt")
WORK, DATA, VERIF = "/workspace/runs", "/workspace/data", "/workspace/verif"
POLL_SEC = int(os.environ.get("POLL_SEC", "300"))
LEADS = list(range(6, 121, 6)) + [144, 168, 192, 216, 240]      # 25 saved lead times
MAP_LEADS = [24, 72, 120, 240]
VERIF_LEADS = [6, 24, 72]
KEEP_DAYS = 4
BACKFILL_HOURS = int(os.environ.get("BACKFILL_HOURS", "0"))   # on start, also run missing cycles of the last N hours
CTYPE = {".png": "image/png", ".json": "application/json", ".nc": "application/x-netcdf", ".npy": "application/octet-stream"}
G = 9.80665


def ctime(cyc): return outputs.cycle_time(cyc)
def cyc_str(t): return t.strftime("%Y%m%d%H")


def analysis(path):
    r = np.load(path)
    return {"z500": r["HGT|500 mb"][:720] * G, "t850": r["TMP|850 mb"][:720], "msl": r["PRMSL|mean sea level"][:720]}


def upload_dir(local, prefix):
    files = [os.path.join(dp, f) for dp, _, fs in os.walk(local) for f in fs]
    def up(p):
        rel = os.path.relpath(p, local).replace(os.sep, "/")
        s3.upload_file(p, BUCKET, key(f"{prefix}/{rel}"),
                       ExtraArgs={"ContentType": CTYPE.get(os.path.splitext(p)[1], "application/octet-stream")})
    with ThreadPoolExecutor(16) as ex: list(ex.map(up, files))
    return len(files), sum(os.path.getsize(p) for p in files) / 1e6


def cleanup(days=KEEP_DAYS):
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    for p in glob.glob(f"{DATA}/gfs_*_raw.npz") + glob.glob(f"{VERIF}/*.npz"):
        c = os.path.basename(p).replace("gfs_", "").split("_")[0].split(".")[0]
        try:
            if ctime(c) < cutoff: os.remove(p)
        except Exception: pass


class Pipeline:
    def __init__(self):
        self.st = Status()
        page = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dashboard", "index.html")
        s3.upload_file(page, BUCKET, key("index.html"), ExtraArgs={"ContentType": "text/html", "CacheControl": "no-cache"})
        self.st.event("info", "service starting, loading WM-3")
        from wm3.runner import WM3
        t = time.time(); self.m = WM3(WEIGHTS)
        self.ch = [self.m.idx(outputs.FIELDS[k][0]) for k in outputs.SHORT]
        self.st.event("info", f"model loaded in {time.time()-t:.1f}s")
        self.done = {r["cycle"] for r in self.st.runs if r["status"] in ("ok", "warn", "flagged")}
        self.tries = {}; self.consecutive_fail = 0
        for d in (WORK, DATA, VERIF): os.makedirs(d, exist_ok=True)

    def process(self, cyc):
        st, T = self.st, {}
        cs = checks.CheckSet()
        run_dir = os.path.join(WORK, cyc); shutil.rmtree(run_dir, ignore_errors=True); os.makedirs(run_dir)
        t_all = time.time()

        st.set(state="running", step="download GFS", cycle=cyc)
        t = time.time(); path, info = gfs.fetch_cycle(cyc, DATA, log=lambda m: st.event("warn", m))
        T["download_s"] = round(time.time() - t, 1)
        if info.get("source"): T["source"] = info["source"].split("/")[2]

        st.set(step="preprocess")
        t = time.time(); xg, xh = prep.inputs_from_gfs(path); T["preprocess_s"] = round(time.time() - t, 1)
        checks.check_inputs(cs, np.load(path), xg, xh)
        hard_in = [c for c in cs.items if not c["ok"] and c["name"].endswith("finite")]
        if hard_in:   # NaN/inf in the inputs: the model cannot produce anything meaningful
            raise RuntimeError("input has NaN/inf: " + "; ".join(f"{c['name']} ({c['detail']})" for c in hard_in[:4]))
        soft_in = [c["name"] + f" ({c['detail']})" for c in cs.items if not c["ok"]]
        if soft_in: st.event("warn", f"cycle {cyc}: unusual input, running anyway and flagging: " + "; ".join(soft_in[:3]))

        st.set(step="inference")
        fields, bad, means = {}, {}, {}
        def on(lead, y, n_bad):
            fields[lead] = {k: y[..., i] for i, k in enumerate(outputs.SHORT)}
            bad[lead] = n_bad
            means[lead] = {"2t": checks.gmean(fields[lead]["2t"]), "msl": checks.gmean(fields[lead]["msl"]),
                           "z500": checks.gmean(fields[lead]["z500"])}
        inf = self.m.forecast(xg, xh, ctime(cyc).timestamp(), LEADS, on, channels=self.ch)
        del xg, xh
        T["inference_s"] = round(inf["total_s"], 2); T["peak_vram_gb"] = round(inf["peak_vram_gb"], 1)
        T["s_per_6h_step"] = round(inf["total_s"] / (max(LEADS) / 6), 3)

        st.set(step="validate")
        t = time.time()
        for lead in LEADS: checks.check_lead(cs, lead, fields[lead], bad[lead])
        checks.check_stability(cs, means)
        now = analysis(path); past = {}
        for lead in VERIF_LEADS:
            pc = cyc_str(ctime(cyc) - timedelta(hours=lead))
            vf, ap = os.path.join(VERIF, f"{pc}.npz"), os.path.join(DATA, f"gfs_{pc}_raw.npz")
            if os.path.exists(vf) and os.path.exists(ap):
                v = np.load(vf)
                past[lead] = ({k: v[f"f{lead}_{k}"] for k in ("z500", "t850", "msl")}, analysis(ap))
        skill = checks.check_skill(cs, now, past)
        np.savez(os.path.join(VERIF, f"{cyc}.npz"),
                 **{f"f{l}_{k}": fields[l][k] for l in VERIF_LEADS for k in ("z500", "t850", "msl")})
        T["validate_s"] = round(time.time() - t, 1)
        summ = cs.summary()
        hard_fail = [f for f in summ["failed"] if " finite" in f]
        status = "failed" if hard_fail else ("flagged" if summ["failed"] else "ok")

        # Tiny overshoots (cloud cover -0.01, humidity -1e-6) are model noise: clip them, but count and report how many.
        clipped = {"tcc_cells": 0, "q_cells": 0}
        for lead in LEADS:
            f = fields[lead]
            clipped["tcc_cells"] += int(((f["tcc"] < 0) | (f["tcc"] > 1)).sum()); f["tcc"] = np.clip(f["tcc"], 0, 1)
            for l in outputs.LEVELS_SAVED:
                clipped["q_cells"] += int((f[f"q{l}"] < 0).sum()); f[f"q{l}"] = np.maximum(f[f"q{l}"], 0)
        clipped["fraction_of_values"] = round((clipped["tcc_cells"] + clipped["q_cells"]) / (len(LEADS) * 4 * 720 * 1440), 6)

        st.set(step="write outputs")
        t = time.time()
        os.makedirs(os.path.join(run_dir, "forecast"), exist_ok=True)
        for lead in LEADS:   # sequential on purpose: the netCDF4/HDF5 C library crashes (segfault) when written from threads
            outputs.write_netcdf(os.path.join(run_dir, "forecast", f"wm3_{cyc}_f{lead:03d}.nc"), cyc, lead, fields[lead])
            outputs.write_api(run_dir, lead, fields[lead])
        maps = []
        try:   # maps are nice-to-have: a plotting error must never cost us a forecast
            for lead in MAP_LEADS: maps += outputs.write_maps(os.path.join(run_dir, "maps"), cyc, lead, fields[lead])
        except Exception as e:
            st.event("warn", f"maps failed for {cyc}: {e}")
        with open(os.path.join(run_dir, "checks.json"), "w") as fh:
            import json; json.dump({"summary": summ, "items": cs.items, "skill": skill, "global_means": means}, fh, indent=1)
        T["write_s"] = round(time.time() - t, 1)
        manifest = {"model": "WeatherMesh-3", "initialization_time": ctime(cyc).isoformat(), "cycle": cyc,
                    "initial_condition": f"NOAA GFS 0.25 analysis {cyc} (both encoders; HRES levels emulated)",
                    "forecast_hours": LEADS, "status": status, "checks": summ, "clipped": clipped, "skill": skill, "timings": T,
                    "files": {"netcdf": [f"forecast/wm3_{cyc}_f{l:03d}.nc" for l in LEADS],
                              "api_grids": {n: [f"api/{n}/{l:03d}.npy" for l in LEADS if l <= outputs.API_MAX_LEAD]
                                            for n in outputs.API},
                              "api_units": {n: u for n, (_, _, u) in outputs.API.items()},
                              "maps": [f"maps/{m}" for m in maps]},
                    "grid": {"lat": "90 to -89.75 step -0.25 (720)", "lon": "0 to 359.75 step 0.25 (1440)"}}
        outputs.write_manifest(run_dir, manifest)

        st.set(step="upload to S3")
        t = time.time(); n, mb = upload_dir(run_dir, f"runs/{cyc}")
        T["upload_s"] = round(time.time() - t, 1); T["upload_mb"] = round(mb)
        T["total_s"] = round(time.time() - t_all, 1)
        manifest["timings"] = T; outputs.write_manifest(run_dir, manifest)
        s3.upload_file(os.path.join(run_dir, "manifest.json"), BUCKET, key(f"runs/{cyc}/manifest.json"),
                       ExtraArgs={"ContentType": "application/json", "CacheControl": "no-cache"})

        if status == "ok" and cyc >= max(self.done | {cyc}):   # latest.json: newest cycle that passed every check (backfill never moves it back)
            from wm3.status import put_json
            put_json("latest.json", {"cycle": cyc, "initialization_time": ctime(cyc).isoformat(),
                                     "manifest": f"runs/{cyc}/manifest.json", "updated": datetime.now(timezone.utc).isoformat()})
        st.record_run({"cycle": cyc, "status": status, "timings": T, "checks": summ,
                       "skill": skill_row(skill), "rated": bool(skill),
                       "outputs_prefix": f"runs/{cyc}/", "maps": maps})
        shutil.rmtree(run_dir, ignore_errors=True)
        cleanup()
        msg = f"cycle {cyc} {status}: {summ['passed']}/{summ['total']} checks, inference {T['inference_s']}s, total {T['total_s']}s, {n} files {mb:.0f} MB"
        if not skill: msg += " (skill not rated yet: needs older runs)"
        if status == "failed": st.event("error", msg + " | FAILED: " + "; ".join(hard_fail[:5]))
        elif status == "flagged":
            st.event("warn", msg + " | FLAGGED (published, needs review): " + "; ".join(summ["failed"][:5]))
            st.alert(f"cycle {cyc} flagged: " + "; ".join(summ["failed"][:3]))
        else: st.event("info", msg)
        self.rescore()
        return status

    def rescore(self):
        """Backfill/out-of-order runs: a cycle processed before its older neighbours could not be scored.
        Re-score every published cycle whose 6/24/72 h comparisons have become possible since."""
        changed = False
        for row in self.st.runs:
            if row.get("status") not in ("ok", "flagged", "warn"): continue
            cyc = row["cycle"]; have = set((row.get("skill") or {}).keys())
            ap = os.path.join(DATA, f"gfs_{cyc}_raw.npz")
            if not os.path.exists(ap): continue
            past = {}
            for lead in VERIF_LEADS:
                if str(lead) in have: continue
                pc = cyc_str(ctime(cyc) - timedelta(hours=lead))
                vf, pa = os.path.join(VERIF, f"{pc}.npz"), os.path.join(DATA, f"gfs_{pc}_raw.npz")
                if os.path.exists(vf) and os.path.exists(pa):
                    v = np.load(vf); past[lead] = ({k: v[f"f{lead}_{k}"] for k in ("z500", "t850", "msl")}, analysis(pa))
            if not past: continue
            cs = checks.CheckSet(); sk = checks.check_skill(cs, analysis(ap), past)
            row["skill"] = {**(row.get("skill") or {}), **skill_row(sk)}; row["rated"] = True
            bad = cs.summary()["failed"]
            if bad:
                row["status"] = "flagged"; row.setdefault("checks", {}).setdefault("failed", []).extend(bad)
                self.st.event("warn", f"re-scored {cyc}: FLAGGED " + "; ".join(bad[:3]))
            else:
                self.st.event("info", f"re-scored {cyc}: " + ", ".join(f"{l} h z500 {sk[l]['z500']['model']} m vs {sk[l]['z500']['persistence']} m" for l in sk))
            changed = True
        if changed:
            from wm3.status import put_json
            put_json("runs.json", self.st.runs)

    def run_cycle(self, cyc):
        """Process one cycle with failure bookkeeping. Returns True on success."""
        st = self.st
        try:
            self.process(cyc); self.done.add(cyc); self.consecutive_fail = 0
            return True
        except Exception:
            n = self.tries[cyc] = self.tries.get(cyc, 0) + 1
            self.consecutive_fail += 1
            tb = traceback.format_exc().strip().splitlines()[-1][:300]
            level = "error" if n in (1, 3) or n % 6 == 0 else "warn"      # alert on 1st, 3rd, then every 6th try
            st.event(level, f"cycle {cyc} attempt {n} failed: {tb}")
            st.record_run({"cycle": cyc, "status": "failed", "timings": {}, "checks": {"failed": [tb]}, "outputs_prefix": None})
            try: os.remove(os.path.join(DATA, f"gfs_{cyc}_raw.npz"))   # never retry on a possibly bad cached file
            except OSError: pass
            st.set(state="failed", step=f"will retry {cyc}")
            if self.consecutive_fail >= 3:     # e.g. a broken CUDA context: a fresh process is the cleanest fix
                st.event("error", "3 failures in a row: restarting the process for a clean GPU state")
                os._exit(4)                    # run_forever.sh / Docker restarts us
            return False

    def backfill(self, hours):
        """Run any missing cycles of the last `hours`, OLDEST FIRST, so each later cycle can score earlier forecasts."""
        todo = [c for c in reversed(gfs.recent_cycles(hours)) if c not in self.done]
        todo = [c for c in todo if gfs.available(c)]
        if todo: self.st.event("info", f"backfill: {len(todo)} cycles {todo[0]} -> {todo[-1]}")
        for c in todo: self.run_cycle(c)
        self.rescore()

    def loop(self):
        st = self.st; last_stale_note = 0
        if BACKFILL_HOURS: self.backfill(BACKFILL_HOURS)
        while True:
            if checks_disk() < 5:
                st.event("warn", "disk below 5 GB free: deleting cached inputs older than 1 day"); cleanup(days=1)
            try:
                cyc = gfs.newest_cycle()
            except Exception as e:
                st.event("warn", f"could not list GFS cycles: {e}"); time.sleep(POLL_SEC); continue
            if cyc not in self.done:
                if not self.run_cycle(cyc):
                    time.sleep(POLL_SEC); continue      # keep retrying until it works or a newer cycle replaces it
            newest_ok = max(self.done) if self.done else None
            age_h = (datetime.now(timezone.utc) - ctime(newest_ok)).total_seconds() / 3600 if newest_ok else 0
            if age_h > 13 and time.time() - last_stale_note > 3600:
                st.event("error" if age_h > 18 else "warn", f"newest good forecast is {age_h:.1f} h old (normal max ~10 h)")
                last_stale_note = time.time()
            st.set(state="idle", step=f"waiting for next GFS cycle (latest done {newest_ok or '-'})")
            time.sleep(POLL_SEC)


def skill_row(skill):
    return {str(k): {kk: vv["model"] for kk, vv in v.items()} | {"persist_z500": v["z500"]["persistence"]} for k, v in skill.items()}


def checks_disk():
    st = os.statvfs("/workspace" if os.path.isdir("/workspace") else "/"); return st.f_bavail * st.f_frsize / 1e9


if __name__ == "__main__":
    import sys
    p = Pipeline()
    if "--once" in sys.argv:            # test mode: process the newest cycle (or --cycle X) once and exit
        c = sys.argv[sys.argv.index("--cycle") + 1] if "--cycle" in sys.argv else gfs.newest_cycle()
        print("RESULT:", p.process(c))
    elif "--backfill" in sys.argv:      # e.g. --backfill 72 : run missing past cycles, then exit
        p.backfill(int(sys.argv[sys.argv.index("--backfill") + 1])); print("BACKFILL DONE")
    else:
        p.loop()
