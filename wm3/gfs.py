"""Download one GFS 0.25 deg analysis (f000): only the ~138 GRIB messages WM-3 needs, using .idx byte offsets.

Hardening (the "pull must always work" part):
  * Three independent mirrors, tried in order: NOAA's AWS bucket, Google Cloud's copy, NOAA NOMADS.
    If one is down, slow or missing a file, the next one is used.
  * A cycle only counts as available when f000 AND f006 indexes exist (so f000 is surely fully uploaded).
  * Every byte range is retried 4x with backoff; every decoded field must be 721x1440 and finite.
  * Missing fields raise a clear error (that cycle is retried later and an alert is sent).
  * The saved .npz is written to a temp file and renamed, so a crash can never leave a half file behind.
"""
import os, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import numpy as np, requests

SOURCES = [s.strip() for s in os.environ.get("GFS_SOURCES", ",".join([
    "https://noaa-gfs-bdp-pds.s3.amazonaws.com",
    "https://storage.googleapis.com/global-forecast-system",
    "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod",
])).split(",") if s.strip()]
LEVELS_GFS = [10, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 925, 950, 975, 1000]
LEVELS_HRES = [10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 850, 900, 925, 950, 1000]
PR_LEVELS = sorted(set(LEVELS_GFS) | set(LEVELS_HRES))          # 26 levels (GFS list + 20 hPa)
PR_VARS = ["HGT", "TMP", "UGRD", "VGRD", "SPFH"]
SFC = [("UGRD", "10 m above ground"), ("VGRD", "10 m above ground"), ("TMP", "2 m above ground"),
       ("DPT", "2 m above ground"), ("PRMSL", "mean sea level"),
       ("UGRD", "100 m above ground"), ("VGRD", "100 m above ground"), ("TCDC", "entire atmosphere")]
WANT = {(v, f"{l} mb") for v in PR_VARS for l in PR_LEVELS} | set(SFC)


def url(cyc, fh=0, base=None):
    d, h = cyc[:8], cyc[8:]
    return f"{base or SOURCES[0]}/gfs.{d}/{h}/atmos/gfs.t{h}z.pgrb2.0p25.f{fh:03d}"


def get_idx(cyc, fh=0, base=None):
    for attempt in range(3):
        try:
            r = requests.get(url(cyc, fh, base) + ".idx", timeout=30)
            if r.status_code in (403, 404): return None
            r.raise_for_status()
            rows = []
            for line in r.text.strip().splitlines():
                p = line.split(":")
                rows.append({"start": int(p[1]), "var": p[3], "lev": p[4], "kind": p[5]})
            for i, row in enumerate(rows):
                row["end"] = rows[i + 1]["start"] - 1 if i + 1 < len(rows) else None
            return rows
        except Exception:
            time.sleep(2 ** attempt)
    return None


def _exists(u):
    try: return requests.head(u, timeout=15).status_code == 200
    except Exception: return False


def available(cyc):
    """First mirror where this cycle is fully published (f000 + f006 idx present), or None."""
    for base in SOURCES:
        if _exists(url(cyc, 0, base) + ".idx") and _exists(url(cyc, 6, base) + ".idx"):
            return base
    return None


def recent_cycles(hours_back=36):
    t = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    t -= timedelta(hours=t.hour % 6)
    return [(t - timedelta(hours=6 * i)).strftime("%Y%m%d%H") for i in range(hours_back // 6 + 1)]


def newest_cycle(hours_back=36):
    for cyc in recent_cycles(hours_back):
        if available(cyc): return cyc
    raise RuntimeError(f"no complete GFS cycle on any mirror in the last {hours_back} h")


def fetch_range(u, row):
    hdr = {"Range": f"bytes={row['start']}-{'' if row['end'] is None else row['end']}"}
    for attempt in range(4):
        try:
            r = requests.get(u, headers=hdr, timeout=60); r.raise_for_status()
            if row["end"] is not None and len(r.content) != row["end"] - row["start"] + 1:
                raise IOError("short read")
            return r.content
        except Exception:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed {row['var']} {row['lev']}")


def decode(msg):
    import eccodes                     # imported lazily: only the download step needs the GRIB library
    gid = eccodes.codes_new_from_message(msg)
    try:
        nj, ni = eccodes.codes_get(gid, "Nj"), eccodes.codes_get(gid, "Ni")
        a = eccodes.codes_get_values(gid).reshape(nj, ni).astype(np.float32)
        if a.shape != (721, 1440) or eccodes.codes_get(gid, "latitudeOfFirstGridPointInDegrees") != 90:
            raise ValueError(f"unexpected grid {a.shape}")
        if not np.isfinite(a).all(): raise ValueError("non-finite values in GRIB field")
        return a, {}
    finally:
        eccodes.codes_release(gid)


def _fetch_from(cyc, base):
    idx = get_idx(cyc, 0, base)
    if idx is None: raise RuntimeError("f000 idx missing")
    picked, src = {}, {}
    for row in idx:
        k = (row["var"], row["lev"])
        if k in WANT and k not in picked: picked[k] = row; src[k] = 0
    tcc = ("TCDC", "entire atmosphere")
    if tcc not in picked:          # cloud cover is not always in the analysis file: use the 6 h forecast
        cands = [r for r in (get_idx(cyc, 6, base) or []) if (r["var"], r["lev"]) == tcc]
        inst = [r for r in cands if "ave" not in r["kind"]] or cands
        if inst: picked[tcc] = inst[0]; src[tcc] = 6
    missing = sorted(WANT - set(picked))
    if missing: raise RuntimeError(f"missing fields {missing[:6]}")
    with ThreadPoolExecutor(16) as ex:
        futs = {k: ex.submit(fetch_range, url(cyc, src[k], base), r) for k, r in picked.items()}
        raw = {k: f.result() for k, f in futs.items()}
    return {f"{v}|{lev}": decode(msg)[0] for (v, lev), msg in raw.items()}, sum(len(m) for m in raw.values())


def fetch_cycle(cyc, out_dir="/workspace/data", log=print):
    """Download the fields WM-3 needs for one cycle, trying each mirror. Returns (npz_path, info)."""
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"gfs_{cyc}_raw.npz")
    if os.path.exists(out):
        try:
            with np.load(out) as z:
                if len(z.files) == len(WANT): return out, {"cached": True}
        except Exception: pass
        os.remove(out)                 # unreadable or incomplete cache: download again
    t0, errors = time.time(), []
    for base in SOURCES:
        try:
            arrays, nbytes = _fetch_from(cyc, base)
            tmp = out + ".tmp.npz"; np.savez(tmp, **arrays); os.replace(tmp, out)
            return out, {"cached": False, "source": base, "mb": round(nbytes / 1e6), "seconds": round(time.time() - t0, 1),
                         "mirror_errors": errors}
        except Exception as e:
            errors.append(f"{base.split('/')[2]}: {e}")
            log(f"GFS {cyc}: {base.split('/')[2]} failed ({e}); trying next mirror")
    raise RuntimeError(f"GFS {cyc} failed on every mirror: {errors}")
