"""Turn raw GFS fields (from gfs_fetch.py) into the two WM-3 encoder inputs.

Model input layout per encoder: (720, 1440, 157) float32, normalized, channels =
  5 pressure vars [z, t, u, v, q] x 28 levels (levels_medium, var-major)  -> 140
  4 core surface [10u, 10v, 2t, msl] + 4 extra [tcc, 2d, 100u, 100v]     -> 8
  9 zero-pad channels                                                     -> 9

Level mismatch: WM-3 works on 28 levels internally, but its GFS encoder was trained on 25 GFS levels
and its HRES encoder on 20 HRES levels; the (closed-source) loader filled the rest. We fill missing
levels by linear interpolation in log(pressure) on physical values, then normalize with the
28-level statistics. For the HRES encoder we first cut GFS down to the 20 HRES levels
("emulate" mode) so that encoder sees the same kind of input it was trained on.
"""
import json, os
import numpy as np

LEVELS_GFS = [10, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 925, 950, 975, 1000]
LEVELS_HRES = [10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 850, 900, 925, 950, 1000]
LEVELS_MEDIUM = [10, 30, 50, 70, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 875, 900, 925, 950, 975, 1000]
LEVELS_FULL = [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000]
PR_VARS = ["129_z", "130_t", "131_u", "132_v", "133_q"]
GFS_PR = {"129_z": "HGT", "130_t": "TMP", "131_u": "UGRD", "132_v": "VGRD", "133_q": "SPFH"}
CORE_SFC = [("165_10u", "UGRD|10 m above ground"), ("166_10v", "VGRD|10 m above ground"),
            ("167_2t", "TMP|2 m above ground"), ("151_msl", "PRMSL|mean sea level")]
EXTRA_SFC = [("45_tcc", "TCDC|entire atmosphere"), ("168_2d", "DPT|2 m above ground"),
             ("246_100u", "UGRD|100 m above ground"), ("247_100v", "VGRD|100 m above ground")]
N_PAD = 9
G = 9.80665
NORM = json.load(open(os.path.join(os.environ.get("WM3_REPO", "/workspace/WeatherMesh-3"), "constants", "normalization.json")))


def pr_stats(levels):
    idx = [LEVELS_FULL.index(l) for l in levels]
    mean = np.array([np.array(NORM[v]["mean"])[idx] for v in PR_VARS], np.float32)  # (5, L)
    std = np.array([np.array(NORM[v]["std"])[idx] for v in PR_VARS], np.float32)
    return mean, std


def interp_levels(pr, src, dst):
    """pr: (5, len(src), H, W) physical values -> (5, len(dst), H, W); linear in log(p)."""
    src = np.asarray(src, float); lp = np.log(src)
    out = np.empty((pr.shape[0], len(dst)) + pr.shape[2:], np.float32)
    for j, p in enumerate(dst):
        if p in src:
            out[:, j] = pr[:, list(src).index(p)]
            continue
        k = np.searchsorted(src, p)            # src[k-1] < p < src[k]
        w = (np.log(p) - lp[k - 1]) / (lp[k] - lp[k - 1])
        out[:, j] = (1 - w) * pr[:, k - 1] + w * pr[:, k]
    return out


def gfs_pressure(raw, levels):
    """(5, L, 721, 1440) physical from gfs_fetch npz; z converted from geopotential height (m) to m2/s2."""
    out = np.empty((5, len(levels), 721, 1440), np.float32)
    for i, v in enumerate(PR_VARS):
        for j, l in enumerate(levels):
            out[i, j] = raw[f"{GFS_PR[v]}|{l} mb"]
    out[0] *= G
    return out


def gfs_surface(raw):
    sfc = {name: raw[key].astype(np.float32) for name, key in CORE_SFC + EXTRA_SFC}
    sfc["45_tcc"] = sfc["45_tcc"] / 100.0      # % -> fraction
    return sfc


def build_input(pr_src_phys, src_levels, sfc_phys):
    """Physical (5, Ls, 721, 1440) + dict of surface fields -> normalized (720, 1440, 157) float32."""
    pr28 = interp_levels(pr_src_phys, src_levels, LEVELS_MEDIUM)
    m, s = pr_stats(LEVELS_MEDIUM)
    pr28 = (pr28 - m[:, :, None, None]) / s[:, :, None, None]
    chans = [pr28.reshape(140, 721, 1440)]
    for name, _ in CORE_SFC + EXTRA_SFC:
        chans.append(((sfc_phys[name] - NORM[name]["mean"]) / NORM[name]["std"])[None])
    chans.append(np.zeros((N_PAD, 721, 1440), np.float32))
    x = np.concatenate(chans, 0)[:, :720]          # drop the -90 row
    return np.ascontiguousarray(x.transpose(1, 2, 0)).astype(np.float32)


def inputs_from_gfs(raw_npz_path, hres_mode="emulate"):
    raw = dict(np.load(raw_npz_path))
    sfc = gfs_surface(raw)
    x_gfs = build_input(gfs_pressure(raw, LEVELS_GFS), LEVELS_GFS, sfc)
    hres_levels = LEVELS_HRES if hres_mode == "emulate" else sorted(set(LEVELS_GFS) | {20})
    x_hres = build_input(gfs_pressure(raw, hres_levels), hres_levels, sfc)
    return x_gfs, x_hres


def gfs_25_normalized(raw_npz_path):
    """Same layout as WindBorne's sample neogfs f000 'pr' array: (721, 1440, 5, 25) normalized."""
    raw = dict(np.load(raw_npz_path))
    pr = gfs_pressure(raw, LEVELS_GFS)
    m, s = pr_stats(LEVELS_GFS)
    pr = (pr - m[:, :, None, None]) / s[:, :, None, None]
    sfc = gfs_surface(raw)
    sf = np.stack([(sfc[n] - NORM[n]["mean"]) / NORM[n]["std"] for n, _ in CORE_SFC], -1)
    ex = {n: (sfc[n] - NORM[n]["mean"]) / NORM[n]["std"] for n, _ in EXTRA_SFC}
    return pr.transpose(2, 3, 0, 1), sf, ex


def inputs_from_sample(root):
    """Load WindBorne's preprocessed sample (normalized, 721 rows) into the same 157-channel layout."""
    out = []
    for src, levels in [("neogfs", LEVELS_GFS), ("neohres", LEVELS_HRES)]:
        base = os.path.join(root, src)
        f = [os.path.join(dp, x) for dp, _, fs in os.walk(os.path.join(base, "f000")) for x in fs if x.endswith(".npz")][0]
        d = np.load(f)
        m, s = pr_stats(levels)
        pr = d["pr"].astype(np.float32).transpose(2, 3, 0, 1)          # (5, L, 721, 1440) normalized
        pr = pr * s[:, :, None, None] + m[:, :, None, None]             # back to physical for interpolation
        core = d["sfc"].astype(np.float32)
        sfc = {n: core[..., i] * NORM[n]["std"] + NORM[n]["mean"] for i, (n, _) in enumerate(CORE_SFC)}
        for n, _ in EXTRA_SFC:
            g = [os.path.join(dp, x) for dp, _, fs in os.walk(os.path.join(base, "extra", n)) for x in fs if x.endswith(".npz")][0]
            sfc[n] = np.load(g)["x"].astype(np.float32) * NORM[n]["std"] + NORM[n]["mean"]
        out.append(build_input(pr, levels, sfc))
        ts = int(os.path.basename(f).split(".")[0])
    return out[0], out[1], ts
