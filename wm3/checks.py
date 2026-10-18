"""Automatic sanity checks run on every forecast cycle.

Two severities:
  HARD  - anything named "... finite": NaN/inf in inputs or outputs. The forecast is unusable -> cycle FAILED.
  SOFT  - everything else (ranges, overshoot, drift, skill). Real weather can be extreme, and a check limit can
          be wrong, so a soft failure never stops a forecast: it is published with status FLAGGED and reported.

Three kinds:
  1. physical  - no NaN/inf, every field inside the range real weather can take
  2. stability - global averages must not drift (a model going unstable shows up here first)
  3. skill     - once older runs exist: their 6/24/72 h forecasts are scored against today's
                 GFS analysis and must beat persistence ("tomorrow = today")
"""
import numpy as np

G = 9.80665
LAT = np.arange(90, -90.01, -0.25)[:-1]
W = np.cos(np.deg2rad(LAT))[:, None] * np.ones((1, 1440)); W /= W.mean()


def gmean(a): return float(np.mean(W * a))
def rmse(a, b): return float(np.sqrt(np.mean(W * (a - b) ** 2)))


# (field, min, max, units shown) - generous physical limits, not climatology
RANGES = [("2t", 170, 345, "K"), ("2d", 150, 320, "K"), ("msl", 85000, 109000, "Pa"),
          ("z500", 4300 * G, 6300 * G, "m2/s2"), ("t850", 200, 325, "K"), ("t250", 180, 260, "K")]

# Bounded variables (cloud cover 0-1, humidity >= 0). WM-3 is an unbounded regression model, so it overshoots
# these limits a little even from WindBorne's OWN sample inputs (measured: cloud cover -1.04..2.48, up to 5.5% of
# cells outside 0-1; humidity at 850 hPa down to -0.001 kg/kg, up to 0.4% of cells). Our GFS-only inputs give up
# to 9.5% and 2.7% at 6 h, fading by 72 h. Policy: such overshoot is clipped before writing (and counted), and the
# cycle only FAILS if overshoot is far beyond what the reference run showed (roughly 2x the worst seen).
BOUNDED = [("tcc", "cloud cover", 0.0, 1.0, -1.5, 3.0, 0.20),       # (key, label, lo, hi, hard_min, hard_max, max share outside)
           ("q850", "humidity 850", 0.0, None, -0.003, 0.035, 0.06),
           ("q500", "humidity 500", 0.0, None, -0.003, 0.025, 0.02),
           ("q250", "humidity 250", 0.0, None, -0.001, 0.01, 0.01)]
WIND = [("10u", "10v", 80), ("u250", "v250", 160), ("u850", "v850", 110), ("100u", "100v", 90)]


class CheckSet:
    def __init__(self): self.items = []
    def add(self, name, ok, detail=""):
        self.items.append({"name": name, "ok": bool(ok), "detail": detail})
    def summary(self):
        failed = [c["name"] + (f" ({c['detail']})" if c["detail"] else "") for c in self.items if not c["ok"]]
        return {"passed": sum(c["ok"] for c in self.items), "total": len(self.items), "failed": failed}


def check_inputs(cs, raw, x_gfs, x_hres):
    """Catch bad NOAA data BEFORE it reaches the model: physical ranges on the raw GFS fields,
    and normalized model inputs finite and below a corruption-level limit."""
    for k, lo, hi, u in [("TMP|2 m above ground", 170, 345, "K"), ("PRMSL|mean sea level", 85000, 109000, "Pa"),
                         ("HGT|500 mb", 4300, 6300, "m"), ("TMP|850 mb", 200, 325, "K"),
                         ("SPFH|850 mb", -1e-6, 0.035, "kg/kg"), ("TCDC|entire atmosphere", 0, 100.5, "%")]:
        a = raw[k]; fin = bool(np.isfinite(a).all()); mn, mx = float(np.nanmin(a)), float(np.nanmax(a))
        name = f"input {k.split('|')[0]} {k.split('|')[1]}"
        cs.add(f"{name} finite", fin, "" if fin else f"{int((~np.isfinite(a)).sum())} NaN/inf")
        cs.add(f"{name} range", lo <= mn and mx <= hi, f"{mn:.4g}..{mx:.4g} {u}")
    # Normalized-input limit. Measured on 6 real cycles: GFS's dry floor in the stratosphere (q = 1e-9 kg/kg at
    # 20-50 hPa) always gives ~12.6 sigma, and moist tropical deep convection in the W Pacific (developing
    # Typhoon Champi, Oct 2026) reaches 70 hPa with q ~1e-5 kg/kg = 15-28 sigma. Real weather, not corruption.
    for name, x in (("gfs", x_gfs), ("hres", x_hres)):
        ok = bool(np.isfinite(x).all()); mx = float(np.abs(x).max()) if ok else float("nan")
        n_out = int((np.abs(x) > 15).sum()) if ok else -1
        cs.add(f"input {name} normalized finite", ok)
        cs.add(f"input {name} normalized |x|<60", ok and mx < 60, f"max |x| {mx:.1f}, {n_out} values beyond 15 sigma")


def check_lead(cs, lead, f, n_bad):
    """f: dict short-name -> (720,1440) physical array for one lead time."""
    cs.add(f"f{lead:03d} finite", n_bad == 0, f"{n_bad} NaN/inf" if n_bad else "")
    for k, lo, hi, u in RANGES:
        mn, mx = float(np.nanmin(f[k])), float(np.nanmax(f[k]))
        cs.add(f"f{lead:03d} {k} range", lo <= mn and mx <= hi, f"{mn:.4g}..{mx:.4g} {u}")
    for uk, vk, lim in WIND:
        sp = float(np.hypot(f[uk], f[vk]).max())
        cs.add(f"f{lead:03d} wind {uk}/{vk} max", sp < lim, f"{sp:.1f} m/s")
    for k, label, lo, hi, hmin, hmax, share in BOUNDED:
        a = f[k]; mn, mx = float(a.min()), float(a.max())
        out = float(np.mean((a < lo) | ((a > hi) if hi is not None else False)))
        cs.add(f"f{lead:03d} {label} overshoot", hmin <= mn and mx <= hmax and out <= share,
               f"{mn:.3g}..{mx:.3g}, {100*out:.2f}% of cells outside limits (allowed {100*share:.0f}%)")
    frac = float(np.mean(f["2d"] <= f["2t"] + 0.5))      # dewpoint can't exceed temperature
    cs.add(f"f{lead:03d} dewpoint<=temp", frac > 0.995, f"{100*frac:.2f}% ok")


def check_stability(cs, means):
    """means: {lead: {'2t': global mean K, 'msl': Pa, 'z500': m2/s2}} - compare every lead with the first."""
    leads = sorted(means); first = means[leads[0]]
    d2t = max(abs(means[l]["2t"] - first["2t"]) for l in leads)
    dmsl = max(abs(means[l]["msl"] - first["msl"]) for l in leads) / 100
    dz = max(abs(means[l]["z500"] - first["z500"]) for l in leads) / G
    cs.add("global 2t drift < 1.5 K", d2t < 1.5, f"{d2t:.2f} K")
    cs.add("global msl drift < 2 hPa", dmsl < 2, f"{dmsl:.2f} hPa")
    cs.add("global z500 drift < 30 m", dz < 30, f"{dz:.1f} m")


def check_skill(cs, analysis_now, past):
    """past: {lead: (forecast_fields_valid_now, analysis_at_that_init)}. Model must beat persistence."""
    out = {}
    for lead, (fc, an0) in sorted(past.items()):
        row = {}
        for k, scale, u in (("z500", G, "m"), ("t850", 1, "K"), ("msl", 100, "hPa")):
            m = rmse(fc[k], analysis_now[k]) / scale
            p = rmse(an0[k], analysis_now[k]) / scale
            row[k] = {"model": round(m, 2), "persistence": round(p, 2), "units": u}
            cs.add(f"skill {lead}h {k} beats persistence", m < p, f"{m:.2f} vs {p:.2f} {u}")
        out[lead] = row
    return out
