"""Write forecast outputs: NetCDF per lead (with metadata), WindBorne-API-style .npy grids, PNG maps."""
import json, os
os.environ["MPLBACKEND"] = "Agg"   # Jupyter sets an inline backend that does not exist in our venv
from datetime import datetime, timedelta, timezone
import numpy as np

G = 9.80665
LAT = np.arange(90, -90.01, -0.25)[:-1].astype(np.float32)
LON = np.arange(0, 359.99, 0.25).astype(np.float32)
LEVELS_SAVED = [250, 500, 850]
# short name -> (model channel, long name, units)
FIELDS = {"10u": ("165_10u", "10 metre U wind component", "m s-1"), "10v": ("166_10v", "10 metre V wind component", "m s-1"),
          "2t": ("167_2t", "2 metre temperature", "K"), "msl": ("151_msl", "Mean sea level pressure", "Pa"),
          "tcc": ("45_tcc", "Total cloud cover", "(0 - 1)"), "2d": ("168_2d", "2 metre dewpoint temperature", "K"),
          "100u": ("246_100u", "100 metre U wind component", "m s-1"), "100v": ("247_100v", "100 metre V wind component", "m s-1")}
for l in LEVELS_SAVED:
    FIELDS[f"z{l}"] = (f"129_z_{l}", f"Geopotential at {l} hPa", "m2 s-2")
    FIELDS[f"t{l}"] = (f"130_t_{l}", f"Temperature at {l} hPa", "K")
    FIELDS[f"u{l}"] = (f"131_u_{l}", f"U wind at {l} hPa", "m s-1")
    FIELDS[f"v{l}"] = (f"132_v_{l}", f"V wind at {l} hPa", "m s-1")
    FIELDS[f"q{l}"] = (f"133_q_{l}", f"Specific humidity at {l} hPa", "kg kg-1")
SHORT = list(FIELDS)
# WindBorne API gridded endpoints -> (short name, converter, units)
API = {"temperature_2m": ("2t", lambda x: x - 273.15, "degC"), "dewpoint_2m": ("2d", lambda x: x - 273.15, "degC"),
       "wind_u_10m": ("10u", lambda x: x, "m/s"), "wind_v_10m": ("10v", lambda x: x, "m/s"),
       "pressure_msl": ("msl", lambda x: x / 100, "hPa"), "geopotential_500": ("z500", lambda x: x, "m2/s2"),
       "temperature_850": ("t850", lambda x: x - 273.15, "degC")}
API_MAX_LEAD = 120


def cycle_time(cyc): return datetime.strptime(cyc, "%Y%m%d%H").replace(tzinfo=timezone.utc)


def write_netcdf(path, cyc, lead, f):
    import xarray as xr
    init = cycle_time(cyc); valid = init + timedelta(hours=lead)
    ds = xr.Dataset(coords={"latitude": ("latitude", LAT, {"units": "degrees_north"}),
                            "longitude": ("longitude", LON, {"units": "degrees_east"})})
    enc = {}
    for k in SHORT:
        ch, long, units = FIELDS[k]
        x = f[k].astype(np.float32)
        ds[k] = (("latitude", "longitude"), x, {"long_name": long, "units": units, "wm3_channel": ch})
        lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
        scale = (hi - lo) / 65000 or 1.0
        enc[k] = {"dtype": "int16", "scale_factor": scale, "add_offset": (hi + lo) / 2, "_FillValue": -32768,
                  "zlib": True, "complevel": 3}
    ds.attrs = {"title": "WeatherMesh-3 global forecast", "model": "WeatherMesh-3 (WindBorne open weights)",
                "initialization_time": init.isoformat(), "valid_time": valid.isoformat(), "forecast_hour": lead,
                "initial_condition": f"NOAA GFS 0.25 deg analysis {cyc} used for both GFS and HRES encoders",
                "grid": "0.25 deg regular lat/lon, 90N..89.75S, 0..359.75E", "Conventions": "CF-1.8",
                "created": datetime.now(timezone.utc).isoformat()}
    tmp = path + ".tmp"; ds.to_netcdf(tmp, encoding=enc); os.replace(tmp, path)


def write_api(run_dir, lead, f):
    if lead > API_MAX_LEAD: return []
    out = []
    for name, (k, conv, _) in API.items():
        d = os.path.join(run_dir, "api", name); os.makedirs(d, exist_ok=True)
        p = os.path.join(d, f"{lead:03d}.npy"); np.save(p, conv(f[k]).astype(np.float32)); out.append(p)
    return out


_LAND = None
def _land():
    global _LAND
    if _LAND is None:
        try:
            m = np.load(os.path.join(os.environ.get("WM3_REPO", "/workspace/WeatherMesh-3"), "constants/additional_variables/land_mask.npy")).astype(np.float32)
            iy = np.linspace(0, m.shape[0] - 1, 720).astype(int); ix = np.linspace(0, m.shape[1] - 1, 1440).astype(int)
            _LAND = m[iy][:, ix]
        except Exception:
            _LAND = False
    return _LAND


def write_maps(map_dir, cyc, lead, f):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(map_dir, exist_ok=True)
    roll = 720                                  # put 0E in the middle -> -180..180
    lon = np.roll(np.where(LON >= 180, LON - 360, LON), roll)
    R = lambda a: np.roll(a, roll, axis=1)
    land = _land()
    valid = cycle_time(cyc) + timedelta(hours=lead)
    title = f"WM-3  init {cyc[:8]} {cyc[8:]}Z  +{lead}h  valid {valid:%Y-%m-%d %H}Z"
    specs = [("t2m_msl", R(f["2t"]) - 273.15, "RdYlBu_r", (-40, 45), "2 m temperature (°C), MSL pressure (hPa)", R(f["msl"]) / 100, np.arange(940, 1060, 4)),
             ("wind250_z500", R(np.hypot(f["u250"], f["v250"])), "viridis", (0, 90), "250 hPa wind speed (m/s), 500 hPa height (dam)", R(f["z500"]) / G / 10, np.arange(480, 600, 6)),
             ("tcc", R(f["tcc"]), "Greys_r", (0, 1), "Total cloud cover (0-1)", None, None)]
    names = []
    for name, field, cmap, (vmin, vmax), label, cont, lv in specs:
        fig, ax = plt.subplots(figsize=(10, 5.2), dpi=80)
        im = ax.pcolormesh(lon, LAT, field, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto", rasterized=True)
        if land is not False:
            ax.contour(lon, LAT, R(land), levels=[0.5], colors="k", linewidths=0.4)
        if cont is not None:
            cs = ax.contour(lon, LAT, cont, levels=lv, colors="w" if name != "t2m_msl" else "k", linewidths=0.5)
            ax.clabel(cs, fontsize=6, fmt="%d")
        ax.set_title(title, fontsize=10); ax.set_xlim(-180, 180); ax.set_ylim(-90, 90); ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, shrink=0.75, label=label)
        fig.tight_layout(); fn = f"{name}_f{lead:03d}.png"; fig.savefig(os.path.join(map_dir, fn)); plt.close(fig)
        names.append(fn)
    return names


def write_manifest(run_dir, manifest):
    with open(os.path.join(run_dir, "manifest.json"), "w") as fh: json.dump(manifest, fh, indent=1)
