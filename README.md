# WeatherMesh-3 real-time inference pipeline

Runs WindBorne's open WeatherMesh-3 (WM-3) weights every 6 hours from live NOAA GFS data on one GPU,
checks every forecast automatically, and publishes results plus a live status page to S3.

```
NOAA GFS (AWS open data) --byte-range .idx--> 138 fields, ~120 MB, ~8 s
   -> prep: units, 25/20 -> 28 levels (log-p interpolation), normalize -> 2 x (720,1440,157)
   -> WM-3 on RTX 4090 (fp32 weights, fp16 autocast): 0-240 h in ~20 s, 8.3 GB VRAM
   -> ~360 automatic checks (finite, physical ranges, dewpoint<=temp, global drift, skill vs persistence)
   -> NetCDF per lead (CF metadata) + WindBorne-API-style .npy grids + PNG maps + manifest.json
   -> S3 runs/<cycle>/ ; runs.json + status.json (heartbeat) ; index.html status page
```

## Layout
| path | what |
|---|---|
| `wm3/gfs.py` | find newest complete GFS cycle, download only needed GRIB messages via `.idx` byte ranges |
| `wm3/prep.py` | GFS -> model inputs. Unit fixes (gpm x g, cloud % -> 0-1), level interpolation, normalization |
| `wm3/runner.py` | **inference**: loads WM-3, verifies weights, runs forecasts, de-normalizes outputs |
| `wm3/checks.py` | sanity checks run on every cycle |
| `wm3/outputs.py` | NetCDF / API grids / maps / manifest writers |
| `wm3/status.py` | heartbeat + status/runs JSON on S3, ntfy alerts |
| `wm3/service.py` | the forever loop (retries, catch-up to newest cycle, cleanup) |
| `run_forever.sh` | restart wrapper + single-instance lock (for hosts without Docker restart policies) |
| `dashboard/index.html` | static status page served from the bucket |
| `scripts/` | setup checks: `inspect_inputs.py`, `smoke_test.py`, `check_stage2.py`, `gfs_fetch.py`, `test_status.py` |

## Running
**RunPod (what was used):** RTX 4090 on-demand, image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, 80 GB disk.
```
bash setup.sh                                    # venv with torch 2.4.0+cu121, NATTEN 0.17.3, deps
cp env.example /workspace/wm3.env  # fill in AWS keys, S3_BUCKET, NTFY_TOPIC
python -m wm3.service --once                     # one full cycle as a test
nohup bash run_forever.sh > /dev/null 2>&1 &     # run continuously
```
**Docker:** see the header of `Dockerfile`.

## Reliability
| failure | what happens |
|---|---|
| NOAA AWS bucket down / slow / file missing | next mirror: Google Cloud copy, then NOAA NOMADS |
| GFS late | cycle counts as ready only when f000 **and** f006 exist; keeps polling every 5 min; alert if newest good forecast > 13 h old (error > 18 h) |
| download error / corrupt or partial cache | per-request retries; cache validated (all 138 fields) or re-downloaded; failed cycle's cache deleted before retry |
| bad input data | ~8 input checks before the model runs; cycle fails + alert, model never sees it |
| bad forecast | ~400 output checks; cycle marked failed, alert, `latest.json` stays on last good cycle |
| one cycle keeps failing | retried every 5 min until it works or a newer cycle replaces it; alert on 1st, 3rd, every 6th try |
| GPU/CUDA gets into a bad state | 3 failures in a row -> process exits -> restarted fresh |
| process hangs | watchdog thread: a step with no progress for 20 min -> alert + process exit -> restart |
| process crashes | `run_forever.sh` (or Docker `--restart`) restarts it in 30 s |
| whole pod dies | status page turns red from the stale heartbeat; optional healthchecks.io ping emails you |
| disk filling | old inputs deleted after 4 days; emergency cleanup below 5 GB free |

`scripts/chaos_test.py` injects these faults (dead mirror, corrupt cache, unit bugs, NaNs, blow-ups) and checks each is handled.
Backfill: set `BACKFILL_HOURS=72` and missing past cycles are run oldest-first on start, so skill scores exist immediately.

## Outputs on S3
```
index.html, status.json, runs.json
runs/<YYYYMMDDHH>/manifest.json        every file, timings, check summary, skill scores
runs/<YYYYMMDDHH>/checks.json          every individual check with its value
runs/<YYYYMMDDHH>/forecast/wm3_<cycle>_f<HHH>.nc   23 fields (sfc + z/t/u/v/q at 250/500/850), 25 leads 6..240 h
runs/<YYYYMMDDHH>/api/<variable>/<HHH>.npy         temperature_2m, dewpoint_2m (degC), wind_u_10m, wind_v_10m (m/s),
                                                   pressure_msl (hPa), geopotential_500 (m2/s2), temperature_850 (degC); (lat,lon) 0.25 deg, 6..120 h
runs/<YYYYMMDDHH>/maps/*.png
```

## Validation done before going live
* Our GFS preprocessing reproduces WindBorne's sample input for 2025-03-07 00Z: correlation 1.00000, RMSE 0.0002-0.0003 (fp16 rounding) for all 129 core channels.
* 10-day forecasts stay physical and stable (global mean 2 m temperature within 0.4 K).
* Against later GFS analyses, 500 hPa height error is 6.6 m at 24 h (persistence 66 m) and 41 m at 120 h (persistence 113 m).

## Known issues / decisions
* Repo pins clash: torch 2.5.1 vs a NATTEN wheel built for torch 2.4.0 -> we use torch 2.4.0+cu121. Blackwell GPUs are not supported by this NATTEN.
* NATTEN wheel host's TLS certificate expired -> `--trusted-host shi-labs.com`.
* Calling `.half()` on the model breaks instance norm -> fp32 weights + fp16 autocast.
* The four "extra" input fields (cloud cover, 2 m dewpoint, 100 m wind) differ from WindBorne's sample (cloud cover corr 0.61): they were built from a different GFS product than the f000 analysis.
* HRES encoder gets GFS cut to the 20 HRES levels then interpolated to 28 ("emulate"), matching what it saw in training.
* Precipitation and min/max temperature outputs are log-transformed in the model; not exported yet.
