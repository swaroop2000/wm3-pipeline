"""Load WM-3 once and run forecasts. Model weights stay fp32; math runs under fp16 autocast
(forcing the whole model to .half() breaks instance-norm layers)."""
import os, sys, time, types
import numpy as np
import torch

REPO = os.environ.get("WM3_REPO", "/workspace/WeatherMesh-3")


def _shim_utils_lite():
    import utils
    m = types.ModuleType("utils_lite")
    for k in dir(utils):
        if not k.startswith("_"): setattr(m, k, getattr(utils, k))
    for c in ["WeatherTrainerConfig", "LRScheduleConfig", "DataConfig", "OptimConfig"]:
        setattr(m, c, type(c, (), {"__init__": lambda self, *x, **y: None}))
    sys.modules["utils_lite"] = m


class WM3:
    def __init__(self, weights, device="cuda"):
        weights = os.path.abspath(weights)
        cwd = os.getcwd(); os.chdir(REPO); sys.path.insert(0, REPO)
        try:
            import model as wm
            wm.load_weights = lambda m, p: None      # we load ourselves so mismatches are visible
            net = wm.get_WeatherMesh3(weights)
            _shim_utils_lite()
            sd = torch.load(weights, map_location="cpu", weights_only=False)["model_state_dict"]
            res = net.load_state_dict(sd, strict=False)
        finally:
            os.chdir(cwd)
        # the only keys allowed to be missing are training-loss buffers rebuilt at init
        ok_missing = {"decoders.0.normalization_matrix_std", "decoders.0.geospatial_loss_weight", "decoders.0.variable_loss_weight"}
        bad = set(res.missing_keys) - ok_missing
        if bad or res.unexpected_keys:
            raise RuntimeError(f"weights mismatch: missing {sorted(bad)[:10]} unexpected {res.unexpected_keys[:10]}")
        self.net = net.to(device).eval()
        self.device = device
        mesh = net.decoders[0].mesh
        self.out_vars = list(mesh.full_varlist)
        self.out_mean = torch.tensor(mesh.normalization_matrix_mean, device=device)
        self.out_std = torch.tensor(mesh.normalization_matrix_std, device=device)
        self.lats = np.arange(90, -90.01, -0.25)[:-1]
        self.lons = np.arange(0, 359.99, 0.25)

    def idx(self, name): return self.out_vars.index(name)

    @torch.no_grad()
    def forecast(self, x_gfs, x_hres, t0_unix, lead_hours, on_output, channels=None):
        """on_output(lead_h, y, n_bad) gets PHYSICAL (720,1440,C) float32 numpy for each lead (C = channels or all 157)
        and n_bad = count of NaN/inf values across ALL 157 output channels."""
        xg = torch.from_numpy(x_gfs)[None].to(self.device)
        xh = torch.from_numpy(x_hres)[None].to(self.device)
        t0 = torch.tensor([int(t0_unix)], device=self.device)
        timings = {}

        def cb(dt, out):
            y = out[0][0].float() * self.out_std + self.out_mean
            n_bad = int((~torch.isfinite(y)).sum())
            if channels is not None: y = y[..., channels]
            torch.cuda.synchronize(); timings[dt] = time.time() - start
            on_output(dt, y.cpu().numpy(), n_bad)

        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        with torch.autocast("cuda", dtype=torch.float16):
            self.net([xg, xh, t0], sorted(lead_hours), callback=cb)
        torch.cuda.synchronize()
        return {"total_s": time.time() - start, "per_lead_s": timings,
                "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9}
