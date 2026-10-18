"""Heartbeat + status files on S3. The GPU script is the only writer; the web page only reads.

Every HEARTBEAT_SEC a background thread rewrites status.json (time, state, current step, GPU info,
last 200 events). runs.json holds one row per forecast cycle. The page compares the heartbeat time
with the viewer's clock, so if the pod dies the page turns red by itself - no cron, no extra server.
Failures also push a phone alert through ntfy.sh if NTFY_TOPIC is set.

Watchdog (same thread as the heartbeat):
  * HANG: if one pipeline step makes no progress for STALL_SEC (default 20 min; a normal whole cycle
    is ~2-3 min), it alerts and kills the process with os._exit -> run_forever.sh / Docker restarts it.
  * DEAD MAN: if HEALTHCHECK_URL is set (e.g. a free healthchecks.io check), every heartbeat pings it.
    If the pod or the process dies, pings stop and healthchecks.io emails / pushes you. That covers
    the one failure the process can't report itself: being dead."""
import json, os, re, socket, subprocess, threading, time, traceback
from datetime import datetime, timezone
import boto3, requests

BUCKET = os.environ["S3_BUCKET"]
PREFIX = os.environ.get("S3_PREFIX", "").strip("/")
HEARTBEAT_SEC = int(os.environ.get("HEARTBEAT_SEC", "300"))
STALL_SEC = int(os.environ.get("STALL_SEC", "1200"))
ENV_FILE = "/workspace/wm3.env"

def setting(name):
    """Read alert settings fresh each time, so adding NTFY_TOPIC / HEALTHCHECK_URL to wm3.env works without a restart."""
    try:
        m = re.search(rf"^{name}=(.*)$", open(ENV_FILE).read(), re.M)
        if m and m.group(1).strip(): return m.group(1).strip()
    except Exception: pass
    return os.environ.get(name, "")
s3 = boto3.client("s3")

def now(): return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
def key(k): return f"{PREFIX}/{k}" if PREFIX else k

def put_json(k, obj):
    s3.put_object(Bucket=BUCKET, Key=key(k), Body=json.dumps(obj, indent=1).encode(),
                  ContentType="application/json", CacheControl="no-cache, max-age=0")

def get_json(k, default):
    try: return json.loads(s3.get_object(Bucket=BUCKET, Key=key(k))["Body"].read())
    except Exception: return default

def gpu_info():
    try:
        q = "name,utilization.gpu,memory.used,memory.total,temperature.gpu"
        out = subprocess.check_output(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"], text=True).strip()
        n, u, mu, mt, t = [s.strip() for s in out.split(",")]
        return {"name": n, "util_pct": int(u), "mem_used_mb": int(mu), "mem_total_mb": int(mt), "temp_c": int(t)}
    except Exception: return {}

def disk_free_gb(path="/workspace" if os.path.isdir("/workspace") else "/"):
    st = os.statvfs(path); return round(st.f_bavail * st.f_frsize / 1e9, 1)

class Status:
    def __init__(self):
        self.lock = threading.Lock()
        self.s = {"state": "starting", "step": "", "cycle": None, "started": now(), "host": socket.gethostname(),
                  "heartbeat": now(), "heartbeat_every_sec": HEARTBEAT_SEC, "last_error": None, "events": []}
        self.runs = get_json("runs.json", [])
        self.last_progress = time.time()
        self.flush()
        threading.Thread(target=self._beat, daemon=True).start()

    def _beat(self):
        while True:
            time.sleep(min(HEARTBEAT_SEC, 60))
            stuck = time.time() - self.last_progress
            if self.s["state"] == "running" and stuck > STALL_SEC:
                msg = f"WATCHDOG: step '{self.s['step']}' of cycle {self.s['cycle']} stuck {stuck/60:.0f} min - restarting process"
                try: self.event("error", msg)
                except Exception: pass
                self.ping_health("/fail")
                os._exit(3)                       # run_forever.sh / Docker restarts us
            if time.time() - self._last_flush >= HEARTBEAT_SEC - 1:
                try: self.flush()
                except Exception: traceback.print_exc()

    def ping_health(self, suffix=""):
        url = setting("HEALTHCHECK_URL")
        if url:
            try: requests.get(url.rstrip("/") + suffix, timeout=10)
            except Exception: pass

    def flush(self):
        self._last_flush = time.time()
        if self.s.get("state") != "failed": self.ping_health()
        with self.lock:
            self.s["heartbeat"] = now(); self.s["gpu"] = gpu_info(); self.s["disk_free_gb"] = disk_free_gb()
            put_json("status.json", self.s)

    def event(self, level, msg, flush=True):
        print(f"[{now()}] {level.upper()} {msg}", flush=True)
        with self.lock:
            self.s["events"] = (self.s["events"] + [{"t": now(), "level": level, "msg": msg}])[-200:]
        if level == "error":
            self.s["last_error"] = {"t": now(), "msg": msg}
            self.alert(msg)
        if flush: self.flush()

    def set(self, state=None, step=None, cycle=None):
        with self.lock:
            if state: self.s["state"] = state
            if step is not None: self.s["step"] = step
            if cycle is not None: self.s["cycle"] = cycle
        self.last_progress = time.time()
        self.flush()

    def record_run(self, row):
        """row: {cycle, status, timings{...}, checks{passed,total,failed[]}, outputs_prefix}"""
        self.runs = [r for r in self.runs if r["cycle"] != row["cycle"]] + [row]
        self.runs = sorted(self.runs, key=lambda r: r["cycle"])[-120:]
        put_json("runs.json", self.runs)

    def alert(self, msg):
        topic = setting("NTFY_TOPIC")
        if topic:
            try: requests.post(f"https://ntfy.sh/{topic}", data=f"WM3: {msg}"[:400].encode(),
                               headers={"Title": "WM-3 pipeline", "Priority": "high"}, timeout=10)
            except Exception: pass
        self.ping_health("/fail")
