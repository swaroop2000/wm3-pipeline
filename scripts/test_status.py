"""Stage 1d: prove S3 + the web page work before any model code.
Uploads the page, writes a few fake events and a fake run, then keeps a heartbeat for N minutes.
Run:  python scripts/test_status.py --minutes 3      (then open the S3 website URL)"""
import argparse, os, sys, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HEARTBEAT_SEC", "30")
from wm3.status import Status, s3, BUCKET, key

ap = argparse.ArgumentParser(); ap.add_argument("--minutes", type=float, default=3); a = ap.parse_args()
page = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dashboard", "index.html")
s3.upload_file(page, BUCKET, key("index.html"), ExtraArgs={"ContentType": "text/html", "CacheControl": "no-cache"})
st = Status()
st.event("info", "test_status.py started (this is a TEST, not a forecast)")
st.set(state="idle", step="waiting for next GFS cycle")
st.record_run({"cycle": "TEST0000", "status": "ok", "timings": {"download_s": 41.0, "preprocess_s": 12.3,
               "inference_s": 9.8, "write_s": 20.1, "upload_s": 15.0}, "checks": {"passed": 9, "total": 9, "failed": []},
               "outputs_prefix": "runs/TEST0000/"})
t_end = time.time() + 60 * a.minutes
while time.time() < t_end:
    time.sleep(30); st.event("info", f"tick, fake gpu util {random.randint(0, 5)}%")
st.event("warn", "test finished; heartbeat will stop now, so the page should go amber then red")
print("DONE. Website:", f"http://{BUCKET}.s3-website-{s3.meta.region_name}.amazonaws.com/")
