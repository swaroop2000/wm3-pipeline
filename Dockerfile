# WeatherMesh-3 real-time pipeline. Needs an Ada/Ampere-class NVIDIA GPU (tested: RTX 4090) - see README.
#   docker build -t wm3-pipeline .
#   docker run -d --gpus all --restart unless-stopped --env-file wm3.env \
#       -v /path/to/workspace:/workspace wm3-pipeline
# /workspace must contain weights/WeatherMesh3.pt; data, runs and logs are written there too.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
# NATTEN 0.17.3 wheel built for torch 2.4.0 + cu121. shi-labs.com's TLS cert had expired (Oct 2026), hence --trusted-host.
RUN pip install --no-cache-dir "natten==0.17.3+torch240cu121" -f https://shi-labs.com/natten/wheels/ --trusted-host shi-labs.com \
 && pip install --no-cache-dir --no-deps --index-url https://test.pypi.org/simple/ matepoint \
 && pip install --no-cache-dir "numpy<2" xarray cfgrib eccodes eccodeslib netCDF4 boto3 requests matplotlib

RUN git clone --depth 1 https://github.com/windborne/WeatherMesh-3.git /opt/WeatherMesh-3
ENV WM3_REPO=/opt/WeatherMesh-3 WM3_WEIGHTS=/workspace/weights/WeatherMesh3.pt PYTHONUNBUFFERED=1
COPY . /app
WORKDIR /app
# Docker's --restart policy plays the role of run_forever.sh
CMD ["python", "-m", "wm3.service"]
