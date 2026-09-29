# Strata runtime for AMD gfx906 (MI50), ROCm 7.14.  The engine is built beforehand by docker/build.sh (in the dev
# image, same ROCm) and staged by docker/stage.sh; this image only adds Python for the server and the tools.
FROM mixa3607/rocm-gfx906:7.14-complete
RUN grep -rl arkprojects /etc/apt/ | xargs -r rm -f; apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-numpy python3-jinja2 python3-regex python3-yaml python3-pil python3-psutil python3-requests \
      python3-tqdm && rm -rf /var/lib/apt/lists/*
COPY strata /opt/strata
ENV STRATA_GGUF_PY=/opt/strata/third_party/llama.cpp/gguf-py PYTHONUNBUFFERED=1
WORKDIR /opt/strata
ENV LD_LIBRARY_PATH=/opt/strata/llama
