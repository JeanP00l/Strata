# Strata dev image for AMD gfx906 (MI50): ROCm 7.14 (mixa3607/rocm-gfx906) + the build tools.
# The source is mounted, not copied: docker run -v $PWD:/src jarvis/strata-dev:r714 ...
FROM mixa3607/rocm-gfx906:7.14-complete
RUN rm -f /etc/apt/sources.list.d/*gfx906* /etc/apt/sources.list.d/*ark* 2>/dev/null; grep -rl arkprojects /etc/apt/ | xargs -r rm -f; apt-get update && apt-get install -y --no-install-recommends cmake ninja-build git python3 python3-pip \
      python3-numpy build-essential ccache && rm -rf /var/lib/apt/lists/*
ENV CMAKE_PREFIX_PATH=/opt/rocm PATH=/opt/rocm/bin:$PATH
WORKDIR /src
