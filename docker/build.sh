#!/bin/bash
# Build Strata for gfx906 in the dev image.  ./docker/build.sh [ninja targets...]   (default: strata)
set -e
cd "$(dirname "$0")/.."
T=${*:-strata}
docker run --rm -u $(id -u):$(id -g) -v $PWD:/src -w /src -e CCACHE_DIR=/src/.ccache jarvis/strata-dev:r714 bash -c "
  [ -f build-hip/build.ninja ] || cmake -S . -B build-hip -G Ninja -DSTRATA_HIP_GFX906=ON -DSTRATA_PORTABLE=ON \
    -DSTRATA_GGML_DIR=/src/third_party/llama.cpp -DCMAKE_HIP_ARCHITECTURES=gfx906 \
    -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ >/dev/null
  ninja -C build-hip -k 0 $T"
