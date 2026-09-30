#!/bin/bash
# Stage what the runtime image needs into a tar on stdout: engine binary, serve/, tools/, data/, gguf-py.
#   ./docker/stage.sh | ssh jarvis 'docker build -t jarvis/strata:gfx906 -f strata/docker/strata-gfx906.Dockerfile -'
set -e
cd "$(dirname "$0")/.."
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
mkdir -p "$T/strata/engine" "$T/strata/third_party/llama.cpp" "$T/strata/docker"
cp build-hip/strata "$T/strata/engine/"
for f in build-hip/*_parity build-hip/strata-device build-hip/strata-gguf; do cp "$f" "$T/strata/engine/"; done
cp -r serve tools data chat.py LICENSE "$T/strata/"
cp -r third_party/llama.cpp/gguf-py "$T/strata/third_party/llama.cpp/"
mkdir -p "$T/strata/llama" && cp -a third_party/llama.cpp/build-hip/bin/. "$T/strata/llama/"
cp docker/*.Dockerfile "$T/strata/docker/"
git rev-parse --short HEAD > "$T/strata/VERSION.git"
tar -C "$T" -cf - strata
