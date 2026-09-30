# Strata runtime for gfx906 on top of an earlier runtime image: the same as strata-gfx906.Dockerfile, but without
# apt - for when the server cannot reach the Ubuntu archive.  The Python packages come from the base image; only
# /opt/strata (engine, serve/, tools/, data/) is replaced.
#   ./docker/stage.sh | ssh jarvis 'docker build -t jarvis/strata:gfx906 -f strata/docker/strata-gfx906-poverh.Dockerfile -'
ARG BASE=jarvis/strata:gfx906-pre30
FROM ${BASE}
RUN rm -rf /opt/strata
COPY strata /opt/strata
WORKDIR /opt/strata
