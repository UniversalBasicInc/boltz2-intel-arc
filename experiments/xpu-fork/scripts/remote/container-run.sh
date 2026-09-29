#!/bin/bash
# Run a command inside the pinned Intel GPU runtime container with the B70 and /work mounted.
exec podman run --rm --device /dev/dri --network host --ipc host \
  -v <jarvis-work>:/work -w /work -e HOME=<container-work>/home \
  -e PATH=<container-work>/home/.local/bin:<container-work>/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  localhost/babel-arc-b70:latest bash -lc "$*"
