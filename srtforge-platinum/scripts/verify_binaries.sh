#!/usr/bin/env bash
set -euo pipefail

for binary in ffmpeg rover sox; do
  if ! command -v "$binary" >/dev/null 2>&1; then
    echo "Missing required binary: $binary" >&2
    exit 1
  fi
  echo "Found $binary: $(command -v "$binary")"
done
