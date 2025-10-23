#!/usr/bin/env bash
set -euo pipefail

if command -v rover >/dev/null 2>&1; then
  echo "rover already available"
  exit 0
fi

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT
cd "$tmpdir"
git clone https://github.com/usnistgov/SCTK.git
cd SCTK
make config
make all
cp bin/* /usr/local/bin/
