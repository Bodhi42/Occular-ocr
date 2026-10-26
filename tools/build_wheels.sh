#!/usr/bin/env bash
# Оба колеса релиза в dist/: универсальное py3-none-any и py3-none-win_amd64 (с C++-рантаймом Microsoft).
#   tools/build_wheels.sh [OUT_DIR]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-dist}"
mkdir -p "$OUT"
[ "$(ls occular/_vcrt/*.dll 2>/dev/null | wc -l)" = 5 ] || python3 tools/fetch_vcrt.py
rm -rf build
uv build --wheel -q -o "$OUT" .
rm -rf build
OCCULAR_WHEEL_WIN_AMD64=1 uv build --wheel -q -o "$OUT" .
rm -rf build
ls -la "$OUT"/*.whl
