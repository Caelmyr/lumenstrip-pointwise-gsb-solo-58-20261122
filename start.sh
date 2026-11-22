#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export HOST="${HOST:-127.0.0.1}" PORT="${PORT:-19964}"
exec "${LUMENSTRIP_PYTHON:-python3}" -B -m lumenstrip
