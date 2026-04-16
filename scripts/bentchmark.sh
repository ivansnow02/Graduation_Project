#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[deprecated] scripts/bentchmark.sh has been archived; forwarding to scripts/archive/legacy_benchmark/bentchmark.sh" >&2
exec bash "${SCRIPT_DIR}/archive/legacy_benchmark/bentchmark.sh" "$@"
