#!/usr/bin/env bash
set -euo pipefail

mkdir -p /app/data /app/models /app/outputs

exec "$@"
