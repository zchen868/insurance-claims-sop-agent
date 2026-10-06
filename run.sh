#!/usr/bin/env bash
# Local dev launcher: creates a venv, installs deps, starts the server on :8000.
set -e
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q -r requirements-dev.txt
[ -f .env ] && set -a && . ./.env && set +a
exec .venv/bin/uvicorn sop_agent.server:app --host 0.0.0.0 --port "${PORT:-8000}"
