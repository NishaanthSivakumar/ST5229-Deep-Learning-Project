#!/usr/bin/env bash
# Fast end-to-end smoke test: verifies the pipeline runs without errors.
set -e
cd "$(dirname "$0")/.."
python src/run_all.py --smoke
