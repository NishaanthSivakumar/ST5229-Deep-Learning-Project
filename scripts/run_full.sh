#!/usr/bin/env bash
# Full run (defaults in config.py: 10 teacher epochs, 50 pretrain epochs, 20 probe epochs).
# On a CPU this takes a while -- consider launching overnight.
set -e
cd "$(dirname "$0")/.."
python src/run_all.py
