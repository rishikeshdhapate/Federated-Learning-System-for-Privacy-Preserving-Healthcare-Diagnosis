#!/usr/bin/env bash
# build.sh — Render build script
# Runs once during the build phase (before the web process starts).
# 1. Install Python dependencies
# 2. Pre-generate the synthetic hospital dataset so the API starts cleanly.

set -e

echo "=== Installing dependencies ==="
pip install -r requirements.txt

echo "=== Generating hospital dataset ==="
python -c "
import sys
sys.path.insert(0, '.')
from data.preprocess import prepare_and_save
prepare_and_save(
    processed_dir='data/processed',
    n_hospitals=5,
    n_samples=50000,
    seed=42
)
"

echo "=== Build complete ==="
