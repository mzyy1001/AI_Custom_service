#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

python -m feature_engine.train \
  --tree feature_engine/nodes_trained.json \
  --segments feature_engine/train_input2.in \
  --out feature_engine/nodes_trained_v2.json
