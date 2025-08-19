#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

python -m feature_engine.train \
  --tree feature_engine/nodes.json \
  --segments feature_engine/train_input.in \
  --out feature_engine/nodes_trained.json
