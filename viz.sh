#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

python feature_engine/visualize_tree.py \
  feature_engine/nodes_trained.json \
  -o tree.html
