set -e
cd "$(dirname "$0")"

python -m feature_engine.produce \
  --tree feature_engine/nodes_trained.json \
