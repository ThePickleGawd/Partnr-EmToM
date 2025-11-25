#!/usr/bin/env bash
set -euo pipefail

INIT_JSON="data/emtom/initial_state_dicts.json"
GEN_CONFIG="data/emtom/gen_config.json"
DATASET_OUT="data/emtom/test_house.json.gz"

echo "Generating dataset at ${DATASET_OUT}..."
python -m dataset_generation.benchmark_generation.generate_episodes \
  --init-state-dicts "${INIT_JSON}" \
  --gen-config "${GEN_CONFIG}"
