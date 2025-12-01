#!/usr/bin/env bash
set -euo pipefail

# Run scene mapping with bundled overrides (val_mini + video).
HYDRA_FULL_ERROR=1 python -m habitat_llm.examples.scene_mapping
