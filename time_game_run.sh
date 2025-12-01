#!/usr/bin/env bash
set -euo pipefail

set -a
if [ -f .env ]; then
  source .env
fi
set +a

python -m habitat_llm.examples.emtom --config-name game/time_game \
    mode="cli" \
    evaluation.save_video=True \
    +game.manual_agents=[] \
    +game.manual_obs_popup=true
