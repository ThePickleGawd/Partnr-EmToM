# EMTOM Games Overview

Maintain this doc as new games are added. It’s a map of where to edit configs, tools, and video settings so future changes are easy to make.

## Key Config Entry Points
- Game defaults: `habitat_llm/conf/game/<game>.yaml` (planner overrides, partial obs/asymmetry, agent overrides, instructions).
- Game-specific agent configs: `habitat_llm/conf/agent/game/<game>/agent_0.yaml` and `agent_1.yaml` (tool loadout + env).
- Sensor/camera setup: `habitat_llm/conf/habitat_conf/emtom_oracle_spot_kinematic_multi_agent_sym.yaml` (identical Spot robots, head/arm/jaw RGB/depth/panoptic).
- Trajectory logger: `habitat_llm/conf/trajectory/trajectory_logger.yaml` (save path, cameras).

## Tools (Hydra)
- Agent tool loadouts live in the game-specific agent configs. They pull from:
  - Motor skills: `oracle_nav`, `oracle_explore`, `wait` (`conf/tools/motor_skills/*.yaml`).
  - Perception: `communication_tool`, `find_receptacle_tool`, `find_object_tool`, `find_room_tool`, `find_agent_action_tool` (`conf/tools/perception/*.yaml`).
- Game tools are injected at runtime by the game spec (`game/<game>/<game>.py`). Example: bomb game adds `DefuseBomb`/`DefuseBombTool` when an agent is in the bomb room.

## First-Person Video / Trajectories
- Trajectory saving: `trajectory_logger.yaml` + `EnvironmentInterface.save_trajectory_step` writing to `data/trajectories/...` using `trajectory.agent_names` and `camera_prefixes` (e.g., `articulated_agent_jaw` for both robots).
- FPV video: `FirstPersonVideoRecorder` (`habitat_llm/examples/example_utils.py`) uses the trajectory config to pick observation keys (e.g., `agent_0_articulated_agent_jaw_rgb`, `agent_1_articulated_agent_jaw_rgb`) and writes per-agent MP4s under `outputs/.../videos/`.
- Third-person video: `DebugVideoUtil` captures `third_rgb` when `evaluation.save_video=True`.

## Running (example: bomb game)
```bash
source /home/dylanlu/miniconda3/etc/profile.d/conda.sh
conda activate habitat-llm
set -a && source .env && set +a   # loads OPENAI_API_KEY
./bomb_game_run.sh
```
Use `timeout 30s ./bomb_game_run.sh` for quick tests.

## Notes
- Partial observability and agent asymmetry are set in each game config.
- Agent configs for games live under `conf/agent/game/<game>/agent_{0,1}.yaml`; global `tom_agent` is not used by games.
- If tools are “not found,” ensure the game’s agent overrides point to the game-specific agent configs and that observation keys match the habitat config.
