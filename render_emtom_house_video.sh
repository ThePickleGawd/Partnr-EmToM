#!/usr/bin/env bash
set -euo pipefail

# Render a non-interactive flythrough video of the emtom game house.
# Defaults point to the generated house in data/emtom (scene id 102817140).
# Override via env vars: SCENE_PATH, SCENE_DATASET, OUTPUT, DURATION, FPS,
# SPEED, MIN_SEGMENT, WIDTH, HEIGHT, CAMERA_HEIGHT, SEED.

SCENE_PATH="${SCENE_PATH:-data/hssd-hab/scenes-partnr-filtered/102817140.scene_instance.json}"
SCENE_DATASET="${SCENE_DATASET:-data/hssd-hab/hssd-hab-partnr.scene_dataset_config.json}"
OUTPUT="${OUTPUT:-outputs/emtom_house.mp4}"
DURATION="${DURATION:-}"
FPS="${FPS:-30}"
SPEED="${SPEED:-2}"
MIN_SEGMENT="${MIN_SEGMENT:-1.5}"
WIDTH="${WIDTH:-800}"
HEIGHT="${HEIGHT:-600}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-1.5}"
SEED="${SEED:-}"
WAYPOINTS="${WAYPOINTS:-}"
ARTICULATED_MODE="${ARTICULATED_MODE:-auto}"

out_dir="$(dirname "${OUTPUT}")"
if [[ -n "${out_dir}" && "${out_dir}" != "." ]]; then
  mkdir -p "${out_dir}"
fi

python -m habitat_llm.examples.house_video \
  --scene "${SCENE_PATH}" \
  --dataset "${SCENE_DATASET}" \
  --output "${OUTPUT}" \
  ${DURATION:+--duration "${DURATION}"} \
  --fps "${FPS}" \
  --speed "${SPEED}" \
  --min-segment "${MIN_SEGMENT}" \
  ${WAYPOINTS:+--waypoints "${WAYPOINTS}"} \
  --width "${WIDTH}" \
  --height "${HEIGHT}" \
  --articulated-mode "${ARTICULATED_MODE}" \
  --camera-height "${CAMERA_HEIGHT}" \
  ${SEED:+--seed "${SEED}"} \
  "$@"
