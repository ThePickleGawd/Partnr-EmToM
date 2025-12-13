# Height Difference - Random Walk Script

## Overview

This script loads a Habitat scene and makes the robot agent perform random walks by exploring different rooms in the environment.

## How It Works

1. **Loads the scene** - Uses the same infrastructure as `skill_runner`
2. **Deploys the robot agent** - Automatically selects agent 0 (robot)
3. **Random walks** - The robot navigates to random furniture using the `Navigate` skill
4. **Visualization** - Shows videos of each walk and creates a cumulative video

## Usage

### Basic Command

```bash
HYDRA_FULL_ERROR=1 python -m main.mechanics.height_difference.height_difference \
    hydra.run.dir="." \
    +skill_runner_show_topdown=True \
    habitat.dataset.data_path=data/datasets/partnr_episodes/v0_0/val_mini.json.gz \
    +skill_runner_episode_id="334"
```

### Configuration Options

**Episode Selection** (mutually exclusive):
- `+skill_runner_episode_index=0` - Select episode by index
- `+skill_runner_episode_id="334"` - Select episode by ID

**Visualization**:
- `+skill_runner_show_topdown=True` - Show top-down map at start
- `+skill_runner_show_videos=True` - Show videos after each walk (default: True)
- `evaluation.save_video=True` - Save videos to disk (default: True)

**Random Walk Configuration**:
- `+num_random_walks=5` - Number of random walks to perform (default: 5)

**Output**:
- `paths.results_dir="./results/"` - Directory for saving videos/results

### Example Commands

**Quick test with 3 walks:**
```bash
HYDRA_FULL_ERROR=1 python -m main.mechanics.height_difference.height_difference \
    hydra.run.dir="." \
    +skill_runner_episode_id="334" \
    +num_random_walks=3
```

**No videos, just run:**
```bash
HYDRA_FULL_ERROR=1 python -m main.mechanics.height_difference.height_difference \
    hydra.run.dir="." \
    +skill_runner_episode_id="334" \
    +skill_runner_show_videos=False \
    evaluation.save_video=False
```

**Custom dataset:**
```bash
HYDRA_FULL_ERROR=1 python -m main.mechanics.height_difference.height_difference \
    hydra.run.dir="." \
    habitat.dataset.data_path=data/datasets/partnr_episodes/v0_0/train.json.gz \
    +skill_runner_episode_index=0 \
    +num_random_walks=10
```

## What Happens

1. Scene is loaded from the specified episode
2. All furniture in the scene is identified
3. For each walk:
   - A random furniture item is selected
   - Robot navigates to that furniture
   - Video is recorded (if enabled)
4. A cumulative video of all walks is created
5. Results are saved to the results directory

## Output Files

- Individual walk videos: `{results_dir}/walk_0_*.mp4`, `walk_1_*.mp4`, etc.
- Cumulative video: `{results_dir}/all_random_walks.mp4`
- Top-down map: `{results_dir}/topdown.png` (if enabled)

## Key Differences from skill_runner

| Feature | skill_runner | height_difference.py |
|---------|--------------|----------------------|
| Agent selection | Interactive (user picks) | Automatic (robot only) |
| Action selection | Interactive (user types commands) | Automatic (random walks) |
| Navigation target | User specifies | Random furniture |
| Execution | One command at a time | Batch of N walks |
| Use case | Debugging/testing | Automated exploration |
