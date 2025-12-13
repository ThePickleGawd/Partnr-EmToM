#!/usr/bin/env python3
# Script to make the robot agent do random walks around the scene

import sys
import random
import subprocess
from typing import List, Any

# Append parent directory path
sys.path.append("../../..")

import omegaconf
import hydra
from hydra.utils import instantiate

from habitat_llm.utils import cprint, setup_config, fix_config

# Patch the video player to use mpv/ffplay instead of broken OpenCV Qt
def play_video_x11(filename: str) -> None:
    """Play video using mpv which works better with X11 forwarding"""
    print(f"     ...playing video with mpv (press 'q' to skip)...")
    try:
        # Play video once with mpv (works great with X11 forwarding)
        subprocess.run([
            "mpv",
            "--keep-open=no",  # Close after playing
            "--really-quiet",  # Less verbose output
            filename
        ], check=False)
    except Exception as e:
        print(f"Could not play video with mpv: {e}")
        print(f"Video saved to: {filename}")

from habitat_llm.agent.env import (
    EnvironmentInterface,
    register_actions,
    register_measures,
    register_sensors,
    remove_visual_sensors,
)
from habitat_llm.agent.env.dataset import CollaborationDatasetV0
from habitat.sims.habitat_simulator.debug_visualizer import DebugVisualizer
from habitat_llm.utils.sim import init_agents
from habitat_llm.examples.example_utils import execute_skill, DebugVideoUtil
from habitat_llm.utils.world_graph import (
    print_all_entities,
    print_furniture_entity_handles,
    print_object_entity_handles,
)

# Monkey-patch DebugVideoUtil to use our X11-friendly video player
DebugVideoUtil.play_video = lambda self, filename: play_video_x11(filename)


@hydra.main(
    config_path="../../../habitat_llm/conf",
    config_name="examples/skill_runner_default_config.yaml",
)
def run_random_walks(config: omegaconf.DictConfig) -> None:
    """
    Main function that loads a scene and makes the robot agent do random walks.

    :param config: Hydra config (use same config as skill_runner)
    """
    fix_config(config)

    # Setup seed for reproducibility
    seed = 47668090

    # Setup hardcoded config overrides
    with omegaconf.open_dict(config):
        config_dict = omegaconf.OmegaConf.create(
            omegaconf.OmegaConf.to_container(config.habitat, resolve=True)
        )
        config_dict.dataset.metadata = {"metadata_folder": "data/hssd-hab/metadata"}
        config.habitat = config_dict

    config = setup_config(config, seed)

    assert config.env == "habitat", "Only valid for Habitat."

    # Video configuration
    show_videos = config.get("skill_runner_show_videos", True)
    make_video = config.evaluation.save_video or show_videos

    if not make_video:
        remove_visual_sensors(config)

    # Register sensors, actions, and measures
    register_sensors(config)
    register_actions(config)
    register_measures(config)

    # Create dataset
    dataset = CollaborationDatasetV0(config.habitat.dataset)
    print(f"Loading EpisodeDataset from: {config.habitat.dataset.data_path}")

    # Initialize environment interface
    env_interface = EnvironmentInterface(config, dataset=dataset)

    # Select episode (by index or id)
    if hasattr(config, "skill_runner_episode_index"):
        episode_index = config.skill_runner_episode_index
        print(f"Loading episode_index = {episode_index}")
        env_interface.env.habitat_env.episode_iterator.set_next_episode_by_index(
            episode_index
        )
    elif hasattr(config, "skill_runner_episode_id"):
        episode_id = config.skill_runner_episode_id
        print(f"Loading episode_id = {episode_id}")
        env_interface.env.habitat_env.episode_iterator.set_next_episode_by_id(
            str(episode_id)
        )

    env_interface.reset_environment()

    # Initialize planner
    planner_conf = config.evaluation.planner
    planner = instantiate(planner_conf)
    planner = planner(env_interface=env_interface)
    agent_config = config.evaluation.agents
    planner.agents = init_agents(agent_config, env_interface)
    planner.reset()

    sim = env_interface.sim

    # Show topdown map if requested
    if config.get("skill_runner_show_topdown", False):
        dbv = DebugVisualizer(sim, config.paths.results_dir)
        dbv.create_dbv_agent(resolution=(1000, 1000))
        top_down_map = dbv.peek("stage")
        if show_videos:
            top_down_map.show()
        if config.evaluation.save_video:
            top_down_map.save(output_path=config.paths.results_dir, prefix="topdown")
        dbv.remove_dbv_agent()
        dbv.create_dbv_agent()
        dbv.remove_dbv_agent()

    # Print scene information
    cprint("=== Random Walk Script Started ===", "green")
    cprint(
        f"Episode ID: {sim.ep_info.episode_id}, Scene: {sim.ep_info.scene_id}",
        "green",
    )

    # Print all entities in the scene
    print("\n=== Entities in Scene ===")
    print_all_entities(env_interface.perception.gt_graph)
    print_furniture_entity_handles(env_interface.perception.gt_graph)
    print_object_entity_handles(env_interface.perception.gt_graph)

    # Get the robot agent (agent 0)
    robot_agent = planner.get_agent_from_uid(env_interface.robot_agent_uid)
    cprint(f"\nUsing Robot Agent (uid={env_interface.robot_agent_uid})", "blue")

    # Get all furniture in the scene
    world_graph = env_interface.world_graph[robot_agent.uid]
    all_furniture = world_graph.get_all_furnitures()

    if not all_furniture:
        cprint("No furniture found in the scene! Cannot perform random walks.", "red")
        return

    cprint(f"\nFound {len(all_furniture)} furniture items in the scene", "blue")
    cprint(f"Sample furniture: {[f.name for f in all_furniture[:5]]}", "blue")

    # Configuration for random walks
    num_walks = config.get("num_random_walks", 5)  # Default: 5 random walks
    cprint(f"\nPerforming {num_walks} random walks around the scene...\n", "green")

    # Collect frames for cumulative video
    cumulative_frames: List[Any] = []

    # Perform random walks
    for walk_idx in range(num_walks):
        # Pick a random furniture to navigate to
        target_furniture = random.choice(all_furniture)

        cprint(f"=== Walk {walk_idx + 1}/{num_walks} ===", "yellow")
        cprint(f"Navigating to: {target_furniture.name}", "yellow")

        # Create high-level action to navigate to this furniture
        # Format: {agent_id: (skill_name, target, None)}
        high_level_skill_actions = {
            robot_agent.uid: ("Navigate", target_furniture.name, None)
        }

        try:
            # Execute the navigate skill
            responses, _, frames = execute_skill(
                high_level_skill_actions,
                planner,
                vid_postfix=f"walk_{walk_idx}_",
                make_video=make_video,
                play_video=show_videos,
            )

            # Print the response
            response_msg = responses[robot_agent.uid]
            cprint(f"Result: {response_msg}\n", "green")

            # Accumulate frames
            cumulative_frames.extend(frames)

        except Exception as e:
            cprint(f"Failed to execute walk {walk_idx + 1}: {str(e)}", "red")
            continue

    # Create cumulative video of all walks
    if len(cumulative_frames) > 0 and make_video:
        cprint("\n" + "="*60, "green")
        cprint("Creating COMBINED video of all walks...", "green")
        cprint("="*60, "green")

        dvu = DebugVideoUtil(env_interface, config.paths.results_dir)
        dvu.frames = cumulative_frames
        dvu._make_video(postfix="all_random_walks", play=show_videos)

        cumulative_video_path = f"{config.paths.results_dir}/videos/video-all_random_walks.mp4"
        cprint(f"\n✓ Combined video saved to: {cumulative_video_path}", "green")
        cprint(f"  Total frames: {len(cumulative_frames)}", "blue")

        if show_videos:
            cprint("\nPlaying combined video of all walks...", "yellow")

    cprint("\n=== Random Walk Script Completed ===", "green")
    cprint(f"Completed {num_walks} random walks.", "green")
    cprint(f"Videos saved in: {config.paths.results_dir}/videos/", "blue")


if __name__ == "__main__":
    cprint(
        "\nStarting random walk script for robot agent in Habitat environment.",
        "blue",
    )

    run_random_walks()

    cprint(
        "\nRandom walk script finished.",
        "blue",
    )
