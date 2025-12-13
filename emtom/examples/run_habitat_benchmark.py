#!/usr/bin/env python3
"""
Example script for running the EMTOM benchmark with Habitat integration.

This script demonstrates running EMTOM tasks in the Habitat simulator
with proper video recording (third-person and first-person views) and
LLM-driven agent actions.

Usage:
    # Run with Habitat (requires proper config and scene data)
    python emtom/examples/run_habitat_benchmark.py \
        --config-name examples/planner_multi_agent_demo_config \
        evaluation.save_video=True

    # The script uses the existing habitat_llm configuration system
"""

import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from habitat_llm.agent.env import (
    EnvironmentInterface,
    register_actions,
    register_measures,
    register_sensors,
    remove_visual_sensors,
)
from habitat_llm.agent.env.dataset import CollaborationDatasetV0
from habitat_llm.evaluation import DecentralizedEvaluationRunner
from habitat_llm.utils import cprint, setup_config, fix_config

from emtom.task_gen import GeneratedTask

# Import mechanics to register them
from emtom.mechanics import inverse_open, remote_switch, counting_trigger


def load_tasks(task_file: str) -> list:
    """Load tasks from JSON file."""
    with open(task_file) as f:
        data = json.load(f)

    tasks = []
    for task_data in data.get("tasks", []):
        task = GeneratedTask.from_dict(task_data)
        tasks.append(task)

    return tasks


def task_to_instruction(task: GeneratedTask) -> str:
    """Convert an EMTOM task to a natural language instruction for the agents."""
    # Build an instruction that includes the task context
    instruction_parts = [
        f"Task: {task.title}",
        f"Description: {task.description}",
        f"Goal: {task.success_condition.description}",
    ]

    # Add any special knowledge that agents have
    if task.agent_knowledge:
        instruction_parts.append("\nAgent-specific knowledge:")
        for agent_id, knowledge in task.agent_knowledge.items():
            if knowledge:
                instruction_parts.append(f"  {agent_id}: {', '.join(knowledge)}")

    return "\n".join(instruction_parts)


@hydra.main(version_base=None, config_path="../../habitat_llm/conf")
def main(config: DictConfig) -> None:
    """Main entry point with Hydra configuration."""
    fix_config(config)
    config = setup_config(config, seed=47668090)

    # Ensure video saving is enabled
    with open_dict(config):
        config.evaluation.save_video = True

    cprint("\n" + "="*60, "blue")
    cprint("EMTOM Habitat Benchmark", "blue")
    cprint("="*60, "blue")

    # Register Habitat components
    register_sensors(config)
    register_actions(config)
    register_measures(config)

    # Create dataset
    dataset = CollaborationDatasetV0(config.habitat.dataset)
    cprint(f"Loaded dataset with {len(dataset.episodes)} episodes", "green")

    # Create environment interface
    cprint("Initializing Habitat environment...", "blue")
    env_interface = EnvironmentInterface(config, dataset=dataset, init_wg=False)

    try:
        env_interface.initialize_perception_and_world_graph()
    except Exception as e:
        cprint(f"Warning: Failed to initialize world graph: {e}", "yellow")

    cprint("Environment initialized!", "green")

    # Create evaluation runner - this handles planners, agents, video recording
    cprint("Initializing evaluation runner with LLM planners...", "blue")
    eval_runner = DecentralizedEvaluationRunner(config.evaluation, env_interface)
    cprint(f"Evaluation runner created: {eval_runner}", "green")

    # Print agent info
    cprint(f"\nAgents: {eval_runner.agent_list}", "blue")

    # Setup output directory
    output_dir = config.paths.results_dir
    os.makedirs(output_dir, exist_ok=True)
    cprint(f"Output directory: {output_dir}", "blue")

    # Load EMTOM tasks or use episode instruction
    task_file = "data/tasks/emtom_challenges_20251212_191827.json"

    if os.path.exists(task_file):
        tasks = load_tasks(task_file)
        cprint(f"Loaded {len(tasks)} EMTOM tasks", "green")

        # Run first task as demonstration
        if tasks:
            task = tasks[0]
            instruction = task_to_instruction(task)

            cprint(f"\n{'='*60}", "blue")
            cprint(f"EMTOM TASK: {task.title}", "blue")
            cprint(f"{'='*60}", "blue")
            print(f"Description: {task.description}")
            print(f"Mechanics: {task.required_mechanics}")
            print(f"Goal: {task.success_condition.description}")
            print(f"\nInstruction to agents:\n{instruction}")

            # Run the instruction using the evaluation runner
            # This will use LLM planners to generate actions and record video
            try:
                cprint("\nStarting task execution with LLM planners...", "blue")
                info = eval_runner.run_instruction(
                    instruction=instruction,
                    output_name=f"emtom_{task.task_id}"
                )

                cprint("\nTask execution completed!", "green")
                print(f"Results: {json.dumps(info, indent=2, default=str)}")

            except Exception as e:
                cprint(f"Error during task execution: {e}", "red")
                traceback.print_exc()
                # Try to save video even on error
                cprint("Attempting to save video despite error...", "yellow")
                try:
                    if hasattr(eval_runner, 'dvu') and eval_runner.dvu is not None:
                        eval_runner.dvu._make_video(play=False, postfix=f"emtom_{task.task_id}_error")
                        cprint("Third-person video saved!", "green")
                    if hasattr(eval_runner, '_fpv_recorder') and eval_runner._fpv_recorder is not None:
                        eval_runner._make_first_person_videos()
                        cprint("First-person videos saved!", "green")
                except Exception as ve:
                    cprint(f"Could not save video: {ve}", "red")
    else:
        cprint(f"Task file not found: {task_file}", "yellow")
        cprint("Running with episode instruction from dataset...", "blue")

        # Get the instruction from the current episode
        curr_env = env_interface.env.env.env._env
        instruction = curr_env.current_episode.instruction

        cprint(f"\nEpisode instruction: {instruction}", "blue")

        try:
            info = eval_runner.run_instruction(
                instruction=instruction,
                output_name="emtom_episode_demo"
            )
            cprint("\nEpisode completed!", "green")
            print(f"Results: {json.dumps(info, indent=2, default=str)}")
        except Exception as e:
            cprint(f"Error during execution: {e}", "red")
            traceback.print_exc()

    # Cleanup
    env_interface.env.close()
    cprint("\nBenchmark complete!", "green")
    cprint(f"Check {output_dir} for videos and logs", "blue")


if __name__ == "__main__":
    cprint("\nEMTOM Habitat Benchmark Runner", "blue")
    cprint("This script runs EMTOM tasks in Habitat with LLM planners and video recording.\n", "blue")

    if len(sys.argv) < 2:
        cprint("Usage: python run_habitat_benchmark.py --config-name <config>", "yellow")
        cprint("Example: python run_habitat_benchmark.py --config-name examples/planner_multi_agent_demo_config", "yellow")
        sys.exit(1)

    main()
