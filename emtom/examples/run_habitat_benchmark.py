#!/usr/bin/env python3
"""
Example script for running the EMTOM benchmark with Habitat integration.

This script demonstrates running EMTOM tasks in the Habitat simulator
with proper video recording (third-person and first-person views).

Usage:
    # Run with Habitat (requires proper config and scene data)
    python emtom/examples/run_habitat_benchmark.py \\
        --config-name examples/planner_multi_agent_demo_config \\
        evaluation.save_video=True

    # The script uses the existing habitat_llm configuration system
"""

import argparse
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
from habitat_llm.examples.example_utils import DebugVideoUtil, FirstPersonVideoRecorder
from habitat_llm.utils import cprint, setup_config, fix_config

from emtom.benchmark import (
    ScriptedAgent,
    BenchmarkEvaluator,
    BenchmarkResults,
    HabitatRunConfig,
)
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


def create_scripted_agents_for_task(task: GeneratedTask) -> dict:
    """Create scripted agents with predefined actions for testing."""
    agents = {}

    for agent_id in task.agent_roles.keys():
        knowledge = task.agent_knowledge.get(agent_id, [])
        script = []

        if "inverse" in str(task.required_mechanics).lower():
            if "Doors in this house work backwards" in knowledge:
                script = [
                    {"action": "close", "target": "door_1"},
                    {"action": "communicate", "target": "agent_1" if agent_id == "agent_0" else "agent_0",
                     "message": "The doors work backwards! Use 'close' to open them."},
                    {"action": "close", "target": "door_3"},
                ]
            else:
                script = [
                    {"action": "open", "target": "door_2"},
                    {"action": "wait"},
                    {"action": "close", "target": "door_2"},
                ]
        elif "counting" in str(task.required_mechanics).lower():
            if agent_id == "agent_0":
                script = [
                    {"action": "move", "target": "vault_room"},
                    {"action": "press", "target": "security_button"},
                    {"action": "move", "target": "waiting_room"},
                ]
            else:
                script = [
                    {"action": "wait"},
                    {"action": "move", "target": "vault_room"},
                    {"action": "press", "target": "security_button"},
                    {"action": "press", "target": "security_button"},
                ]
        else:
            script = [{"action": "look"}, {"action": "wait"}]

        agents[agent_id] = ScriptedAgent(agent_id, script)

    return agents


class EMTOMHabitatRunner:
    """
    Runs EMTOM benchmark tasks in Habitat with video recording.

    Uses the existing habitat_llm video utilities (DebugVideoUtil, FirstPersonVideoRecorder)
    to record third-person and first-person views of the agents.
    """

    def __init__(
        self,
        env_interface: EnvironmentInterface,
        output_dir: str,
        save_video: bool = True,
        video_fps: int = 30,
    ):
        self.env_interface = env_interface
        self.output_dir = output_dir
        self.save_video = save_video
        self.video_fps = video_fps

        # Video recording
        self._dvu = None
        self._fpv_recorder = None
        self._fpv_recorder_failed = False

        if save_video:
            self._setup_video_recording()

    def _setup_video_recording(self) -> None:
        """Initialize video recording utilities."""
        os.makedirs(self.output_dir, exist_ok=True)

        # Third-person split-screen video
        self._dvu = DebugVideoUtil(
            self.env_interface,
            self.output_dir,
            unique_postfix=True,
        )

        # First-person video recorder
        try:
            self._fpv_recorder = FirstPersonVideoRecorder(
                self.env_interface,
                output_dir=self.output_dir,
                fps=self.video_fps,
            )
        except Exception as e:
            cprint(f"[EMTOM] Failed to initialize FPV recorder: {e}", "yellow")
            self._fpv_recorder_failed = True

    def run_task(
        self,
        task: GeneratedTask,
        max_steps: int = 100,
        verbose: bool = True,
    ) -> dict:
        """
        Run a single EMTOM task in Habitat.

        This executes the agents in the Habitat environment and records video.
        """
        if verbose:
            cprint(f"\n{'='*60}", "blue")
            cprint(f"EMTOM TASK: {task.title}", "blue")
            cprint(f"{'='*60}", "blue")
            print(f"Description: {task.description}")
            print(f"Mechanics: {task.required_mechanics}")
            print(f"Goal: {task.success_condition.description}")

        # Clear video buffers
        if self._dvu:
            self._dvu.frames.clear()
        if self._fpv_recorder:
            self._fpv_recorder._frames = {}

        # Get initial observations
        observations = self.env_interface.get_observations()

        # Record initial frame
        if self.save_video:
            self._record_frame(observations, {})

        result = {
            "task_id": task.task_id,
            "task_title": task.title,
            "status": "running",
            "steps": 0,
            "video_paths": {},
        }

        # Run for max_steps (in real usage, agents would control the loop)
        for step in range(max_steps):
            if verbose and step % 10 == 0:
                print(f"  Step {step}...")

            # In a real benchmark, agents would select actions here
            # For now, we just step the environment to show video recording works
            try:
                # Create no-op actions for each agent
                # low_level_actions is Dict[agent_id, np.ndarray]
                # Each agent's action is a FULL action vector (joint space), filled with zeros
                # for a no-op. When added together they form the final action.
                import numpy as np
                from habitat_baselines.utils.common import get_num_actions

                # Get the total action dimension from the flattened action space
                action_space = self.env_interface.action_space
                total_action_dim = get_num_actions(action_space)

                # Determine number of agents
                num_agents = 2
                try:
                    num_agents = len(self.env_interface.conf.evaluation.agents)
                except Exception:
                    pass

                # Each agent returns a FULL action vector (size = total_action_dim)
                # For no-op, all zeros works since actions are added together
                low_level_actions = {}
                for agent_idx in range(num_agents):
                    low_level_actions[agent_idx] = np.zeros(total_action_dim, dtype=np.float32)

                # Step environment
                obs, reward, done, info = self.env_interface.step(low_level_actions)
                observations = self.env_interface.parse_observations(obs)

                # Record frame
                if self.save_video:
                    # Create action info for video overlay
                    action_info = {0: ("exploring", ""), 1: ("exploring", "")}
                    self._record_frame(observations, action_info)

                result["steps"] = step + 1

                if done:
                    result["status"] = "done"
                    break

            except Exception as e:
                cprint(f"[EMTOM] Error during step {step}: {e}", "red")
                traceback.print_exc()
                result["status"] = "error"
                result["error"] = str(e)
                break

        # Save videos
        if self.save_video:
            video_paths = self._save_videos(task.task_id)
            result["video_paths"] = video_paths

        if verbose:
            cprint(f"Task completed: {result['status']} in {result['steps']} steps", "green")

        return result

    def _record_frame(self, observations: dict, actions: dict) -> None:
        """Record a video frame."""
        # Third-person video
        if self._dvu:
            try:
                self._dvu._store_for_video(observations, actions, popup_images={})
            except Exception as e:
                pass  # Silently continue if frame recording fails

        # First-person video
        if self._fpv_recorder and not self._fpv_recorder_failed:
            try:
                self._fpv_recorder.record_step(observations)
            except Exception as e:
                pass

    def _save_videos(self, task_id: str) -> dict:
        """Save recorded videos."""
        video_paths = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        postfix = f"{task_id}_{timestamp}"

        # Save third-person video
        if self._dvu and self._dvu.frames:
            try:
                self._dvu._make_video(play=False, postfix=postfix)
                video_dir = os.path.join(self.output_dir, "videos")
                video_path = os.path.join(video_dir, f"video-{postfix}.mp4")
                if os.path.exists(video_path):
                    video_paths["third_person"] = video_path
                    cprint(f"[EMTOM] Third-person video saved: {video_path}", "green")
            except Exception as e:
                cprint(f"[EMTOM] Failed to save third-person video: {e}", "yellow")

        # Save first-person videos
        if self._fpv_recorder and not self._fpv_recorder_failed:
            try:
                fpv_paths = self._fpv_recorder.save(postfix=postfix)
                for agent_name, path in fpv_paths.items():
                    video_paths[f"fpv_{agent_name}"] = path
                    cprint(f"[EMTOM] FPV video saved for {agent_name}: {path}", "green")
            except Exception as e:
                cprint(f"[EMTOM] Failed to save FPV videos: {e}", "yellow")

        return video_paths


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

    # Print initial world state
    try:
        from habitat_llm.utils.world_graph import print_all_entities
        cprint("\nInitial world entities:", "cyan")
        print_all_entities(env_interface.perception.gt_graph)
    except Exception:
        pass

    # Setup output directory
    output_dir = config.paths.results_dir
    os.makedirs(output_dir, exist_ok=True)
    cprint(f"Output directory: {output_dir}", "blue")

    # Create EMTOM runner
    runner = EMTOMHabitatRunner(
        env_interface=env_interface,
        output_dir=output_dir,
        save_video=True,
        video_fps=30,
    )

    # Load EMTOM tasks (or use episode instruction)
    task_file = "data/tasks/emtom_challenges_20251212_191827.json"
    if os.path.exists(task_file):
        tasks = load_tasks(task_file)
        cprint(f"Loaded {len(tasks)} EMTOM tasks", "green")

        # Run first task as demonstration
        if tasks:
            result = runner.run_task(tasks[0], max_steps=50, verbose=True)
            print(f"\nResult: {json.dumps(result, indent=2)}")
    else:
        cprint(f"Task file not found: {task_file}", "yellow")
        cprint("Running demonstration with episode from dataset...", "blue")

        # Create a simple demo task
        demo_task = GeneratedTask(
            task_id="demo_task",
            title="Habitat Video Demo",
            category="coordination",
            description="Demonstration of video recording in Habitat",
            initial_world_state={},
            required_mechanics=[],
            num_agents=2,
            agent_roles={"agent_0": "Explorer", "agent_1": "Observer"},
            agent_knowledge={},
            subtasks=[],
            success_condition=None,
            failure_conditions=[],
            difficulty=1,
            estimated_steps=10,
        )

        # Minimal success condition
        from emtom.task_gen.task_generator import SuccessCondition
        demo_task.success_condition = SuccessCondition(
            description="Complete exploration",
            required_states=[],
            time_limit=50,
        )

        result = runner.run_task(demo_task, max_steps=30, verbose=True)
        print(f"\nResult: {json.dumps(result, indent=2)}")

    # Cleanup
    env_interface.env.close()
    cprint("\nBenchmark complete!", "green")


if __name__ == "__main__":
    cprint("\nEMTOM Habitat Benchmark Runner", "blue")
    cprint("This script runs EMTOM tasks in Habitat with video recording.\n", "blue")

    if len(sys.argv) < 2:
        cprint("Usage: python run_habitat_benchmark.py --config-name <config>", "yellow")
        cprint("Example: python run_habitat_benchmark.py --config-name examples/planner_multi_agent_demo_config", "yellow")
        sys.exit(1)

    main()
