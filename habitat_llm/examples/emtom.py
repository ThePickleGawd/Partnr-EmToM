#!/usr/bin/env python3
# isort: skip_file

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import csv
import sys
import time
import os
import traceback
import json
import shutil
from omegaconf import OmegaConf


# append the path of the
# parent directory
sys.path.append("..")

import hydra
from typing import Dict

from torch import multiprocessing as mp

from habitat_llm.agent.env.evaluation.evaluation_functions import (
    aggregate_measures,
)

from habitat_llm.utils import cprint, setup_config, fix_config


from habitat_llm.agent.env import (
    EnvironmentInterface,
    register_actions,
    register_measures,
    register_sensors,
    remove_visual_sensors,
)
from habitat_llm.evaluation import (
    CentralizedEvaluationRunner,
    DecentralizedEvaluationRunner,
    EvaluationRunner,
)
from habitat_llm.agent.env.dataset import CollaborationDatasetV0
from habitat_baselines.utils.info_dict import extract_scalars_from_info
from game.game import GameOrchestrator
from game.bomb_game import BombGameSpec
from game.time_game import TimeGameSpec
from game.habitat_adapter import HabitatEnvironmentAdapter
from game.game_runner import GameDecentralizedEvaluationRunner
from omegaconf import OmegaConf, open_dict


def _ensure_game_config(config):
    """
    Make sure a game config is present. If the Hydra run did not specify a /game
    group, inject a disabled placeholder so downstream logic can rely on it.
    """
    if not hasattr(config, "game"):
        config.game = OmegaConf.create({"enable": False})
    defaults = {
        "manual_agents": [],
        "manual_obs_popup": False,
    }
    with open_dict(config.game):
        for k, v in defaults.items():
            if not hasattr(config.game, k):
                setattr(config.game, k, v)
    return config


def get_output_file(config, env_interface):
    dataset_file = env_interface.conf.habitat.dataset.data_path.split("/")[-1]
    episode_id = env_interface.env.env.env._env.current_episode.episode_id
    output_file = os.path.join(
        config.paths.results_dir,
        dataset_file,
        "stats",
        f"{episode_id}.json",
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    return output_file


# Function to write data to the CSV file
def write_to_csv(file_name, result_dict):
    # Sort the dictionary by keys
    # Needed to ensure sanity in multi-process operation
    result_dict = dict(sorted(result_dict.items()))
    with open(file_name, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=result_dict.keys())

        # Check if the file is empty (to write headers)
        file.seek(0, 2)
        file_empty = file.tell() == 0
        if file_empty:
            writer.writeheader()

        writer.writerow(result_dict)


def save_exception_message(config, env_interface):
    output_file = get_output_file(config, env_interface)
    exc_string = traceback.format_exc()
    failure_dict = {"success": False, "info": str(exc_string)}
    with open(output_file, "w+") as f:
        f.write(json.dumps(failure_dict))


def save_success_message(config, env_interface, info):
    output_file = get_output_file(config, env_interface)
    failure_dict = {"success": True, "stats": json.dumps(info)}
    with open(output_file, "w+") as f:
        f.write(json.dumps(failure_dict))


# Write the config file into the results folders
def write_config(config):
    dataset_file = config.habitat.dataset.data_path.split("/")[-1]
    output_file = os.path.join(config.paths.results_dir, dataset_file)
    os.makedirs(output_file, exist_ok=True)
    with open(f"{output_file}/config.yaml", "w+") as f:
        f.write(OmegaConf.to_yaml(config))

    # Copy over the RLM config
    planner_configs = []
    suffixes = []
    if "planner" in config.evaluation:
        # Centralized
        if "plan_config" in config.evaluation.planner is not None:
            planner_configs = [config.evaluation.planner.plan_config]
            suffixes = [""]
    else:
        for agent_name in config.evaluation.agents:
            suffixes.append(f"_{agent_name}")
            planner_configs.append(
                config.evaluation.agents[agent_name].planner.plan_config
            )

    for plan_config, suffix_rlm in zip(planner_configs, suffixes):
        if "llm" in plan_config and "serverdir" in plan_config.llm:
            yaml_rlm_path = plan_config.llm.serverdir
            if len(yaml_rlm_path) > 0:
                yaml_rlm_file = f"{yaml_rlm_path}/config.yaml"
                if os.path.isfile(yaml_rlm_file):
                    shutil.copy(
                        yaml_rlm_file, f"{output_file}/config_rlm{suffix_rlm}.yaml"
                    )


# Method to load agent planner from the config
@hydra.main(config_path="../conf")
def run_eval(config):
    fix_config(config)
    # Setup a seed
    seed = 47668090
    t0 = time.time()
    # Setup config
    config = setup_config(config, seed)
    config = _ensure_game_config(config)
    dataset = CollaborationDatasetV0(config.habitat.dataset)

    write_config(config)
    if config.get("resume", False):
        dataset_file = config.habitat.dataset.data_path.split("/")[-1]
        # stats_dir = os.path.join(config.paths.results_dir, dataset_file, "stats")
        plan_log_dir = os.path.join(
            config.paths.results_dir, dataset_file, "planner-log"
        )

        # Find incomplete episodes
        incomplete_episodes = []
        for episode in dataset.episodes:
            episode_id = episode.episode_id
            # stats_file = os.path.join(stats_dir, f"{episode_id}.json")
            planlog_file = os.path.join(
                plan_log_dir, f"planner-log-episode_{episode_id}_0.json"
            )
            if not os.path.exists(planlog_file):
                incomplete_episodes.append(episode)
        print(
            f"Resuming with {len(incomplete_episodes)} incomplete episodes: {[e.episode_id for e in incomplete_episodes]}"
        )
        # Update dataset with only incomplete episodes
        dataset = CollaborationDatasetV0(
            config=config.habitat.dataset, episodes=incomplete_episodes
        )

    # filter episodes by mod for running on multiple nodes
    if config.get("episode_mod_filter", None) is not None:
        rem, mod = config.episode_mod_filter
        episode_subset = [x for x in dataset.episodes if int(x.episode_id) % mod == rem]
        print(f"Mod filter: {rem}, {mod}")
        print(f"Episodes: {[e.episode_id for e in episode_subset]}")
        dataset = CollaborationDatasetV0(
            config=config.habitat.dataset, episodes=episode_subset
        )

    num_episodes = len(dataset.episodes)
    if config.num_proc == 1:
        if config.get("episode_indices", None) is not None:
            if config.get("resume", False):
                raise ValueError("episode_indices and resume cannot be used together")
            episode_subset = [dataset.episodes[x] for x in config.episode_indices]
            dataset = CollaborationDatasetV0(
                config=config.habitat.dataset, episodes=episode_subset
            )
        run_planner(config, dataset)
    else:
        # Process episodes in parallel
        mp_ctx = mp.get_context("forkserver")
        proc_infos = []
        config.num_proc = min(config.num_proc, num_episodes)
        ochunk_size = num_episodes // config.num_proc
        # Prepare chunked datasets
        chunked_datasets = []
        # TODO: we may want to chunk by scene
        start = 0
        for i in range(config.num_proc):
            chunk_size = ochunk_size
            if i < (num_episodes % config.num_proc):
                chunk_size += 1
            end = min(start + chunk_size, num_episodes)
            indices = slice(start, end)
            chunked_datasets.append(indices)
            start += chunk_size

        for episode_index_chunk in chunked_datasets:
            episode_subset = dataset.episodes[episode_index_chunk]
            new_dataset = CollaborationDatasetV0(
                config=config.habitat.dataset, episodes=episode_subset
            )

            parent_conn, child_conn = mp_ctx.Pipe()
            proc_args = (config, new_dataset, child_conn)
            p = mp_ctx.Process(target=run_planner, args=proc_args)
            p.start()
            proc_infos.append((parent_conn, p))
            print("START PROCESS")

        # Get back info
        all_stats_episodes: Dict[str, Dict] = {
            str(i): {} for i in range(config.num_runs_per_episode)
        }
        for conn, proc in proc_infos:
            stats_episodes = conn.recv()
            for run_id, stats_run in stats_episodes.items():
                all_stats_episodes[str(run_id)].update(stats_run)
            proc.join()

        all_metrics = aggregate_measures(
            {run_id: aggregate_measures(v) for run_id, v in all_stats_episodes.items()}
        )
        cprint("\n---------------------------------", "blue")
        cprint("Metrics Across All Runs:", "blue")
        for k, v in all_metrics.items():
            cprint(f"{k}: {v:.3f}", "blue")
        cprint("\n---------------------------------", "blue")
        # Write aggregated results across experiment
        write_to_csv(config.paths.end_result_file_path, all_metrics)

    e_t = time.time() - t0
    print(f"Time elapsed since start of experiment: {e_t} seconds.")


def run_planner(config, dataset: CollaborationDatasetV0 = None, conn=None):
    if config == None:
        cprint("Failed to setup config. Exiting", "red")
        return

    game_orchestrator = None
    game_instruction_override = None

    # Setup interface with the simulator if the planner depends on it
    if config.env == "habitat":
        # Remove sensors if we are not saving video

        # TODO: have a flag for this, or some check
        keep_rgb = False
        if "use_rgb" in config.evaluation:
            keep_rgb = config.evaluation.use_rgb
        if not config.evaluation.save_video and not keep_rgb:
            remove_visual_sensors(config)

        # TODO: Can we move this inside the EnvironmentInterface?
        # We register the dynamic habitat sensors
        register_sensors(config)
        # We register custom actions
        register_actions(config)
        # We register custom measures
        register_measures(config)

        # Initialize the environment interface for the agent
        env_interface = EnvironmentInterface(config, dataset=dataset, init_wg=False)

        try:
            env_interface.initialize_perception_and_world_graph()
        except Exception:
            print("Error initializing the environment")
            if config.evaluation.log_data:
                save_exception_message(config, env_interface)

        # Print initial world graph for emtom runs (similar to skill_runner)
        try:
            from habitat_llm.utils.world_graph import (
                print_all_entities,
                print_furniture_entity_handles,
                print_object_entity_handles,
            )

            print("\n[EMTOM] Initial world graph entities:")
            print_all_entities(env_interface.perception.gt_graph)
            print_furniture_entity_handles(env_interface.perception.gt_graph)
            print_object_entity_handles(env_interface.perception.gt_graph)
        except Exception as exc:
            print(f"[EMTOM] Failed to print initial world graph: {exc}")

        # Optional game layer
        if hasattr(config, "game") and config.game.enable:
            adapter = HabitatEnvironmentAdapter(env_interface)
            if config.game.type == "bomb_game":
                spec = BombGameSpec(
                    max_defuse_attempts=config.game.max_defuse_attempts,
                    auto_defuse_on_enter=config.game.auto_defuse_on_enter,
                )
            elif config.game.type == "time_game":
                spec = TimeGameSpec(
                    max_submissions=getattr(config.game, "max_submissions", 3),
                )
            else:
                raise ValueError(f"Unknown game type {config.game.type}")
            game_orchestrator = GameOrchestrator(spec, adapter)
            # Optional global turn limit across all game types.
            game_orchestrator.turn_limit = getattr(config.game, "turn_limit", None)
            if config.game.instruction_override:
                game_instruction_override = config.game.instruction_override
    else:
        env_interface = None

    # Optional manual control per agent
    manual_agents = getattr(config.game, "manual_agents", [])
    if manual_agents:
        for aid in manual_agents:
            agent_key = f"agent_{aid}"
            if hasattr(config.evaluation.agents, agent_key):
                # Preserve any existing LLM config so manual planner can still use it.
                llm_conf = None
                try:
                    llm_conf = config.evaluation.agents[agent_key].planner.plan_config.llm
                except Exception:
                    pass
                config.evaluation.agents[agent_key].planner = OmegaConf.create(
                    {
                        "_target_": "habitat_llm.planner.manual_planner.ManualPlanner",
                        "_partial_": True,
                        "_recursive_": False,
                        "plan_config": {
                            "manual_obs_popup": config.game.manual_obs_popup,
                            "llm": llm_conf,
                        },
                    }
                )

    # Instantiate the agent planner
    eval_runner: EvaluationRunner = None
    if config.evaluation.type == "centralized":
        eval_runner = CentralizedEvaluationRunner(config.evaluation, env_interface)
    elif config.evaluation.type == "decentralized":
        if game_orchestrator:
            eval_runner = GameDecentralizedEvaluationRunner(
                config.evaluation,
                env_interface,
                game_orchestrator,
                base_instruction=game_instruction_override,
            )
        else:
            eval_runner = DecentralizedEvaluationRunner(
                config.evaluation, env_interface
            )
    else:
        cprint(
            "Invalid planner type. Please select between 'centralized' or 'decentralized'. Exiting",
            "red",
        )
        return

    # Pretty-print win condition/context at start for games
    if game_orchestrator:
        # Ensure game state exists
        if game_orchestrator.state is None:
            agent_ids = [str(agent.uid) for agent in eval_runner.agents.values()]
            try:
                game_orchestrator.start(agent_ids)
            except Exception:
                pass
        if game_orchestrator.state:
            banner = "=" * 60
            cprint("\n" + banner, "green")
            cprint("GAME START: Review hidden info and win condition.", "green")
            # Use game-spec debug summary if available
            try:
                summary_lines = game_orchestrator.game_spec.debug_summary(
                    game_orchestrator.state, game_orchestrator.env
                )
            except Exception:
                summary_lines = []
            for line in summary_lines or []:
                cprint(line, "yellow")
            # Print full world graph for context
            try:
                wg = env_interface.full_world_graph
                world_desc = wg.get_world_descr(is_human_wg=False)
                cprint("Full world graph:", "cyan")
                print(world_desc)
            except Exception:
                pass
            cprint(banner + "\n", "green")

    # Print the planner
    cprint(f"Successfully constructed the '{config.evaluation.type}' planner!", "green")
    print(eval_runner)

    # Declare observability mode
    cprint(
        f"Partial observability is set to: '{config.world_model.partial_obs}'", "green"
    )

    # Print the agent list
    print("\nAgent List:")
    print(eval_runner.agent_list)

    # Print the agent description
    print("\nAgent Description:")
    print(eval_runner.agent_descriptions)

    # Highlight the mode of operation
    cprint("\n---------------------------------------", "blue")
    cprint(f"Planner Mode: {config.evaluation.type.capitalize()}", "blue")
    # cprint(f"LLM model: {config.planner.llm.llm._target_}", "blue")
    cprint(f"Partial Observability: {config.world_model.partial_obs}", "blue")
    cprint("---------------------------------------\n", "blue")

    os.makedirs(config.paths.results_dir, exist_ok=True)

    def compose_instruction(base_instruction: str) -> str:
        if hasattr(config, "game"):
            prefix = getattr(config.game, "instruction_prefix", "")
            override = getattr(config.game, "instruction_override", "")
            if override:
                base_instruction = override
            if prefix:
                base_instruction = prefix + " " + base_instruction
        return base_instruction

    # Run the planner
    if config.mode == "cli":
        instruction = "Go to the bed" if not config.instruction else config.instruction
        instruction = compose_instruction(instruction)

        cprint(f'\nExecuting instruction: "{instruction}"', "blue")
        try:
            info = eval_runner.run_instruction(instruction)
        except Exception as e:
            # Temporary debug: print full traceback to diagnose failures (manual CLI, etc.)
            traceback.print_exc()
            print("An error occurred:", e)

    else:
        stats_episodes: Dict[str, Dict] = {
            str(i): {} for i in range(config.num_runs_per_episode)
        }

        num_episodes = len(env_interface.env.episodes)
        for run_id in range(config.num_runs_per_episode):
            for _ in range(num_episodes):
                # Get episode id
                episode_id = env_interface.env.env.env._env.current_episode.episode_id

                # Get instruction
                instruction = env_interface.env.env.env._env.current_episode.instruction
                instruction = compose_instruction(instruction)
                print("\n\nEpisode", episode_id)

                try:
                    info = eval_runner.run_instruction(
                        output_name=f"episode_{episode_id}_{run_id}"
                    )

                    info_episode = {
                        "run_id": run_id,
                        "episode_id": episode_id,
                        "instruction": instruction,
                    }
                    stats_keys = {
                        "task_percent_complete",
                        "task_state_success",
                        "sim_step_count",
                        "replanning_count",
                        "runtime",
                    }

                    # add replanning counts to stats_keys as scalars if replanning_count is a dict
                    if "replanning_count" in info and isinstance(
                        info["replanning_count"], dict
                    ):
                        for agent_id, replan_count in info["replanning_count"].items():
                            stats_keys.add(f"replanning_count_{agent_id}")
                            info[f"replanning_count_{agent_id}"] = replan_count

                    stats_episode = extract_scalars_from_info(
                        info, ignore_keys=info.keys() - stats_keys
                    )
                    stats_episodes[str(run_id)][episode_id] = stats_episode

                    cprint("\n---------------------------------", "blue")
                    cprint(f"Metrics For Run {run_id} Episode {episode_id}:", "blue")
                    for k, v in stats_episodes[str(run_id)][episode_id].items():
                        cprint(f"{k}: {v:.3f}", "blue")
                    cprint("\n---------------------------------", "blue")
                    # Log results onto a CSV
                    epi_metrics = stats_episodes[str(run_id)][episode_id] | info_episode
                    if config.evaluation.log_data:
                        save_success_message(config, env_interface, stats_episode)
                    write_to_csv(config.paths.epi_result_file_path, epi_metrics)
                except Exception as e:
                    # print exception and trace
                    traceback.print_exc()
                    print("An error occurred while running the episode:", e)
                    print(f"Skipping evaluating episode: {episode_id}")
                    if config.evaluation.log_data:
                        save_exception_message(config, env_interface)

                try:
                    # Reset env_interface (moves onto the next episode in the dataset)
                    env_interface.reset_environment()
                except Exception as e:
                    # print exception and trace
                    traceback.print_exc()
                    print("An error occurred while resetting the env_interface:", e)
                    print("Skipping evaluating episode.")
                    if config.evaluation.log_data:
                        save_exception_message(config, env_interface)

                # Reset evaluation runner
                eval_runner.reset()

            # aggregate metrics across the current run.
            run_metrics = aggregate_measures(stats_episodes[str(run_id)])
            cprint("\n---------------------------------", "blue")
            cprint(f"Metrics For Run {run_id}:", "blue")
            for k, v in run_metrics.items():
                cprint(f"{k}: {v:.3f}", "blue")
            cprint("\n---------------------------------", "blue")

            # Write aggregated results across run
            write_to_csv(config.paths.run_result_file_path, run_metrics)

        # aggregate metrics across all runs.
        if conn is None:
            all_metrics = aggregate_measures(
                {run_id: aggregate_measures(v) for run_id, v in stats_episodes.items()}
            )
            cprint("\n---------------------------------", "blue")
            cprint("Metrics Across All Runs:", "blue")
            for k, v in all_metrics.items():
                cprint(f"{k}: {v:.3f}", "blue")
            cprint("\n---------------------------------", "blue")
            # Write aggregated results across experiment
            write_to_csv(config.paths.end_result_file_path, all_metrics)
        else:
            conn.send(stats_episodes)

    env_interface.env.close()
    del env_interface

    if conn is not None:
        # Potentially we may want to send something

        conn.close()


if __name__ == "__main__":
    cprint(
        "\nStart of the example program to demonstrate multi-agent planner demo.",
        "blue",
    )

    if len(sys.argv) < 2:
        cprint("Error: Configuration file path is required.", "red")
        sys.exit(1)

    # Run planner
    run_eval()

    cprint(
        "\nEnd of the example program to demonstrate multi-agent planner demo.",
        "blue",
    )
