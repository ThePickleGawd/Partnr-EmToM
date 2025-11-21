#!/usr/bin/env python3
# isort: skip_file

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
import sys
from typing import Any, Dict, List, Tuple


# append the path of the
# parent directory
sys.path.append("..")

import omegaconf
import hydra

from hydra.utils import instantiate

from habitat_llm.utils import cprint, setup_config, fix_config

from habitat_llm.tools import PerceptionTool
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


# Method to load agent planner from the config
@hydra.main(
    config_path="../conf", config_name="examples/skill_runner_default_config.yaml"
)
def run_skills(config: omegaconf.DictConfig) -> None:
    """
    The main function for executing the skill_runner tool. A default config is provided.
    See the `main` function for example CLI command to run the tool.

    :param config: input is a habitat-llm config from Hydra. Can contain CLI overrides.
    """
    fix_config(config)
    # Setup a seed
    seed = 47668090
    # Setup some hardcoded config overrides (e.g. the metadata path)
    with omegaconf.open_dict(config):
        config_dict = omegaconf.OmegaConf.create(
            omegaconf.OmegaConf.to_container(config.habitat, resolve=True)
        )
        config_dict.dataset.metadata = {"metadata_folder": "data/hssd-hab/metadata"}
        config.habitat = config_dict
    config = setup_config(config, seed)

    assert config.env == "habitat", "Only valid for Habitat skill testing."

    # whether or not to show blocking videos after each command call
    show_command_videos = (
        config.skill_runner_show_videos
        if hasattr(config, "skill_runner_show_videos")
        else True
    )
    # make videos only if showing or saving them
    make_video = config.evaluation.save_video or show_command_videos

    if not make_video:
        remove_visual_sensors(config)

    # We register the dynamic habitat sensors
    register_sensors(config)

    # We register custom actions
    register_actions(config)

    # We register custom measures
    register_measures(config)

    # create the dataset
    dataset = CollaborationDatasetV0(config.habitat.dataset)
    print(f"Loading EpisodeDataset from: {config.habitat.dataset.data_path}")
    # Initialize the environment interface for the agent
    env_interface = EnvironmentInterface(config, dataset=dataset)

    ##########################################
    # select and initialize the desired episode by index or id
    # NOTE: use "+skill_runner_episode_index=2" in CLI to set the episode index ( e.g. episode 2)
    # NOTE: use "+skill_runner_episode_id=<id>" in CLI to set the episode id ( e.g. episode "")
    assert not (
        hasattr(config, "skill_runner_episode_index")
        and hasattr(config, "skill_runner_episode_id")
    ), "Episode selection options are mutually exclusive."
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
    ###########################################

    # Initialize the planner
    planner_conf = config.evaluation.planner
    planner = instantiate(planner_conf)
    planner = planner(env_interface=env_interface)
    agent_config = config.evaluation.agents
    planner.agents = init_agents(agent_config, env_interface)
    planner.reset()

    sim = env_interface.sim

    # show the topdown map if requested
    if hasattr(config, "skill_runner_show_topdown"):
        dbv = DebugVisualizer(sim, config.paths.results_dir)
        dbv.create_dbv_agent(resolution=(1000, 1000))
        top_down_map = dbv.peek("stage")
        if show_command_videos:
            top_down_map.show()
        if config.evaluation.save_video:
            top_down_map.save(output_path=config.paths.results_dir, prefix="topdown")
        dbv.remove_dbv_agent()
        dbv.create_dbv_agent()
        dbv.remove_dbv_agent()

    ############################
    # done with setup, prompt the user and start running skills

    exit_skill = "exit"
    help_skill = "help"
    entity_skill = "entities"
    pdb_skill = "debug"
    cumulative_video_skill = "make_video"

    cprint("Welcome to skill_runner!", "green")
    cprint(
        f"Current Episode (id=={sim.ep_info.episode_id}) is running in scene {sim.ep_info.scene_id} with info: {sim.ep_info.info}.",
        "green",
    )

    print_all_entities(env_interface.perception.gt_graph)
    print_furniture_entity_handles(env_interface.perception.gt_graph)
    print_object_entity_handles(env_interface.perception.gt_graph)

    tool_metadata: Dict[str, Dict[str, Any]] = {}
    tool_to_agents: Dict[str, List[int]] = {}
    for agent in planner.agents:
        for tool_name, tool in agent.tools.items():
            tool_to_agents.setdefault(tool_name, []).append(agent.uid)
            if tool_name not in tool_metadata:
                tool_metadata[tool_name] = {
                    "description": tool.description,
                    "type": "perception"
                    if isinstance(tool, PerceptionTool)
                    else "motor",
                    "argument_types": tool.argument_types,
                }

    tool_lines = [
        f"- {name} ({meta['type']}): {meta['description']}"
        for name, meta in sorted(tool_metadata.items())
    ]
    help_text = (
        "Call tools as '[agent_id:]ToolName[arg1, arg2, ...]' or "
        "'ToolName <agent_id> <args>'. Separate multiple arguments with commas "
        "for multi-argument motor skills (e.g., Place or Rearrange). "
        "Use 'entities' to re-print entity handles.\n"
        + "\n".join(tool_lines)
        + f"\nOther commands: '{exit_skill}', '{help_skill}', '{entity_skill}', '{pdb_skill}', '{cumulative_video_skill}'."
    )
    cprint(help_text, "green")

    # setup a sequence of commands to run immediately without manual input
    scripted_commands: List[str] = []
    if hasattr(config, "skill_runner_scripted_commands"):
        scripted_commands = config.skill_runner_scripted_commands
        multi_arg_skills = {"Place", "Rearrange"}
        # we need special handling for multi-argument skills because arguments are comma separated and need to be joined
        for skill_name in multi_arg_skills:
            skill_indices = [
                i for i, x in enumerate(scripted_commands) if skill_name in x
            ]
            for i, skill_ix in enumerate(skill_indices):
                corrected_ix = skill_ix - i * 4  # account for removed elements
                for j in range(1, 5):
                    # concat the elements
                    scripted_commands[corrected_ix] += (
                        "," + scripted_commands[corrected_ix + j]
                    )
                scripted_commands = (
                    scripted_commands[: corrected_ix + 1]
                    + scripted_commands[corrected_ix + 5 :]
                )
    print(scripted_commands)

    # collect debug frames to create a final video
    cumulative_frames: List[Any] = []

    command_index = 0
    # history of skill commands and their responses
    command_history: List[Tuple[str, str]] = []
    while True:
        cprint("Enter Command", "blue")
        if len(scripted_commands) > command_index:
            user_input = scripted_commands[command_index]
            print(user_input)
        else:
            user_input = input("> ")

        selected_skill = None

        if user_input == exit_skill:
            print("==========================")
            print("Exiting. Command History:")
            for ix, t in enumerate(command_history):
                print(f" [{ix}]: '{t[0]}' -> '{t[1]}'")
            print("==========================")
            exit()
        elif user_input == help_skill:
            cprint(help_text, "green")
        elif user_input == entity_skill:
            print_all_entities(env_interface.perception.gt_graph)
        elif user_input == pdb_skill:
            # peek an entity
            dbv = DebugVisualizer(sim, config.paths.results_dir)
            dbv.create_dbv_agent()
            # NOTE: do debugging calls here
            # example to peek an entity: dbv.peek(env_interface.world_graph.get_node_from_name('table_50').sim_handle).show()
            breakpoint()
            dbv.remove_dbv_agent()
        elif user_input == cumulative_video_skill:
            # create a video of all accumulated frames thus far and play it
            if len(cumulative_frames) > 0:
                dvu = DebugVideoUtil(
                    env_interface, env_interface.conf.paths.results_dir
                )
                dvu.frames = cumulative_frames
                dvu._make_video(postfix="cumulative", play=show_command_videos)
        else:
            raw_command = user_input.strip()
            agent_hint = None
            agent_match = re.match(r"^(?P<agent>[0-9]+)[:]\s*(?P<body>.+)$", raw_command)
            if agent_match:
                agent_hint = agent_match.group("agent")
                raw_command = agent_match.group("body").strip()

            bracket_match = re.match(
                r"^(?P<tool>[A-Za-z0-9_]+)\s*\[(?P<args>.*)\]\s*$", raw_command
            )
            if bracket_match:
                selected_skill = bracket_match.group("tool")
                agent_ix = agent_hint
                target_entity_name = bracket_match.group("args").strip()
            else:
                skill_components = raw_command.split(" ", 1)
                selected_skill = skill_components[0]
                agent_ix = agent_hint
                target_entity_name = ""
                if len(skill_components) > 1:
                    target_entity_name = skill_components[1].strip()
            if selected_skill not in tool_metadata:
                selected_skill = None
                cprint("... invalid command.", "red")
                command_index += 1
                continue

            valid_agents = tool_to_agents.get(selected_skill, [])
            scripted_mode = len(scripted_commands) > command_index
            if agent_ix is None:
                if len(valid_agents) == 1:
                    agent_ix = str(valid_agents[0])
                elif scripted_mode:
                    fallback_agent = (
                        env_interface.robot_agent_uid
                        if env_interface.robot_agent_uid in valid_agents
                        else valid_agents[0]
                    )
                    agent_ix = str(fallback_agent)
                else:
                    agent_ix = input(
                        f"Agent Index for {selected_skill} {valid_agents} (0=robot, 1=human) = "
                    )

            try:
                agent_ix_int = int(agent_ix)
            except (TypeError, ValueError):
                cprint("... invalid Agent Index, aborting.", "red")
                command_index += 1
                continue

            if agent_ix_int not in valid_agents:
                cprint(
                    f"... invalid Agent Index for {selected_skill}, choose from {valid_agents}.",
                    "red",
                )
                command_index += 1
                continue

            if target_entity_name == "" and not scripted_mode:
                target_entity_name = input("Tool argument(s) = ")

            if selected_skill in {"Place", "Rearrange"} and "," not in target_entity_name:
                tokens = target_entity_name.split()
                if len(tokens) == 5:
                    target_entity_name = ",".join(tokens)

            high_level_skill_actions = {
                int(agent_ix_int): (selected_skill, target_entity_name, None)
            }

            agent_tool = planner.get_agent_from_uid(agent_ix_int).get_tool_from_name(
                selected_skill
            )
            try:
                if isinstance(agent_tool, PerceptionTool):
                    observations = planner.env_interface.get_observations()
                    _, responses = planner.process_high_level_actions(
                        high_level_skill_actions, observations
                    )
                    command_history.append((user_input, responses[int(agent_ix_int)]))
                    print(
                        f"{selected_skill} completed. Response = '{responses[int(agent_ix_int)]}'"
                    )
                else:
                    responses, _, frames = execute_skill(
                        high_level_skill_actions,
                        planner,
                        vid_postfix=f"{command_index}_",
                        make_video=make_video,
                        play_video=show_command_videos,
                    )
                    command_history.append((user_input, responses[int(agent_ix_int)]))
                    skill_name = high_level_skill_actions[int(agent_ix_int)][0]
                    print(
                        f"{skill_name} completed. Response = '{responses[int(agent_ix_int)]}'"
                    )
                    cumulative_frames.extend(frames)
            except Exception as e:
                failure_string = f"Failed to execute skill with exception: {str(e)}"
                print(failure_string)
                command_history.append((user_input, failure_string))
        command_index += 1


##########################################
# CLI Example:
# HYDRA_FULL_ERROR=1 python -m habitat_llm.examples.skill_runner hydra.run.dir="."
# or
# python habitat_llm/examples/skill_runner.py
#
# NOTE: conf/examples/skill_runner_default_config.yaml is consumed to initialize parameters
##########################################
# Script Specific CLI overrides:
#
# (mutually exclusive)
# - '+skill_runner_episode_index=0' - initialize the episode with the specified index within the dataset
# - '+skill_runner_episode_id=' - initialize the episode with the specified "id" within the dataset
#
# - '+skill_runner_show_topdown=True' - (default False) show a topdown view of the scene upon initialization for context
#
# (output control options)
# - '+skill_runner_show_videos=False' - (default True) turn off showing videos immediately after running a command
# - 'evaluation.save_video=False' - (default True) option to save videos to files. Also affects cumulative videos produced with "make_video" command.
# NOTE: videos are made only if either of the above options are True
# - 'paths.results_dir=<relative_path>' (default './results/') relative path to desired output directory for evaluation
#
##########################################
# Other useful CLI overrides:
#
# - 'habitat.dataset.data_path="<path to dataset .json.gz>"' - set the desired episode dataset
#
if __name__ == "__main__":
    cprint(
        "\nStart of the example program to run custom skill commands in a CollaborationEpisode.",
        "blue",
    )

    # Run the skills
    run_skills()

    cprint(
        "\nEnd of the example program to run custom skill commands in a CollaborationEpisode.",
        "blue",
    )
