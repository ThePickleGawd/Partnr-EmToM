"""
Game-aware evaluation runner wrapper. Reuses the standard decentralized runner loop
but injects game context into the instruction and monitors game termination.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, Optional

from habitat_llm.evaluation.decentralized_evaluation_runner import (
    DecentralizedEvaluationRunner,
)
from habitat_llm.utils import rollout_print
from game.game_tool import GameTool


class GameDecentralizedEvaluationRunner(DecentralizedEvaluationRunner):
    def __init__(
        self,
        evaluation_runner_config_arg,
        env_arg,
        game_orchestrator,
        base_instruction: Optional[str] = None,
    ):
        self.game_orchestrator = game_orchestrator
        self.base_instruction = base_instruction
        super().__init__(evaluation_runner_config_arg, env_arg)

    def _inject_game_tools(self) -> None:
        """
        Attach game tools to each agent's tool registry so planners can call them.
        """
        if not self.game_orchestrator:
            return
        state = self.game_orchestrator.state
        if state is None:
            return
        for agent in self.agents.values():
            descs = self.game_orchestrator.game_spec.get_tools_for_agent(
                str(agent.uid), state, self.game_orchestrator.env
            )
            added = []
            for desc in descs:
                if desc.name in agent.tools:
                    continue
                tool = GameTool(desc, self.game_orchestrator, agent.uid)
                agent.tools[tool.name] = tool
                added.append(desc.name)
            if added:
                print(
                    f"[BombGame] Added game tools for agent {agent.uid}: {added}. "
                    f"Now available: {sorted(agent.tools.keys())}"
                )

    def _compose_instruction(self, fallback_instruction: str) -> str:
        """
        Merge game public context and role info with the base task instruction.
        """
        if not self.game_orchestrator or not self.game_orchestrator.state:
            return fallback_instruction

        state = self.game_orchestrator.state
        public = self.game_orchestrator.game_spec.render_public_context(state)
        roles_overview = ", ".join(
            [f"{aid}:{role.name}" for aid, role in state.agent_roles.items()]
        )

        # Add per-agent private info and allowed tools (if specified in the game config).
        per_agent_blocks = []
        allowed_tools = getattr(self.game_orchestrator.game_spec, "allowed_tools", None)
        for aid, role in state.agent_roles.items():
            private = self.game_orchestrator.game_spec.render_private_context(aid, state)
            tool_line = (
                f"Allowed tools: {sorted(list(allowed_tools))}"
                if allowed_tools
                else "Use the provided game tools."
            )
            block = f"Agent {aid} ({role.name}): {private if private else 'No private info.'} {tool_line}"
            per_agent_blocks.append(block)

        per_agent_text = "\n".join(per_agent_blocks)

        image_note = ""
        if state.secret_state.get("latest_image_path"):
            image_note = f"\nImage available: {state.secret_state.get('latest_image_path')}"

        return (
            f"{public}\nRoles: {roles_overview}\n{per_agent_text}{image_note}\nTask: {fallback_instruction}"
        )

    def _maybe_update_game(self) -> None:
        if not self.game_orchestrator or not self.game_orchestrator.state:
            return
        spec = self.game_orchestrator.game_spec
        if hasattr(spec, "maybe_auto_resolve"):
            spec.maybe_auto_resolve(self.game_orchestrator.state, self.game_orchestrator.env)
        terminal, outcome = spec.check_end_condition(
            self.game_orchestrator.state, self.game_orchestrator.env
        )
        self.game_orchestrator.state.terminal = terminal
        self.game_orchestrator.state.outcome = outcome

    def run_instruction(
        self, instruction: Optional[str] = None, output_name: str = ""
    ) -> Dict[str, Any]:
        """
        Mirrors EvaluationRunner.run_instruction with game-state awareness.
        """
        # Derive base instruction
        base_instruction = (
            instruction
            if instruction is not None
            else self.base_instruction
            if self.base_instruction is not None
            else None
        )
        if base_instruction is None:
            # fallback to episode instruction
            curr_env = self.env_interface.env.env.env._env
            base_instruction = curr_env.current_episode.instruction

        # Start the game if it hasn't been started yet
        if self.game_orchestrator and self.game_orchestrator.state is None:
            agent_ids = [str(agent.uid) for agent in self.agents.values()]
            self.game_orchestrator.start(agent_ids)
            self._inject_game_tools()
            print("Game + PARTNR tools per agent:")
            for agent in self.agents.values():
                tool_names = sorted(list(agent.tools.keys()))
                print(f"Agent {agent.uid}: {tool_names}")

        composed_instruction = self._compose_instruction(base_instruction)

        # Rest of loop follows EvaluationRunner.run_instruction
        t_0 = time.time()
        total_step_count = 1
        self.reset_planners()
        self.initialize_instruction_metadata(composed_instruction, output_name)
        observations = self.env_interface.get_observations()
        info = {
            "task_percent_complete": 0.0,
            "task_state_success": 0.0,
            "total_step_count": total_step_count,
            "num_steps": 0.0,
        }
        planner_infos = []
        planner_info: Dict[str, Any] = {}
        low_level_actions: Any = []
        should_end = False

        while not should_end:
            if (
                "print" in planner_info
                and len(planner_info["print"])
                and self.evaluation_runner_config.do_print
            ):
                rollout_print(planner_info["print"])

            if len(low_level_actions) > 0:
                obs, reward, done, info = self.env_interface.step(low_level_actions)
                observations = self.env_interface.parse_observations(obs)
                if self.evaluation_runner_config.save_video:
                    self.dvu._store_for_video(
                        observations,
                        planner_info.get("high_level_actions", {}),
                        popup_images=planner_info.get("popup_images", {}),
                    )

            # Update game state based on latest env situation
            self._maybe_update_game()
            # Optional global game turn limit; only count iterations where planners produced actions/info.
            if (
                self.game_orchestrator
                and self.game_orchestrator.state
                and self.game_orchestrator.turn_limit is not None
            ):
                if self.game_orchestrator.should_count_turn(planner_info, low_level_actions):
                    allowed = self.game_orchestrator.increment_turn()
                    if not allowed:
                        should_end = True

            # refresh game tools in case availability changed (e.g., entering bomb room)
            self._inject_game_tools()

            # Get next low level actions
            low_level_actions, planner_info, should_end = self.get_low_level_actions(
                composed_instruction, observations, self.env_interface.world_graph
            )

            # Prepare popup images for subsequent video overlay
            if (
                self.game_orchestrator
                and self.game_orchestrator.state
                and self.game_orchestrator.state.secret_state.get("latest_image_path")
            ):
                img_path = self.game_orchestrator.state.secret_state.get(
                    "latest_image_path"
                )
                planner_info["popup_images"] = {
                    agent.uid: img_path for agent in self.agents.values()
                }
            else:
                planner_info["popup_images"] = {}

        # Allow game termination to short-circuit the loop
            if self.game_orchestrator and self.game_orchestrator.state:
                if self.game_orchestrator.state.terminal and not should_end:
                    print(
                        f"[BombGame] Game ended with outcome: {self.game_orchestrator.state.outcome}"
                    )
                should_end = should_end or self.game_orchestrator.state.terminal

            curr_env = self.env_interface.env.env.env._env
            if total_step_count > curr_env._max_episode_steps:
                should_end = True

            measure_names = [
                "auto_eval_proposition_tracker",
                "task_constraint_validation",
                "task_percent_complete",
                "task_state_success",
                "task_evaluation_log",
                "task_explanation",
            ]
            measures_to_log = [
                "task_percent_complete",
                "task_state_success",
                "task_explanation",
            ]
            if should_end:
                measures = curr_env.task.measurements.measures
                for measure_name in measure_names:
                    measures[measure_name].update_metric(
                        task=curr_env.task, episode=curr_env.current_episode
                    )
                for measure_name in measure_names:
                    if measure_name in info:
                        info[measure_name] = measures[measure_name].get_metric()

            planner_info["stats"] = {
                info_name: info[info_name]
                for info_name in measures_to_log
                if info_name in info
            }
            planner_info["total_step_count"] = total_step_count
            planner_info["sim_step_count"] = info["num_steps"]

            if (
                "replan_required" in planner_info
                and planner_info["replan_required"]
                and any(planner_info["replan_required"].values())
            ) or should_end:
                planner_info["curr_graph"] = {
                    agent_id: self.env_interface.world_graph[agent_id].get_world_descr(
                        is_human_wg=int(agent_id) == self.env_interface.human_agent_uid
                    )
                    for agent_id in range(len(self.agents))
                }

            # Prepare popup images for video overlay downstream
            if (
                self.game_orchestrator
                and self.game_orchestrator.state
                and self.game_orchestrator.state.secret_state.get("latest_image_path")
            ):
                img_path = self.game_orchestrator.state.secret_state.get(
                    "latest_image_path"
                )
                planner_info["popup_images"] = {
                    agent.uid: img_path for agent in self.agents.values()
                }
            else:
                planner_info["popup_images"] = {}

            copy_planner_info = copy.deepcopy(planner_info)
            self.update_agent_state_history(copy_planner_info)
            self.update_agent_action_history(copy_planner_info)
            planner_infos.append(copy_planner_info)
            total_step_count += 1

        if (
            "print" in planner_info
            and len(planner_info["print"])
            and self.evaluation_runner_config.do_print
        ):
            rollout_print(planner_info["print"])

        if self.evaluation_runner_config.save_video:
            self.dvu._make_video(play=False, postfix=self.episode_filename)

        self._log_planner_data(planner_infos)

        t_runtime = time.time() - t_0
        info["runtime"] = t_runtime
        info |= planner_info
        if self.game_orchestrator and self.game_orchestrator.state:
            info["game_outcome"] = self.game_orchestrator.state.outcome
            # Emit a summary line on completion.
            if self.game_orchestrator.state.terminal:
                print(
                    f"[BombGame] Final outcome: {self.game_orchestrator.state.outcome}"
                )
        return info
