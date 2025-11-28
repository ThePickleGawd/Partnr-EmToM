from __future__ import annotations

import time
from typing import Any, Dict, Tuple

from habitat_llm.planner.planner import Planner


class ManualPlanner(Planner):
    """
    Minimal planner that prompts the user on stdin for each step.
    Input format: Tool[arg1, arg2]. Use empty input for Wait.
    Type 'done' to end the episode for this agent.
    """

    def __init__(self, plan_config, env_interface) -> None:
        super().__init__(plan_config, env_interface)

    def _prompt_action(self, agent_uid: int) -> Tuple[str, str, str]:
        agent_obj = next(a for a in self.agents if a.uid == agent_uid)
        available = sorted(list(agent_obj.tools.keys()))
        header = f"\nAgent_{agent_uid} > Tools: {available}"
        prompt = f"{header}\nAgent_{agent_uid} > "
        while True:
            try:
                raw = input(prompt).strip()
            except EOFError:
                # If stdin is closed (non-interactive run), keep waiting instead of crashing.
                time.sleep(1.0)
                continue
            except KeyboardInterrupt:
                self.is_done = True
                return "Done", "", None
            break
        if raw.lower() == "done":
            self.is_done = True
            return "Done", "", None
        if raw == "":
            return "Wait", "", None
        # Expect format Tool[args]
        if "[" in raw and raw.endswith("]"):
            name, rest = raw.split("[", 1)
            args = rest[:-1]  # drop trailing ]
        else:
            name = raw
            args = ""
        name = name.strip()
        # Validate tool name
        if name not in available and name not in {"Wait", "Done"}:
            print(f"[Warning] Invalid tool '{name}'. Valid: {available}. Try again.")
            return self._prompt_action(agent_uid)
        return name, args, None

    def get_next_action(
        self,
        instruction: str,
        observations: Dict[str, Any],
        world_graph: Dict[int, Any],
        verbose: bool = False,
    ):
        if not hasattr(self, "_printed_world"):
            # Print the full world graph once for context.
            try:
                world_desc = world_graph[self.agents[0].uid].get_world_descr(
                    is_human_wg=False
                )
                print(world_desc)
            except Exception:
                pass
            self._printed_world = True

        if self.is_done:
            info = {
                "high_level_actions": {},
                "responses": {},
                "replanned": {agent.uid: False for agent in self.agents},
                "replan_required": {agent.uid: False for agent in self.agents},
                "is_done": {agent.uid: True for agent in self.agents},
            }
            return {}, info, True

        hl_actions: Dict[int, Tuple[str, str, str]] = {}
        for agent in self.agents:
            name, args, err = self._prompt_action(agent.uid)
            if name == "Done":
                self.is_done = True
                hl_actions[agent.uid] = (name, args, err)
                break
            hl_actions[agent.uid] = (name, args, err)

        low_level_actions, responses = self.process_high_level_actions(
            hl_actions, observations
        )

        # Fill empty responses to avoid downstream null issues
        for agent in self.agents:
            if agent.uid not in responses or not responses[agent.uid]:
                responses[agent.uid] = "Manual action acknowledged."
            print(f"Agent_{agent.uid} Response: {responses[agent.uid]}")

        info = {
            "high_level_actions": hl_actions,
            "responses": responses,
            "replanned": {agent.uid: True for agent in self.agents},
            "replan_required": {agent.uid: False for agent in self.agents},
            "is_done": {agent.uid: self.is_done for agent in self.agents},
        }
        return low_level_actions, info, self.is_done

    def reset(self) -> None:
        """Reset manual planner state between episodes."""
        self.is_done = False
        self._printed_world = False
        self._last_action_sig = None
