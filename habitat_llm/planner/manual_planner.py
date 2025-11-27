from __future__ import annotations

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
        available = sorted(list(self.agents[0].tools.keys()))
        prompt = (
            f"Agent_{agent_uid} manual action. Available tools: {available}\n"
            "Enter Tool[arg1, arg2] (or blank for Wait, or 'done' to finish): "
        )
        raw = input(prompt).strip()
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
        return name.strip(), args, None

    def get_next_action(
        self,
        instruction: str,
        observations: Dict[str, Any],
        world_graph: Dict[int, Any],
        verbose: bool = False,
    ):
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

        info = {
            "high_level_actions": hl_actions,
            "responses": responses,
            "replanned": {agent.uid: True for agent in self.agents},
            "replan_required": {agent.uid: False for agent in self.agents},
            "is_done": {agent.uid: self.is_done for agent in self.agents},
        }
        return low_level_actions, info, self.is_done
