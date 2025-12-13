"""
LLM-guided curiosity model for EMTOM exploration.

Selects actions based on novelty and desire to understand the world,
rather than task-directed behavior.

Uses YAML config for prompts (like the benchmark) for consistency and scalability.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from omegaconf import OmegaConf


@dataclass
class ActionChoice:
    """Result of curiosity-driven action selection."""

    action: str
    target: Optional[str]
    expected_outcome: str
    reasoning: str
    confidence: float = 0.5


class CuriosityModel:
    """
    LLM-guided action selection based on novelty and exploration.

    Uses an LLM to select actions that are likely to reveal new information
    about how the world works, particularly to discover unexpected behaviors.

    Now uses YAML config for prompts (matching benchmark structure) for consistency.
    """

    # Default config path relative to habitat_llm/conf/instruct/
    DEFAULT_CONFIG = "emtom_exploration"

    def __init__(
        self,
        llm_client: Any,
        instruct_config: Optional[Any] = None,
        llm_config: Optional[Any] = None,
        exploration_bonus: float = 0.3,
    ):
        """
        Initialize the curiosity model.

        Args:
            llm_client: LLM client with generate(prompt) method
            instruct_config: Optional instruct config (OmegaConf). If None, loads default.
            llm_config: Optional LLM config with system_tag, user_tag, etc.
            exploration_bonus: Bonus for trying new actions (not yet used)
        """
        self.llm = llm_client
        self.exploration_bonus = exploration_bonus

        # Load instruct config if not provided
        if instruct_config is not None:
            self.instruct = instruct_config
        else:
            self.instruct = self._load_default_config()

        # Extract prompt template and other settings
        self.prompt_template = self.instruct.prompt
        self.stopword = self.instruct.get("stopword", "Assigned!")
        self.end_expression = self.instruct.get("end_expression", "Final Thought:")

        # LLM config for tags (defaults if not provided)
        self.llm_config = llm_config or OmegaConf.create({
            "system_tag": "",
            "user_tag": "",
            "assistant_tag": "",
            "eot_tag": "",
        })

    def _load_default_config(self) -> Any:
        """Load the default exploration YAML config."""
        # Find the config file
        config_path = Path(__file__).parent.parent.parent / "habitat_llm" / "conf" / "instruct" / f"{self.DEFAULT_CONFIG}.yaml"

        if not config_path.exists():
            raise FileNotFoundError(
                f"Exploration config not found at {config_path}. "
                "Please ensure emtom_exploration.yaml exists."
            )

        return OmegaConf.load(config_path)

    def set_tool_descriptions(self, tool_descriptions: str):
        """
        Set the tool descriptions to use in prompts.

        Args:
            tool_descriptions: Formatted string of tool descriptions from agent
        """
        self.tool_descriptions = tool_descriptions

    def select_action(
        self,
        agent_id: str,
        world_description: str,
        available_actions: List[Dict[str, Any]],
        exploration_history: Optional[List[Dict[str, Any]]] = None,
        tool_descriptions: Optional[str] = None,
    ) -> ActionChoice:
        """
        Select an action based on curiosity.

        Args:
            agent_id: ID of the agent selecting
            world_description: Text description of current world state
            available_actions: List of available actions with targets
            exploration_history: Recent action history for context
            tool_descriptions: Optional tool descriptions (uses stored if not provided)

        Returns:
            ActionChoice with selected action and reasoning
        """
        # Use provided tool descriptions or stored ones
        tools_desc = tool_descriptions or getattr(self, 'tool_descriptions', self._get_default_tool_descriptions())

        # Build the prompt using the YAML template
        prompt = self._build_prompt(
            agent_id=agent_id,
            world_description=world_description,
            available_actions=available_actions,
            exploration_history=exploration_history or [],
            tool_descriptions=tools_desc,
        )

        response = self.llm.generate(prompt, self.stopword)

        return self._parse_response(response, available_actions)

    def _build_prompt(
        self,
        agent_id: str,
        world_description: str,
        available_actions: List[Dict[str, Any]],
        exploration_history: List[Dict[str, Any]],
        tool_descriptions: str,
    ) -> str:
        """Build the prompt using the YAML template."""
        # Format placeholders
        params = {
            "id": agent_id,
            "tool_descriptions": tool_descriptions,
            "world_description": world_description,
            "available_actions": self._format_actions(available_actions),
            "history": self._format_history(exploration_history),
            "system_tag": self.llm_config.get("system_tag", ""),
            "user_tag": self.llm_config.get("user_tag", ""),
            "assistant_tag": self.llm_config.get("assistant_tag", ""),
            "eot_tag": self.llm_config.get("eot_tag", ""),
        }

        return self.prompt_template.format(**params)

    def _get_default_tool_descriptions(self) -> str:
        """Get default tool descriptions (fallback if none provided)."""
        return """=== MOTOR SKILLS (require navigation first) ===
- Navigate[target]: Move to a room or furniture. You MUST navigate close to objects before interacting.
- Pick[object]: Pick up an object. Must be near it first (Navigate to get close).
- Open[furniture]: Open articulated furniture (cabinets, drawers, fridges). Must be near it first.
- Close[furniture]: Close articulated furniture. Must be near it first.
- Explore[room]: Thoroughly search a room by visiting all furniture in it.

=== PERCEPTION TOOLS ===
- FindObjectTool[query]: Search for objects matching a description.
- FindReceptacleTool[query]: Search for furniture/receptacles matching a description.
- FindRoomTool[query]: Search for rooms matching a description.

=== CUSTOM ACTIONS ===
- Hide[object]: Hide an object so others can't see it. Useful for testing what other agents know.
- Inspect[object]: Carefully examine an object to learn its properties and state.
- WriteMessage[furniture]: Leave a message on a surface for other agents to find."""

    def _format_actions(self, actions: List[Dict[str, Any]]) -> str:
        """Format available actions for the prompt."""
        lines = []
        for action in actions:
            action_name = action.get("name", action.get("action", "unknown"))
            targets = action.get("targets", [])
            description = action.get("description", "")

            if targets:
                # Limit targets shown to avoid overwhelming prompt
                shown_targets = targets[:5]
                if len(targets) > 5:
                    target_str = f" (targets: {', '.join(shown_targets)}, ... and {len(targets) - 5} more)"
                else:
                    target_str = f" (targets: {', '.join(shown_targets)})"
            else:
                target_str = ""

            if description:
                lines.append(f"- {action_name}{target_str}: {description}")
            else:
                lines.append(f"- {action_name}{target_str}")

        return "\n".join(lines) if lines else "No actions available"

    def _format_history(self, history: List[Dict[str, Any]]) -> str:
        """Format recent history for the prompt."""
        if not history:
            return "No recent actions"

        lines = []
        for entry in history[-5:]:  # Last 5 actions
            action = entry.get("action", "unknown")
            target = entry.get("target", "")
            observation = entry.get("observation", "")
            step = entry.get("step", "?")

            action_str = f"{action}"
            if target:
                action_str += f"[{target}]"

            lines.append(f"Step {step}: {action_str}")
            if observation:
                # Truncate long observations
                obs_short = observation[:100] + "..." if len(observation) > 100 else observation
                lines.append(f"  Result: {obs_short}")

        return "\n".join(lines)

    def _parse_response(
        self,
        response: str,
        available_actions: List[Dict[str, Any]],
    ) -> ActionChoice:
        """Parse LLM response into ActionChoice."""
        # Try to extract action from response using the same format as benchmark
        # Look for Agent_X_Action: ActionName[target]
        action_pattern = r"Agent_\d+_Action:\s*(\w+)\[([^\]]*)\]"
        match = re.search(action_pattern, response)

        if match:
            action_name = match.group(1)
            target = match.group(2) if match.group(2) else None

            # Extract reasoning from Thought: line
            thought_match = re.search(r"Thought:\s*(.+?)(?=Agent_|$)", response, re.DOTALL)
            reasoning = thought_match.group(1).strip() if thought_match else ""

            return ActionChoice(
                action=action_name,
                target=target,
                expected_outcome="Exploring the environment",
                reasoning=reasoning,
                confidence=0.7,
            )

        # Fallback: try to extract action in simple format ActionName[target]
        simple_pattern = r"(\w+)\[([^\]]*)\]"
        match = re.search(simple_pattern, response)

        if match:
            return ActionChoice(
                action=match.group(1),
                target=match.group(2) if match.group(2) else None,
                expected_outcome="unknown",
                reasoning="Extracted from LLM response",
                confidence=0.5,
            )

        # Try to find any action name in the response
        action_names = [
            a.get("name", a.get("action", ""))
            for a in available_actions
        ]
        for action_name in action_names:
            if action_name.lower() in response.lower():
                return ActionChoice(
                    action=action_name,
                    target=None,
                    expected_outcome="unknown",
                    reasoning="Extracted action name from LLM response",
                    confidence=0.3,
                )

        # Default to Explore if nothing else works
        return ActionChoice(
            action="Explore",
            target=None,
            expected_outcome="Exploring to find objects",
            reasoning="Could not parse LLM response, defaulting to exploration",
            confidence=0.1,
        )
