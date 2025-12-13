"""
LLM-guided curiosity model for EMTOM exploration.

Selects actions based on novelty and desire to understand the world,
rather than task-directed behavior.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


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
    """

    PROMPT_TEMPLATE = """You are an explorer in a simulated home environment. Your goal is to discover how things work, especially any unusual or unexpected behaviors.

=== MOTOR SKILLS (require navigation first) ===
- Navigate[target]: Move to a room or furniture. You MUST navigate close to objects before interacting.
- Pick[object]: Pick up an object. Must be near it first (Navigate to get close).
- Open[furniture]: Open articulated furniture (cabinets, drawers, fridges). Must be near it first.
- Close[furniture]: Close articulated furniture. Must be near it first.
- Explore[room]: Thoroughly search a room by visiting all furniture in it.

=== CUSTOM ACTIONS ===
- Hide[object]: Hide an object so others can't see it. Useful for testing what other agents know.
- Inspect[object]: Carefully examine an object to learn its properties and state.
- WriteMessage[furniture]: Leave a message on a surface for other agents to find.

WORKFLOW: To interact with objects physically, FIRST navigate to them, THEN interact.
Example: To pick up "apple_1" on "counter_5":
1. Navigate[counter_5]  (get close to the counter)
2. Pick[apple_1]        (now you can pick it up)

Current world state:
{world_description}

Available actions:
{available_actions}

Your recent history:
{history}

Based on your curiosity and desire to learn about this world:
1. Which action would be most informative to try next?
2. What do you expect to happen?
3. Why is this action interesting?

Consider:
- If a Pick/Open/Close failed with "not close enough", you need to Navigate first
- Try opening cabinets and drawers to discover what's inside
- Use Inspect to learn about object properties
- Hide objects to see if mechanics cause unexpected effects
- Explore different rooms to find more interactive objects

Respond in JSON format:
{{
    "action": "<action_name>",
    "target": "<target_id>",
    "expected_outcome": "<what_you_expect_to_happen>",
    "reasoning": "<why_this_action_is_interesting>",
    "confidence": <0.0_to_1.0_how_confident_in_outcome>
}}"""

    def __init__(
        self,
        llm_client: Any,
        exploration_bonus: float = 0.3,
    ):
        """
        Initialize the curiosity model.

        Args:
            llm_client: LLM client with generate(prompt) method
            temperature: LLM sampling temperature
            exploration_bonus: Bonus for trying new actions (not yet used)
        """
        self.llm = llm_client
        self.exploration_bonus = exploration_bonus

    def select_action(
        self,
        agent_id: str,
        world_description: str,
        available_actions: List[Dict[str, Any]],
        exploration_history: Optional[List[Dict[str, Any]]] = None,
    ) -> ActionChoice:
        """
        Select an action based on curiosity.

        Args:
            agent_id: ID of the agent selecting
            world_description: Text description of current world state
            available_actions: List of available actions with targets
            exploration_history: Recent action history for context

        Returns:
            ActionChoice with selected action and reasoning
        """
        prompt = self.PROMPT_TEMPLATE.format(
            world_description=world_description,
            available_actions=self._format_actions(available_actions),
            history=self._format_history(exploration_history or []),
        )

        response = self.llm.generate(
            prompt,
        )

        return self._parse_response(response, available_actions)

    def _format_actions(self, actions: List[Dict[str, Any]]) -> str:
        """Format available actions for the prompt."""
        lines = []
        for action in actions:
            action_name = action.get("name", action.get("action", "unknown"))
            targets = action.get("targets", [])
            description = action.get("description", "")

            if targets:
                target_str = f" (targets: {', '.join(targets)})"
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
                action_str += f" on {target}"

            lines.append(f"Step {step}: {action_str}")
            if observation:
                lines.append(f"  Result: {observation}")

        return "\n".join(lines)

    def _parse_response(
        self,
        response: str,
        available_actions: List[Dict[str, Any]],
    ) -> ActionChoice:
        """Parse LLM response into ActionChoice."""
        # Try to extract JSON from response
        try:
            # Find JSON in response
            json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return ActionChoice(
                    action=data.get("action", "wait"),
                    target=data.get("target"),
                    expected_outcome=data.get("expected_outcome", "unknown"),
                    reasoning=data.get("reasoning", ""),
                    confidence=float(data.get("confidence", 0.5)),
                )
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: try to extract action from text
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
                    reasoning="Extracted from LLM response",
                    confidence=0.3,
                )

        # Default to wait if nothing else works
        return ActionChoice(
            action="wait",
            target=None,
            expected_outcome="Nothing happens",
            reasoning="Could not parse LLM response",
            confidence=0.1,
        )


