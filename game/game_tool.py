from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from habitat_llm.tools import Tool


class GameTool(Tool):
    """
    Generic Tool wrapper around game tool descriptors so planners can invoke
    game-specific actions without modifying agent configs. Returns zero-vector
    low-level actions like WaitSkill to properly integrate with video recording.
    """

    def __init__(self, descriptor, orchestrator, agent_uid: int, env_interface=None):
        super().__init__(descriptor.name, agent_uid_arg=agent_uid)
        self.descriptor = descriptor
        self.orchestrator = orchestrator
        self.env_interface = env_interface
        self._argument_types = ["string"] * len(descriptor.parameters.keys())

        # State for multi-step execution (like WaitSkill)
        self._steps_remaining = 0
        self._pending_response = ""
        self._action_shape = None

    def _get_action_shape(self):
        """Get the shape for zero-vector actions from the environment."""
        if self._action_shape is not None:
            return self._action_shape
        if self.env_interface is not None:
            try:
                # Get action space size from env_interface
                action_space = self.env_interface.env.action_space
                if hasattr(action_space, 'shape'):
                    self._action_shape = action_space.shape
                elif hasattr(action_space, 'n'):
                    self._action_shape = (action_space.n,)
                else:
                    # Fallback: try to get from a sample
                    sample = action_space.sample()
                    if hasattr(sample, 'shape'):
                        self._action_shape = sample.shape
                    else:
                        self._action_shape = (len(sample),) if hasattr(sample, '__len__') else (1,)
            except Exception:
                self._action_shape = (1,)
        else:
            self._action_shape = (1,)
        return self._action_shape

    @property
    def description(self) -> str:
        return self.descriptor.description

    @property
    def argument_types(self) -> List[str]:
        return self._argument_types

    def process_high_level_action(self, input_query, observations):
        """
        Process the game tool action. Returns zero-vector low-level actions
        for multiple steps (based on tool_delay) to integrate with video recording,
        similar to how WaitSkill works.
        """
        # If we're in the middle of a multi-step execution, continue
        if self._steps_remaining > 0:
            self._steps_remaining -= 1
            action_shape = self._get_action_shape()
            zero_action = np.zeros(action_shape, dtype=np.float32)
            # Return response only on the last step
            if self._steps_remaining == 0:
                response = self._pending_response
                self._pending_response = ""
                return zero_action, response
            return zero_action, ""

        # First call: execute the game tool handler
        try:
            result = self.descriptor.handler(
                agent_id=str(self.agent_uid),
                orchestrator=self.orchestrator,
                **self._parse_args(input_query),
            )
            if isinstance(result, dict):
                ok = result.get("ok", True)
                if not ok:
                    # Error case: return immediately with no action
                    return None, result.get("error", "Failed")
                response = self._render_result(result)
            else:
                response = str(result)
        except Exception as e:
            return None, f"Error executing {self.name}: {e}"

        # Calculate number of steps based on tool_delay (like WaitSkill)
        tool_delay = getattr(self.orchestrator, "tool_delay", 0.0) if self.orchestrator else 0.0
        # At 120Hz sim frequency, 2 seconds = 240 steps
        sim_freq = 120  # Default sim frequency
        num_steps = max(1, int(tool_delay * sim_freq))

        action_shape = self._get_action_shape()
        zero_action = np.zeros(action_shape, dtype=np.float32)

        if num_steps <= 1:
            # Single step: return immediately with response
            return zero_action, response
        else:
            # Multi-step: save response for last step
            self._steps_remaining = num_steps - 1
            self._pending_response = response
            return zero_action, ""

    def _parse_args(self, input_query: str) -> Dict[str, Any]:
        args = {}
        if not self.descriptor.parameters:
            return args
        parts = [p.strip() for p in input_query.split(",") if p.strip()]
        for key, val in zip(self.descriptor.parameters.keys(), parts):
            args[key] = val
        return args

    def get_state_description(self):
        # Game tools don't track dynamic state; return a simple status.
        if self._steps_remaining > 0:
            return f"Executing {self.name}..."
        return "Game tool ready"

    # --- Formatting helpers -------------------------------------------------
    def _render_result(self, result: Dict[str, Any]) -> str:
        """
        Turn a handler dict into a planner-friendly string; include structured
        payloads (items/rules/protections/code) when present so the LLM has context.
        """
        parts = []
        if "message" in result:
            parts.append(str(result["message"]))
        if "code" in result:
            # For read_secret_code: show the code that was read
            parts.append(f"Code: {result['code']}")
        if "items" in result:
            parts.append(self._format_items(result["items"]))
        if "rules" in result:
            parts.append(self._format_rules(result.get("rules", [])))
        if "protections" in result:
            parts.append(self._format_protections(result.get("protections", {})))
        if not parts:
            return "Success"
        return "\n".join([p for p in parts if p])

    def _format_items(self, items: Any) -> str:
        if not items:
            return ""
        if isinstance(items, str):
            return items
        lines = []
        for item in items:
            if not isinstance(item, dict):
                lines.append(str(item))
                continue
            name = item.get("name", "item")
            mat = item.get("material", "")
            status = item.get("status", "")
            loc = item.get("location", "")
            note = item.get("note", "")
            prot = item.get("protection", "")
            line_parts = [name]
            if mat:
                line_parts.append(f"material={mat}")
            if status:
                line_parts.append(f"status={status}")
            if loc:
                line_parts.append(f"location={loc}")
            if prot:
                line_parts.append(f"protected={prot}")
            if note:
                line_parts.append(f"note={note}")
            lines.append("; ".join(line_parts))
        return "Items:\n- " + "\n- ".join(lines)

    def _format_rules(self, rules: Any) -> str:
        if not rules:
            return ""
        if isinstance(rules, str):
            return rules
        lines = []
        for rule in rules:
            if isinstance(rule, dict):
                material = rule.get("material", "")
                decay = rule.get("expected_decay", "")
                noise = rule.get("noise", "")
                lines.append(f"{material}: decay≈{decay}, noise≈{noise}")
            else:
                lines.append(str(rule))
        return "Decay rules:\n- " + "\n- ".join(lines)

    def _format_protections(self, protections: Any) -> str:
        if not protections:
            return ""
        if isinstance(protections, str):
            return protections
        lines = []
        for name, cfg in protections.items():
            materials = cfg.get("materials", [])
            note = cfg.get("note", "")
            scale = cfg.get("scale", "")
            lines.append(
                f"{name}: materials={materials}, scale={scale}, note={note}"
            )
        return "Protections:\n- " + "\n- ".join(lines)
