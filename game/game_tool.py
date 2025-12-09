from __future__ import annotations

import time
from typing import Any, Dict, List

from habitat_llm.tools import Tool


class GameTool(Tool):
    """
    Generic Tool wrapper around game tool descriptors so planners can invoke
    game-specific actions without modifying agent configs. Formats handler
    results into planner-friendly strings.
    """

    def __init__(self, descriptor, orchestrator, agent_uid: int):
        super().__init__(descriptor.name, agent_uid_arg=agent_uid)
        self.descriptor = descriptor
        self.orchestrator = orchestrator
        self._argument_types = ["string"] * len(descriptor.parameters.keys())

    @property
    def description(self) -> str:
        return self.descriptor.description

    @property
    def argument_types(self) -> List[str]:
        return self._argument_types

    def process_high_level_action(self, input_query, observations):
        # Apply configurable delay before tool execution (for video visibility)
        delay = getattr(self.orchestrator, "tool_delay", 0.0)
        if delay > 0:
            time.sleep(delay)
        # input_query may include arguments separated by comma; pass raw to handler
        try:
            result = self.descriptor.handler(
                agent_id=str(self.agent_uid),
                orchestrator=self.orchestrator,
                **self._parse_args(input_query),
            )
            if isinstance(result, dict):
                ok = result.get("ok", True)
                if not ok:
                    return None, result.get("error", "Failed")
                return None, self._render_result(result)
            return None, str(result)
        except Exception as e:
            return None, f"Error executing {self.name}: {e}"

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
        return "Game tool ready"

    # --- Formatting helpers -------------------------------------------------
    def _render_result(self, result: Dict[str, Any]) -> str:
        """
        Turn a handler dict into a planner-friendly string; include structured
        payloads (items/rules/protections) when present so the LLM has context.
        """
        parts = []
        if "message" in result:
            parts.append(str(result["message"]))
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

