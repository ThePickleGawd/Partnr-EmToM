"""
TimeShift switch puzzle: Past encodes a hidden target via switch toggles; Future
decodes observed effects. Theory of mind is required because only Past knows the
target/pattern and only Future sees effects. Submission requires effects to be
present. Embodiment uses native PARTNR tools; game tools are for switches/effects.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from game.game import AgentRole, EnvironmentAdapter, GameSpec, GameState, ToolDescriptor


class TimeGameSpec(GameSpec):
    """
    Game flow:
    - Agent_0 (Past) knows the hidden target, the switch->effect codebook, and the
      target-specific effect pattern. Past must toggle switches to emit that pattern
      (cannot observe effects).
    - Agent_1 (Future) does NOT know the target, but knows the effect-pattern legend
      for all candidates. Future queries effects, decodes, explores, and submits.
    - Communication is allowed; submission is blocked until at least one effect is present.
    """

    name = "time_game"

    def __init__(
        self,
        max_submissions: int = 2,
        decay_strength: float = 0.35,  # ignored; kept for backward compatibility
    ) -> None:
        self.max_submissions = max_submissions
        self.decay_strength = decay_strength
        self.rng = random.Random()
        self._catalog = [
            "steel wrench",
            "paper map",
            "wooden crate",
            "fabric banner",
            "glass bottle",
            "battery pack",
            "rope coil",
            "kettle",
            "toy_airplane",
            "phone_stand",
            "mug",
        ]
        self._switches = ["switch_A", "switch_B", "switch_C"]
        self._effects = ["light_blue", "light_green", "light_red", "rune_1", "rune_2", "rune_3"]

    # --- GameSpec hooks --------------------------------------------------
    def initialize(
        self, agent_ids: List[str], env: EnvironmentAdapter
    ) -> GameState:
        if not agent_ids:
            raise ValueError("TimeGameSpec requires at least one agent id.")

        rooms = env.list_rooms()
        room_name = rooms[0] if rooms else "the world"
        past_id = agent_ids[0]
        roles: Dict[str, AgentRole] = {past_id: AgentRole(name="Past", private_info={})}
        for fut_id in agent_ids[1:]:
            roles[fut_id] = AgentRole(name="Future", private_info={})

        candidates = self._choose_candidates(env)
        target_item = self.rng.choice(candidates)
        codebook = self._build_codebook()
        patterns = self._build_patterns(candidates, codebook)
        # Seed the future view with the intended effect pattern so decoding is always possible.
        effect_state: List[str] = []

        roles[past_id].private_info["target_item"] = target_item
        roles[past_id].private_info["codebook"] = codebook
        roles[past_id].private_info["target_pattern"] = patterns.get(target_item, [])

        public_info = {
            "game": "time_game",
            "rooms": rooms,
            "rules": (
                "Past toggles switches; Future sees only resulting effects. "
                "Past knows the target and its effect pattern; Future knows the legend but not the target. "
                "Use PARTNR-native tools to explore/move objects; game tools handle switches/effects/submission."
            ),
        }

        secret_state = {
            "room_name": room_name,
            "target_item": target_item,
            "codebook": codebook,
            "patterns": patterns,  # item -> effect sequence
            "effect_state": effect_state,
            "target_pattern": patterns.get(target_item, []),
            "submissions_left": self.max_submissions,
            "outcome": None,
        }

        return GameState(
            agent_roles=roles,
            public_info=public_info,
            secret_state=secret_state,
        )

    def get_tools_for_agent(
        self, agent_id: str, state: GameState, env: EnvironmentAdapter
    ) -> List[ToolDescriptor]:
        role = state.agent_roles.get(agent_id)
        if role is None:
            return []

        tools: List[ToolDescriptor] = []
        if role.name == "Past":
            tools.append(
                ToolDescriptor(
                    name="toggle_switch",
                    description="Flip a named temporal switch. Effects propagate to the Future view.",
                    parameters={"switch_name": "string"},
                    handler=self._toggle_switch,
                )
            )
        if role.name == "Future":
            tools.append(
                ToolDescriptor(
                    name="query_effect_state",
                    description="Return the current effect state visible in the Future world.",
                    handler=self._query_effect_state,
                )
            )
            tools.append(
                ToolDescriptor(
                    name="submit_target_item",
                    description="Submit the item you believe was the intended target.",
                    parameters={"item_name": "string", "justification": "string"},
                    handler=self._submit_target_item,
                )
            )
        return tools

    def check_end_condition(
        self, state: GameState, env: EnvironmentAdapter
    ) -> Tuple[bool, Optional[str]]:
        outcome = state.secret_state.get("outcome")
        if outcome is not None:
            return True, outcome
        limit = state.secret_state.get("turn_limit", None)
        count = state.secret_state.get("turn_count", 0)
        if limit is not None and count >= limit:
            return True, "failure"
        if state.secret_state.get("submissions_left", 0) <= 0:
            return True, "failure"
        return False, None

    def render_public_context(self, state: GameState) -> str:
        rooms = state.public_info.get("rooms", [])
        return (
            f"Game: Time-separated puzzle. Past toggles switches; Future sees effects. "
            f"Rooms: {rooms}. Use native tools to explore/move objects; game tools handle switches/effects/submission."
        )

    def render_private_context(self, agent_id: str, state: GameState) -> str:
        role = state.agent_roles.get(agent_id)
        if not role:
            return ""
        if role.name == "Past":
            target = role.private_info.get("target_item", "")
            codebook = role.private_info.get("codebook", {})
            pattern = role.private_info.get("target_pattern", [])
            return (
                f"Hidden target: '{target}'. Codebook: {codebook}. Target pattern: {pattern}. "
                "You must emit the full pattern via toggles (no skipping); you cannot see effects. Communicate clearly."
            )
        if role.name == "Future":
            patterns = state.secret_state.get("patterns", {})
            legend = {item: seq for item, seq in patterns.items()}
            return (
                "You do NOT know the target. You know the effect patterns for candidates: "
                f"{legend}. Query effects, decode, explore, and submit via submit_target_item. "
                "Submission is only accepted when the observed effect sequence exactly matches one of the patterns."
            )
        return ""

    # --- Tool handlers ----------------------------------------------------
    def _toggle_switch(
        self, agent_id: str, orchestrator, switch_name: str = "", **_: Dict
    ) -> Dict[str, any]:
        state = orchestrator.state
        codebook = state.secret_state.get("codebook", {})
        if switch_name not in codebook:
            return {"ok": False, "error": f"Unknown switch '{switch_name}'. Valid: {list(codebook.keys())}"}
        effect = codebook[switch_name]
        effects = state.secret_state.get("effect_state", [])
        effects.append(effect)
        state.secret_state["effect_state"] = effects
        return {"ok": True, "message": f"Switch {switch_name} toggled. Current effects: {effects}"}

    def _query_effect_state(
        self, agent_id: str, orchestrator, **_: Dict
    ) -> Dict[str, any]:
        state = orchestrator.state
        effects = state.secret_state.get("effect_state", [])
        return {"ok": True, "effects": effects}

    def _submit_target_item(
        self,
        agent_id: str,
        orchestrator,
        item_name: str = "",
        justification: str = "",
        **_: Dict,
    ) -> Dict[str, any]:
        state = orchestrator.state
        effects = state.secret_state.get("effect_state", [])
        if not effects:
            return {"ok": False, "error": "No effects observed yet; ask Past to toggle switches first."}
        target_pattern = state.secret_state.get("target_pattern", [])
        if not target_pattern:
            return {"ok": False, "error": "No target pattern configured."}
        if len(effects) < len(target_pattern):
            return {"ok": False, "error": "Incomplete effect sequence; wait for full pattern."}
        if effects[: len(target_pattern)] != target_pattern:
            return {
                "ok": False,
                "error": f"Observed effects {effects[:len(target_pattern)]} do not match any valid pattern yet.",
            }
        target = state.secret_state.get("target_item")
        remaining = state.secret_state.get("submissions_left", 0)
        if remaining <= 0:
            return {"ok": False, "error": "No submissions left."}

        state.secret_state["submissions_left"] = max(0, remaining - 1)
        normalized = item_name.strip().lower()
        success = normalized == (target or "").lower()
        if success:
            state.secret_state["outcome"] = "success"
            return {"ok": True, "message": "Correct target identified. You win!"}

        if state.secret_state["submissions_left"] == 0:
            state.secret_state["outcome"] = "failure"
        return {
            "ok": True,
            "message": f"Submission logged: '{item_name}'. Justification: {justification}. "
            f"Submissions left: {state.secret_state['submissions_left']}",
        }

    # --- Helpers ---------------------------------------------------------
    def _choose_candidates(self, env: EnvironmentAdapter, k: int = 5) -> List[str]:
        """
        Choose a candidate set from the world or catalog; ensure at least 3.
        """
        objs: List[str] = []
        try:
            if hasattr(env, "list_objects"):
                objs = env.list_objects()
        except Exception:
            objs = []
        pool = objs if len(objs) >= 3 else self._catalog
        if len(pool) < 3:
            pool = self._catalog
        return self.rng.sample(pool, k=min(k, len(pool)))

    def _build_codebook(self) -> Dict[str, str]:
        """
        Assign each switch a deterministic effect. Could be extended to sequences or multivariate codes.
        """
        rng = random.Random()
        effects = rng.sample(self._effects, k=min(len(self._effects), len(self._switches)))
        return {switch: effect for switch, effect in zip(self._switches, effects)}

    def _build_patterns(self, candidates: List[str], codebook: Dict[str, str]) -> Dict[str, List[str]]:
        """
        Assign each candidate a deterministic effect sequence using the switch->effect mapping.
        Uses the index of the candidate to build a base-N code over the switches.
        """
        switch_list = sorted(codebook.keys())
        base = len(switch_list)
        patterns: Dict[str, List[str]] = {}
        for idx, item in enumerate(candidates):
            digits = self._to_base(idx, base, len(switch_list))
            effects = [codebook[switch_list[d]] for d in digits]
            patterns[item] = effects
        return patterns

    def _to_base(self, number: int, base: int, width: int) -> List[int]:
        digits = []
        n = number
        for _ in range(width):
            digits.append(n % base)
            n //= base
        digits.reverse()
        return digits
