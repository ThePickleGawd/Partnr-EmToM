"""
Time-shifted reasoning game where two agents inhabit the same room at different
points in time. The past agent can prepare objects and leave degraded clues; the
future agent inspects the decayed scene and must identify the intended target.
"""

from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Optional, Tuple

from game.game import (
    AgentRole,
    EnvironmentAdapter,
    GameSpec,
    GameState,
    ToolDescriptor,
)


class TimeGameSpec(GameSpec):
    """
    Game flow:
    - Agent_0 starts as the Past role, Agent_1+ are Future roles.
    - Past can inspect the pristine room, mark/protect items, and then call
      advance_time to lock in the future state.
    - Future inspects the decayed room and submits the target item. Success if
      the submission matches the hidden target. Communication is allowed.
    """

    name = "time_game"

    def __init__(
        self,
        max_submissions: int = 2,
        decay_strength: float = 0.35,
    ) -> None:
        self.max_submissions = max_submissions
        self.decay_strength = decay_strength
        self.rng = random.Random()
        self.allowed_tools = {
            "inspect_past_room",
            "annotate_item",
            "protect_item",
            "reposition_item",
            "advance_time",
            "inspect_future_room",
            "stabilize_item",
            "submit_target_item",
            "decay_rules",
        }

        self._catalog = [
            {"name": "steel wrench", "material": "metal"},
            {"name": "paper map", "material": "paper"},
            {"name": "wooden crate", "material": "wood"},
            {"name": "fabric banner", "material": "fabric"},
            {"name": "glass bottle", "material": "glass"},
            {"name": "battery pack", "material": "electronics"},
            {"name": "rope coil", "material": "fiber"},
        ]
        self._locations = [
            "desk",
            "floor center",
            "by the door",
            "on the shelf",
            "near the window",
            "on the bed",
        ]
        # Material-specific decay parameters; decay is applied once when time advances.
        self._decay_profiles: Dict[str, Dict[str, float]] = {
            "metal": {"decay": 0.28, "noise": 0.07},
            "wood": {"decay": 0.22, "noise": 0.06},
            "paper": {"decay": 0.40, "noise": 0.10},
            "fabric": {"decay": 0.30, "noise": 0.08},
            "electronics": {"decay": 0.42, "noise": 0.12},
            "glass": {"decay": 0.12, "noise": 0.05},
            "fiber": {"decay": 0.26, "noise": 0.07},
        }
        # Treatments the past agent can apply to slow decay.
        self._protections = {
            "oil": {
                "materials": {"metal"},
                "scale": 0.45,
                "fallback_scale": 0.95,
                "note": "Oiled surfaces rust slower.",
            },
            "wrap": {
                "materials": {"paper", "electronics", "fabric"},
                "scale": 0.6,
                "fallback_scale": 0.9,
                "note": "Wrapping reduces moisture and smudging.",
            },
            "elevate": {
                "materials": {"wood", "paper", "fabric"},
                "scale": 0.7,
                "fallback_scale": 0.95,
                "note": "Keeping it off the floor avoids puddles.",
            },
        }

    # --- GameSpec hooks --------------------------------------------------
    def initialize(
        self, agent_ids: List[str], env: EnvironmentAdapter
    ) -> GameState:
        if not agent_ids:
            raise ValueError("TimeGameSpec requires at least one agent id.")

        rooms = env.list_rooms()
        room_name = rooms[0] if rooms else "the room"
        past_id = agent_ids[0]
        roles: Dict[str, AgentRole] = {past_id: AgentRole(name="Past", private_info={})}
        for fut_id in agent_ids[1:]:
            roles[fut_id] = AgentRole(name="Future", private_info={})

        items = self._build_initial_items()
        target_item = self.rng.choice(items)["name"]
        roles[past_id].private_info["target_item"] = target_item

        public_info = {
            "game": "time_game",
            "rooms": [room_name],
            "rules": (
                "Two copies of the same room exist at different times. "
                "The Past agent prepares the room; the Future agent inspects the decayed state and must name the target item. "
                "Communication is allowed, but the environment will distort physical artifacts."
            ),
        }

        secret_state = {
            "phase": "past",
            "room_name": room_name,
            "initial_items": items,
            "future_items": [],
            "decay_applied": False,
            "target_item": target_item,
            "submissions_left": self.max_submissions,
            "outcome": None,
            "latest_image_path": None,
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

        phase = state.secret_state.get("phase", "past")
        tools: List[ToolDescriptor] = [
            ToolDescriptor(
                name="decay_rules",
                description="Show how each material decays over time so you can plan accordingly.",
                handler=self._decay_rules,
            )
        ]

        if role.name == "Past" and phase == "past":
            tools.extend(
                [
                    ToolDescriptor(
                        name="inspect_past_room",
                        description="List the pristine items, materials, and current placement in the past room.",
                        handler=self._inspect_past_room,
                    ),
                    ToolDescriptor(
                        name="annotate_item",
                        description="Attach a short note to an item. Text may smudge over time.",
                        parameters={"item_name": "string", "message": "string"},
                        handler=self._annotate_item,
                    ),
                    ToolDescriptor(
                        name="protect_item",
                        description="Treat an item to resist decay. Methods: oil, wrap, elevate.",
                        parameters={"item_name": "string", "method": "string"},
                        handler=self._protect_item,
                    ),
                    ToolDescriptor(
                        name="reposition_item",
                        description="Move an item to a named spot (e.g., shelf, door, window) to encode a signal.",
                        parameters={"item_name": "string", "location": "string"},
                        handler=self._reposition_item,
                    ),
                    ToolDescriptor(
                        name="advance_time",
                        description="Lock in your preparations and jump the scenario to the future.",
                        handler=self._advance_time,
                    ),
                ]
            )

        if phase == "future":
            tools.append(
                ToolDescriptor(
                    name="inspect_future_room",
                    description="See the decayed items, their condition, and any surviving markings.",
                    handler=self._inspect_future_room,
                )
            )
            tools.append(
                ToolDescriptor(
                    name="stabilize_item",
                    description="Clean or reinforce a future item to slightly improve readability.",
                    parameters={"item_name": "string"},
                    handler=self._stabilize_item,
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
        if state.secret_state.get("submissions_left", 0) <= 0:
            return True, "failure"
        return False, None

    def render_public_context(self, state: GameState) -> str:
        phase = state.secret_state.get("phase", "past")
        room = state.secret_state.get("room_name", "the room")
        return (
            f"Game: Time-separated room puzzle in {room}. Current phase: {phase.upper()}. "
            "Past prepares objects; Future inspects decay. Communication is allowed."
        )

    def render_private_context(self, agent_id: str, state: GameState) -> str:
        role = state.agent_roles.get(agent_id)
        if not role:
            return ""
        if role.name == "Past":
            target = role.private_info.get("target_item", "")
            return (
                f"You are in the early timeline. Protect and mark items so the future agent can recover '{target}'. "
                "When ready, call advance_time."
            )
        if role.name == "Future":
            return (
                "You arrive after time has passed. Inspect the damaged scene and infer the intended target item."
            )
        return ""

    # --- Tool handlers ----------------------------------------------------
    def _inspect_past_room(
        self, agent_id: str, orchestrator, **_: Any
    ) -> Dict[str, Any]:
        items = orchestrator.state.secret_state.get("initial_items", [])
        return {"ok": True, "items": self._summarize_items(items, reveal_notes=True)}

    def _annotate_item(
        self,
        agent_id: str,
        orchestrator,
        item_name: str = "",
        message: str = "",
        **_: Any,
    ) -> Dict[str, Any]:
        items = orchestrator.state.secret_state.get("initial_items", [])
        item = self._find_item(items, item_name)
        if not item:
            return {"ok": False, "error": f"Unknown item '{item_name}'"}
        item["note"] = message.strip()[:140]
        return {"ok": True, "message": f"Annotated {item['name']} with a note."}

    def _protect_item(
        self,
        agent_id: str,
        orchestrator,
        item_name: str = "",
        method: str = "",
        **_: Any,
    ) -> Dict[str, Any]:
        method = method.lower().strip()
        if method not in self._protections:
            return {"ok": False, "error": f"Unknown protection '{method}'. Use oil, wrap, or elevate."}
        items = orchestrator.state.secret_state.get("initial_items", [])
        item = self._find_item(items, item_name)
        if not item:
            return {"ok": False, "error": f"Unknown item '{item_name}'"}
        item["protection"] = method
        return {
            "ok": True,
            "message": f"Applied '{method}' to {item['name']}: {self._protections[method]['note']}",
        }

    def _reposition_item(
        self,
        agent_id: str,
        orchestrator,
        item_name: str = "",
        location: str = "",
        **_: Any,
    ) -> Dict[str, Any]:
        items = orchestrator.state.secret_state.get("initial_items", [])
        item = self._find_item(items, item_name)
        if not item:
            return {"ok": False, "error": f"Unknown item '{item_name}'"}
        loc = location.strip() or self.rng.choice(self._locations)
        item["location"] = loc
        return {"ok": True, "message": f"Placed {item['name']} at '{loc}'."}

    def _advance_time(self, agent_id: str, orchestrator, **_: Any) -> Dict[str, Any]:
        state = orchestrator.state
        if state.secret_state.get("decay_applied"):
            return {"ok": True, "message": "Time already advanced."}
        initial_items = state.secret_state.get("initial_items", [])
        future_items = [self._apply_decay(copy.deepcopy(item)) for item in initial_items]
        state.secret_state["future_items"] = future_items
        state.secret_state["decay_applied"] = True
        state.secret_state["phase"] = "future"
        return {"ok": True, "message": "Time advanced. Future agent may inspect the decayed room now."}

    def _inspect_future_room(
        self, agent_id: str, orchestrator, **_: Any
    ) -> Dict[str, Any]:
        state = orchestrator.state
        if not state.secret_state.get("decay_applied"):
            return {
                "ok": False,
                "error": "Time has not advanced yet. Ask the past agent to call advance_time.",
            }
        future_items = state.secret_state.get("future_items", [])
        return {"ok": True, "items": self._summarize_items(future_items, reveal_notes=True)}

    def _stabilize_item(
        self, agent_id: str, orchestrator, item_name: str = "", **_: Any
    ) -> Dict[str, Any]:
        state = orchestrator.state
        if not state.secret_state.get("decay_applied"):
            return {"ok": False, "error": "Cannot stabilize before time advance."}
        future_items = state.secret_state.get("future_items", [])
        item = self._find_item(future_items, item_name)
        if not item:
            return {"ok": False, "error": f"Unknown item '{item_name}'"}
        item["condition"] = min(1.0, item.get("condition", 1.0) + 0.1)
        item["status"] = self._condition_to_status(item["condition"])
        if item.get("note"):
            item["note"] = item["note"] + " (cleaned)"
        return {"ok": True, "message": f"Stabilized {item['name']}; condition now {item['status']}."}

    def _submit_target_item(
        self,
        agent_id: str,
        orchestrator,
        item_name: str = "",
        justification: str = "",
        **_: Any,
    ) -> Dict[str, Any]:
        state = orchestrator.state
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

    def _decay_rules(self, agent_id: str, orchestrator, **_: Any) -> Dict[str, Any]:
        rules = []
        for material, profile in self._decay_profiles.items():
            rules.append(
                {
                    "material": material,
                    "expected_decay": profile["decay"],
                    "noise": profile["noise"],
                }
            )
        protections = {
            name: {
                **{k: v for k, v in cfg.items() if k != "materials"},
                "materials": sorted(list(cfg.get("materials", []))),
            }
            for name, cfg in self._protections.items()
        }
        return {"ok": True, "rules": rules, "protections": protections}

    # --- Helpers ---------------------------------------------------------
    def _build_initial_items(self) -> List[Dict[str, Any]]:
        count = min(5, len(self._catalog))
        chosen = self.rng.sample(self._catalog, count)
        items: List[Dict[str, Any]] = []
        for base in chosen:
            item = {
                "name": base["name"],
                "material": base["material"],
                "condition": 1.0,
                "note": "",
                "location": self.rng.choice(self._locations),
                "protection": None,
                "status": "intact",
            }
            items.append(item)
        return items

    def _apply_decay(self, item: Dict[str, Any]) -> Dict[str, Any]:
        material = item.get("material", "unknown")
        profile = self._decay_profiles.get(
            material, {"decay": self.decay_strength, "noise": 0.05}
        )
        base_decay = profile["decay"]
        noise = self.rng.uniform(-profile["noise"], profile["noise"])
        decay_amt = max(0.05, min(0.95, base_decay + noise))

        protection = item.get("protection")
        if protection and protection in self._protections:
            prot = self._protections[protection]
            scale = prot.get("fallback_scale", 1.0)
            if not prot.get("materials") or material in prot.get("materials", set()):
                scale = prot.get("scale", scale)
            decay_amt *= scale

        condition = max(0.05, min(1.0, item.get("condition", 1.0) * (1 - decay_amt)))
        item["condition"] = condition
        item["status"] = self._condition_to_status(condition)
        item["note"] = self._degrade_text(item.get("note", ""), decay_amt)
        return item

    def _degrade_text(self, text: str, severity: float) -> str:
        if not text:
            return ""
        keep_ratio = max(0.2, 1 - severity * 1.2)
        trimmed_len = max(1, int(len(text) * keep_ratio))
        degraded = text[:trimmed_len]
        if trimmed_len < len(text):
            degraded += " ..."
        return degraded

    def _condition_to_status(self, condition: float) -> str:
        if condition >= 0.85:
            return "intact"
        if condition >= 0.65:
            return "worn"
        if condition >= 0.4:
            return "damaged"
        if condition >= 0.2:
            return "badly_damaged"
        return "ruined"

    def _find_item(self, items: List[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
        normalized = name.strip().lower()
        for item in items:
            if item.get("name", "").lower() == normalized:
                return item
        return None

    def _summarize_items(self, items: List[Dict[str, Any]], reveal_notes: bool = False) -> List[Dict[str, Any]]:
        summary: List[Dict[str, Any]] = []
        for item in items:
            entry = {
                "name": item.get("name"),
                "material": item.get("material"),
                "condition": round(item.get("condition", 1.0), 2),
                "status": item.get("status", "intact"),
                "location": item.get("location", "unknown"),
            }
            if reveal_notes and item.get("note"):
                entry["note"] = item.get("note")
            if item.get("protection"):
                entry["protection"] = item.get("protection")
            summary.append(entry)
        return summary
