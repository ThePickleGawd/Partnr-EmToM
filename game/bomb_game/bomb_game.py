"""
Bomb-in-room social reasoning task built on top of the generic game framework.
Agents see public context plus role-specific private info; tools are exposed
based on game state and current agent location.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from game.game import (
    AgentRole,
    EnvironmentAdapter,
    GameOrchestrator,
    GameSpec,
    GameState,
    ToolDescriptor,
)


class BombGameSpec(GameSpec):
    name = "bomb_game"

    def __init__(
        self, max_defuse_attempts: int = 1, auto_defuse_on_enter: bool = True
    ):
        self.max_defuse_attempts = max_defuse_attempts
        # If True, simply entering the bomb room will auto-defuse. This is useful
        # when integrating with planners that don't expose a dedicated defuse tool.
        self.auto_defuse_on_enter = auto_defuse_on_enter

    # --- GameSpec hooks --------------------------------------------------
    def initialize(
        self, agent_ids: List[str], env: EnvironmentAdapter
    ) -> GameState:
        rooms = env.list_rooms()
        if not rooms:
            raise ValueError("BombGameSpec requires at least one room in the environment.")
        bomb_room = random.choice(rooms)

        roles: Dict[str, AgentRole] = {}
        if not agent_ids:
            raise ValueError("BombGameSpec requires at least one agent id.")
        # First agent is Guide, others are Seekers.
        guide_id = agent_ids[0]
        roles[guide_id] = AgentRole(
            name="Guide", private_info={"bomb_room": bomb_room}
        )
        for seeker_id in agent_ids[1:]:
            roles[seeker_id] = AgentRole(name="Seeker", private_info={})

        public_info = {
            "game": "bomb_game",
            "rules": (
                "One room hides an armed bomb. Only the Guide knows which room. "
                "The team must move into rooms, communicate, and defuse the bomb. "
                "Defusing is only possible while physically in the bomb room."
            ),
            "rooms": rooms,
            "roles": {aid: role.name for aid, role in roles.items()},
        }

        secret_state = {
            "bomb_room": bomb_room,
            "defused": False,
            "failed": False,
            "defuse_attempts_left": self.max_defuse_attempts,
        }

        # Reveal the hidden room in logs for debugging/verification.
        print(f"[BombGame] Secret bomb room: {bomb_room}")

        return GameState(
            agent_roles=roles,
            public_info=public_info,
            secret_state=secret_state,
        )

    def get_tools_for_agent(
        self, agent_id: str, state: GameState, env: EnvironmentAdapter
    ) -> List[ToolDescriptor]:
        tools: List[ToolDescriptor] = []

        tools.append(
            ToolDescriptor(
                name="list_rooms",
                description="List the rooms in the house.",
                handler=self._list_rooms,
            )
        )
        tools.append(
            ToolDescriptor(
                name="move_to_room",
                description="Move to a target room by name.",
                parameters={"room_name": "string"},
                handler=self._move_to_room,
            )
        )
        tools.append(
            ToolDescriptor(
                name="send_message",
                description="Send a short message to another agent.",
                parameters={"receiver_id": "string", "message": "string"},
                handler=self._send_message,
            )
        )
        tools.append(
            ToolDescriptor(
                name="where_am_i",
                description="Report your current room.",
                handler=self._where_am_i,
            )
        )

        # Bomb-specific tool; only appears when in the bomb room and not finished.
        if not state.secret_state.get("defused") and not state.secret_state.get("failed"):
            current_room = env.get_agent_room(agent_id)
            if current_room and current_room == state.secret_state.get("bomb_room"):
                tools.append(
                    ToolDescriptor(
                        name="defuse_bomb",
                        description="Attempt to defuse the bomb (only available in the bomb room).",
                        handler=self._defuse_bomb,
                    )
                )

        return tools

    def check_end_condition(
        self, state: GameState, env: EnvironmentAdapter
    ) -> Tuple[bool, Optional[str]]:
        if state.secret_state.get("defused"):
            return True, "success"
        if state.secret_state.get("failed"):
            return True, "failure"
        if state.secret_state.get("defuse_attempts_left", 0) <= 0:
            return True, "failure"
        return False, None

    def maybe_auto_resolve(self, state: GameState, env: EnvironmentAdapter) -> None:
        """
        Optionally auto-defuse when any agent enters the bomb room.
        """
        if not self.auto_defuse_on_enter:
            return
        if state.secret_state.get("defused") or state.secret_state.get("failed"):
            return
        bomb_room = state.secret_state.get("bomb_room")
        for agent_id in state.agent_roles.keys():
            room = env.get_agent_room(agent_id)
            if room and room == bomb_room:
                attempts_left = state.secret_state.get("defuse_attempts_left", 0)
                if attempts_left <= 0:
                    state.secret_state["failed"] = True
                    return
                state.secret_state["defused"] = True
                state.secret_state["defuse_attempts_left"] = attempts_left - 1
                return

    def render_public_context(self, state: GameState) -> str:
        return (
            f"Game: Bomb defusal. Rooms: {state.public_info.get('rooms', [])}. "
            "Guideline: navigate rooms, communicate, defuse only in the correct room."
        )

    def render_private_context(self, agent_id: str, state: GameState) -> str:
        role = state.agent_roles.get(agent_id)
        if not role:
            return ""
        if role.name == "Guide":
            return f"You know the bomb is in {role.private_info.get('bomb_room')}."
        return ""

    # --- Tool handlers ----------------------------------------------------
    def _list_rooms(
        self, agent_id: str, orchestrator: GameOrchestrator, **_: Any
    ) -> Dict[str, Any]:
        return {"ok": True, "rooms": orchestrator.env.list_rooms()}

    def _move_to_room(
        self, agent_id: str, orchestrator: GameOrchestrator, room_name: str, **_: Any
    ) -> Dict[str, Any]:
        result = orchestrator.env.move_agent_to_room(agent_id, room_name)
        return result

    def _send_message(
        self,
        agent_id: str,
        orchestrator: GameOrchestrator,
        receiver_id: str,
        message: str,
        **_: Any,
    ) -> Dict[str, Any]:
        orchestrator.env.send_message(agent_id, receiver_id, message)
        return {"ok": True}

    def _where_am_i(
        self, agent_id: str, orchestrator: GameOrchestrator, **_: Any
    ) -> Dict[str, Any]:
        room = orchestrator.env.get_agent_room(agent_id)
        return {"ok": True, "room": room}

    def _defuse_bomb(
        self, agent_id: str, orchestrator: GameOrchestrator, **_: Any
    ) -> Dict[str, Any]:
        state = orchestrator.state
        env = orchestrator.env
        current_room = env.get_agent_room(agent_id)
        bomb_room = state.secret_state.get("bomb_room")
        if current_room != bomb_room:
            return {"ok": False, "error": "Not in the bomb room."}

        if state.secret_state.get("defused"):
            return {"ok": True, "message": "Bomb already defused."}

        attempts_left = state.secret_state.get("defuse_attempts_left", 0)
        if attempts_left <= 0:
            state.secret_state["failed"] = True
            return {"ok": False, "error": "No defuse attempts left."}

        state.secret_state["defused"] = True
        state.secret_state["defuse_attempts_left"] = attempts_left - 1
        return {"ok": True, "message": "Bomb defused!"}
