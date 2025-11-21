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
from habitat_llm.tools import Tool
import os
from PIL import Image, ImageDraw, ImageFont


class BombGameSpec(GameSpec):
    name = "bomb_game"

    def __init__(
        self, max_defuse_attempts: int = 1, auto_defuse_on_enter: bool = False
    ):
        self.max_defuse_attempts = max_defuse_attempts
        # If True, simply entering the bomb room will auto-defuse. Default False to require explicit defuse.
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
                "There is no physical bomb object to pick up. "
                "The team must navigate to the correct room and use the defuse_bomb tool there."
            ),
            "rooms": rooms,
            "roles": {aid: role.name for aid, role in roles.items()},
        }

        secret_state = {
            "bomb_room": bomb_room,
            "defused": False,
            "failed": False,
            "defuse_attempts_left": self.max_defuse_attempts,
            "latest_image_path": None,
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

        # Bomb-specific tool; only appears when in the bomb room and not finished.
        if not state.secret_state.get("defused") and not state.secret_state.get("failed"):
            current_room = env.get_agent_room(agent_id)
            if current_room and current_room == state.secret_state.get("bomb_room"):
                tools.append(
                    ToolDescriptor(
                        name="defuse_bomb",
                        description="Cut the correct wire to defuse the bomb (only available in the bomb room).",
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
        When an agent enters the bomb room, show the image. If auto_defuse_on_enter is True,
        also mark success; otherwise wait for explicit defuse tool call.
        """
        if state.secret_state.get("defused") or state.secret_state.get("failed"):
            return
        bomb_room = state.secret_state.get("bomb_room")
        for agent_id in state.agent_roles.keys():
            room = env.get_agent_room(agent_id)
            if room and room == bomb_room:
                try:
                    state.secret_state["latest_image_path"] = render_wire_puzzle(
                        wires=[("red", "armed"), ("blue", "safe"), ("green", "safe")],
                        highlighted="red",
                        save_dir="outputs/bomb_game",
                        filename="bomb_room.png",
                    )
                except Exception as e:
                    print(f"[BombGame] Failed to render bomb-room image: {e}")
                if self.auto_defuse_on_enter:
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
            "Guideline: there is no bomb object. Navigate to the correct room and call defuse_bomb there."
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
        # render a success state
        try:
            state.secret_state["latest_image_path"] = render_wire_puzzle(
                wires=[("red", "cut"), ("blue", "safe"), ("green", "safe")],
                highlighted="red",
                save_dir="outputs/bomb_game",
                filename="bomb_defused.png",
            )
        except Exception as e:
            print(f"[BombGame] Failed to render defuse image: {e}")
        return {
            "ok": True,
            "message": "Bomb defused!",
            "image": state.secret_state.get("latest_image_path"),
        }


class GameTool(Tool):
    """
    Simple Tool wrapper around BombGameSpec tool descriptors so planners can invoke
    game-specific actions without modifying agent configs.
    """

    def __init__(self, descriptor: ToolDescriptor, orchestrator: GameOrchestrator, agent_uid: int):
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
        # input_query may include arguments separated by comma; pass raw to handler
        try:
            result = self.descriptor.handler(agent_id=str(self.agent_uid), orchestrator=self.orchestrator, **self._parse_args(input_query))
            if isinstance(result, dict):
                if result.get("ok", True):
                    return None, result.get("message", "Success")
                return None, result.get("error", "Failed")
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


# --- Bomb-specific rendering utilities (PIL-based to avoid SDL issues) ------


def render_wire_puzzle(
    wires: List[Tuple[str, str]],
    highlighted: str = "",
    save_dir: str = "outputs",
    filename: str = "wire.png",
) -> str:
    """
    Render a simple wire puzzle image.

    :param wires: list of (color, status) tuples; status is a small label ("safe", "armed", etc.)
    :param highlighted: optional color name to highlight (e.g., the correct wire)
    :param save_dir: directory to write the image
    :param filename: output filename

    :return: path to the saved PNG (empty string on failure)
    """
    width, height = 600, 400
    try:
        img = Image.new("RGB", (width, height), (20, 20, 20))
        draw = ImageDraw.Draw(img)
    except Exception as e:
        print(f"[BombGame] PIL init failed: {e}")
        return ""

    colors: Dict[str, Tuple[int, int, int]] = {
        "red": (220, 20, 60),
        "blue": (65, 105, 225),
        "green": (50, 205, 50),
        "yellow": (255, 215, 0),
        "white": (245, 245, 245),
        "orange": (255, 140, 0),
        "pink": (255, 105, 180),
        "purple": (138, 43, 226),
    }

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    y_step = height // (len(wires) + 1)

    for idx, (color_name, status) in enumerate(wires, start=1):
        color = colors.get(color_name, (200, 200, 200))
        y = idx * y_step
        draw.line((50, y, width - 50, y), fill=color, width=8)
        label = f"{color_name.upper()} ({status})"
        draw.text((60, y - 20), label, fill=(240, 240, 240), font=font)
        if highlighted and highlighted == color_name:
            draw.ellipse(
                (width - 82, y - 12, width - 58, y + 12),
                outline=(255, 255, 255),
                width=3,
            )

    try:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, filename)
        img.save(out_path, format="PNG")
        return out_path
    except Exception as e:
        print(f"[BombGame] Failed to save wire image: {e}")
        return ""
