"""
Switch flip communication game:
- Agent_0 (Flipper) navigates to rooms and flips switches.
- Agent_1 (Observer) sees the switch states but with a room mapping applied.
- Observer has a target pattern and must guide Flipper to achieve it through communication.
- The mapping is hidden - agents must discover it through trial and communication.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from game.game import AgentRole, EnvironmentAdapter, GameSpec, GameState, ToolDescriptor


class SwitchGameSpec(GameSpec):
    name = "switch_game"

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = random.Random(seed)

    def initialize(
        self, agent_ids: List[str], env: EnvironmentAdapter
    ) -> GameState:
        if len(agent_ids) < 2:
            raise ValueError("SwitchGameSpec requires two agents (flipper and observer).")

        flipper_id = agent_ids[0]
        observer_id = agent_ids[1]

        rooms = env.list_rooms()
        if not rooms:
            raise ValueError("SwitchGameSpec requires at least one room in the environment.")

        # Create random permutation mapping (bijection)
        shuffled = rooms.copy()
        self.rng.shuffle(shuffled)
        room_mapping = dict(zip(rooms, shuffled))  # agent_0's room -> agent_1's view

        # Generate target pattern (which rooms should be ON from agent_1's view)
        num_targets = max(1, len(rooms) // 2)
        target_rooms = set(self.rng.sample(rooms, k=num_targets))

        roles: Dict[str, AgentRole] = {
            flipper_id: AgentRole(name="Flipper", private_info={}),
            observer_id: AgentRole(name="Observer", private_info={}),
        }

        # All switches start OFF
        switch_states = {room: False for room in rooms}

        secret_state = {
            "room_mapping": room_mapping,
            "switch_states": switch_states,  # From agent_0's perspective
            "target_pattern": target_rooms,
            "outcome": None,
        }

        public_info = {
            "game": "switch_game",
            "rooms": rooms,
            "rules": (
                "Flipper flips switches in rooms. Observer sees the results and must guide "
                "Flipper to achieve the target pattern. The room that Flipper toggles may "
                "appear as a different room to the Observer."
            ),
        }

        # Debug output
        print(f"[SwitchGame] Room mapping: {room_mapping}")
        print(f"[SwitchGame] Target pattern (Observer view): {sorted(target_rooms)}")

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

        if role.name == "Flipper":
            # Only show flip_switch when agent is in a room
            current_room = env.get_agent_room(agent_id)
            if current_room:
                tools.append(
                    ToolDescriptor(
                        name="flip_switch",
                        description=(
                            "Flip the switch in your current room. Toggles between on/off. "
                            "Example: flip_switch[]"
                        ),
                        parameters={},
                        handler=self._flip_switch,
                    )
                )

        if role.name == "Observer":
            tools.append(
                ToolDescriptor(
                    name="check_game_state",
                    description=(
                        "Check which rooms currently have their switches ON (from your perspective). "
                        "Also shows the target pattern. Example: check_game_state[]"
                    ),
                    parameters={},
                    handler=self._check_game_state,
                )
            )
            tools.append(
                ToolDescriptor(
                    name="submit_pattern",
                    description=(
                        "Submit when you believe the target pattern has been achieved. "
                        "This ends the game. Example: submit_pattern[]"
                    ),
                    parameters={},
                    handler=self._submit_pattern,
                )
            )

        return tools

    def check_end_condition(
        self, state: GameState, env: EnvironmentAdapter
    ) -> Tuple[bool, Optional[str]]:
        outcome = state.secret_state.get("outcome")
        if outcome is not None:
            return True, outcome
        return False, None

    def render_public_context(self, state: GameState) -> str:
        return (
            "Game: Switch flip puzzle. Flipper toggles switches in rooms; "
            "Observer sees mapped results and has a target pattern to achieve."
        )

    def render_private_context(self, agent_id: str, state: GameState) -> str:
        role = state.agent_roles.get(agent_id)
        if not role:
            return ""
        if role.name == "Flipper":
            rooms = state.public_info.get("rooms", [])
            return (
                f"You are the Flipper. Available rooms: {rooms}. "
                "Navigate to rooms and use flip_switch to toggle switches. "
                "The Observer will tell you which rooms to flip. "
                "Note: the room you flip may appear as a different room to the Observer."
            )
        if role.name == "Observer":
            target = sorted(state.secret_state["target_pattern"])
            return (
                f"You are the Observer. Target rooms that must be ON: {target}. "
                "Use check_game_state to see the current state. Guide the Flipper to achieve the pattern. "
                "Note: the room the Flipper toggles may appear as a different room to you."
            )
        return ""

    def debug_summary(self, state: GameState, env: EnvironmentAdapter) -> list:
        mapping = state.secret_state.get("room_mapping", {})
        target = sorted(state.secret_state.get("target_pattern", []))
        switches = state.secret_state.get("switch_states", {})
        return [
            f"Room mapping: {mapping}",
            f"Target pattern: {target}",
            f"Current switches: {switches}",
        ]

    # --- Tool handlers ----------------------------------------------------

    def _flip_switch(
        self, agent_id: str, orchestrator, **_: Dict[str, Any]
    ) -> Dict[str, Any]:
        state = orchestrator.state
        env = orchestrator.env
        current_room = env.get_agent_room(agent_id)

        if not current_room:
            return {"ok": False, "error": "Cannot determine your current room."}

        # Toggle the switch
        switch_states = state.secret_state["switch_states"]
        if current_room not in switch_states:
            return {"ok": False, "error": f"No switch found in room '{current_room}'."}

        switch_states[current_room] = not switch_states[current_room]

        new_state = "ON" if switch_states[current_room] else "OFF"
        return {
            "ok": True,
            "message": f"Flipped switch in {current_room}. It is now {new_state}.",
        }

    def _check_game_state(
        self, agent_id: str, orchestrator, **_: Dict[str, Any]
    ) -> Dict[str, Any]:
        state = orchestrator.state
        room_mapping = state.secret_state["room_mapping"]
        switch_states = state.secret_state["switch_states"]
        target_pattern = state.secret_state["target_pattern"]

        # Transform to observer's view using the mapping
        observed_on = []
        for flipper_room, is_on in switch_states.items():
            if is_on:
                mapped_room = room_mapping[flipper_room]
                observed_on.append(mapped_room)

        rooms_on = sorted(observed_on)
        target = sorted(target_pattern)
        message = f"Current rooms ON: {rooms_on if rooms_on else 'None'}. Target pattern: {target}."

        return {
            "ok": True,
            "message": message,
            "rooms_on": rooms_on,
            "target_pattern": target,
        }

    def _submit_pattern(
        self, agent_id: str, orchestrator, **_: Dict[str, Any]
    ) -> Dict[str, Any]:
        state = orchestrator.state
        room_mapping = state.secret_state["room_mapping"]
        switch_states = state.secret_state["switch_states"]
        target_pattern = state.secret_state["target_pattern"]

        # Get current pattern from observer's view
        current_on = set()
        for flipper_room, is_on in switch_states.items():
            if is_on:
                current_on.add(room_mapping[flipper_room])

        if current_on == target_pattern:
            state.secret_state["outcome"] = "success"
            return {"ok": True, "message": "Correct pattern! You win!"}
        else:
            state.secret_state["outcome"] = "failure"
            return {
                "ok": False,
                "message": f"Incorrect pattern. Current: {sorted(current_on)}, Target: {sorted(target_pattern)}",
            }
