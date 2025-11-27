"""
Lightweight game orchestration framework that sits on top of PARTNR/Habitat without
touching the existing codebase. The orchestrator keeps track of game state, roles,
hidden/public info, and exposes a dynamic tool interface to agents while routing
embodiment actions through a thin environment adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# --- Tooling primitives ----------------------------------------------------

@dataclass
class ToolDescriptor:
    """
    Describes a callable tool that the planner/agent can invoke.

    `handler` gets called with (agent_id, orchestrator) and any kwargs the
    planner supplies.
    """

    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    handler: Callable[..., Dict[str, Any]] = field(default=lambda: {})


# --- Environment adapter ---------------------------------------------------

class EnvironmentAdapter:
    """
    Thin shim in front of the PARTNR/Habitat environment.
    Implement these methods using EnvironmentInterface or a custom driver.
    """

    def list_rooms(self) -> List[str]:
        raise NotImplementedError

    def get_agent_room(self, agent_id: str) -> Optional[str]:
        raise NotImplementedError

    def move_agent_to_room(self, agent_id: str, room_name: str) -> Dict[str, Any]:
        """
        Should perform navigation/teleportation and return a status dict.
        """
        raise NotImplementedError

    def send_message(self, sender_id: str, receiver_id: str, message: str) -> None:
        """
        Broadcast message to another agent; can be implemented through the
        existing communication channel in EnvironmentInterface.
        """
        raise NotImplementedError


class InMemoryAdapter(EnvironmentAdapter):
    """
    Minimal adapter useful for dry-runs or unit tests; tracks agent locations
    without touching Habitat.
    """

    def __init__(self, rooms: List[str], agent_start_room: Optional[str] = None):
        self.rooms = rooms
        self.agent_room: Dict[str, str] = {}
        self.agent_start_room = agent_start_room or (rooms[0] if rooms else "")
        self.messages: List[Tuple[str, str, str]] = []

    def list_rooms(self) -> List[str]:
        return list(self.rooms)

    def get_agent_room(self, agent_id: str) -> Optional[str]:
        return self.agent_room.get(agent_id, self.agent_start_room)

    def move_agent_to_room(self, agent_id: str, room_name: str) -> Dict[str, Any]:
        if room_name not in self.rooms:
            return {"ok": False, "error": f"Unknown room {room_name}"}
        self.agent_room[agent_id] = room_name
        return {"ok": True, "room": room_name}

    def send_message(self, sender_id: str, receiver_id: str, message: str) -> None:
        self.messages.append((sender_id, receiver_id, message))


# --- Game state primitives -------------------------------------------------

@dataclass
class AgentRole:
    name: str
    private_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GameState:
    agent_roles: Dict[str, AgentRole] = field(default_factory=dict)
    public_info: Dict[str, Any] = field(default_factory=dict)
    secret_state: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    terminal: bool = False
    outcome: Optional[str] = None  # e.g., "success" / "failure"


# --- Game spec interface ---------------------------------------------------

class GameSpec:
    """
    Override this to implement a new game. The orchestrator calls these hooks.
    """

    name: str = "base"

    def initialize(
        self, agent_ids: List[str], env: EnvironmentAdapter
    ) -> GameState:
        raise NotImplementedError

    def get_tools_for_agent(
        self, agent_id: str, state: GameState, env: EnvironmentAdapter
    ) -> List[ToolDescriptor]:
        raise NotImplementedError

    def check_end_condition(
        self, state: GameState, env: EnvironmentAdapter
    ) -> Tuple[bool, Optional[str]]:
        """
        Return (terminal, outcome).
        """
        raise NotImplementedError

    def render_public_context(self, state: GameState) -> str:
        """
        Human/LLM-friendly description of the game that can be appended to
        prompts. Should omit hidden info.
        """
        raise NotImplementedError

    def render_private_context(self, agent_id: str, state: GameState) -> str:
        """
        Agent-specific private info (e.g., asymmetric roles).
        """
        return ""


# --- Orchestrator ----------------------------------------------------------

class GameOrchestrator:
    """
    Maintains game state, mediates tool exposure, and calls down into the
    environment adapter for embodiment actions.
    """

    def __init__(self, game_spec: GameSpec, env_adapter: EnvironmentAdapter):
        self.game_spec = game_spec
        self.env = env_adapter
        self.state: Optional[GameState] = None
        self.turn_limit: Optional[int] = None
        self.turn_count: int = 0

    def start(self, agent_ids: List[str]) -> GameState:
        self.state = self.game_spec.initialize(agent_ids, self.env)
        return self.state

    def increment_turn(self) -> bool:
        """
        Increment turn counter. If a turn limit is set and exceeded, mark terminal.
        Returns True if still allowed to proceed, False if limit reached.
        """
        self.turn_count += 1
        if self.turn_limit is not None and self.turn_count >= self.turn_limit:
            if self.state:
                self.state.terminal = True
                if self.state.outcome is None:
                    self.state.outcome = "failure_turn_limit"
            return False
        return True

    def get_public_info(self) -> Dict[str, Any]:
        return self.state.public_info if self.state else {}

    def get_private_info(self, agent_id: str) -> Dict[str, Any]:
        if not self.state:
            return {}
        role = self.state.agent_roles.get(agent_id)
        return role.private_info if role else {}

    def describe(self, agent_id: str) -> str:
        """
        Render a combined public/private description for prompting.
        """
        if not self.state:
            return ""
        public = self.game_spec.render_public_context(self.state)
        private = self.game_spec.render_private_context(agent_id, self.state)
        return public + (("\nPrivate info: " + private) if private else "")

    def available_tools(self, agent_id: str) -> List[ToolDescriptor]:
        if not self.state:
            return []
        return self.game_spec.get_tools_for_agent(agent_id, self.state, self.env)

    def step(
        self, agent_id: str, tool_name: str, **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Execute a tool on behalf of an agent. The tool handler is expected to
        mutate the game state as needed.
        """
        if not self.state:
            raise RuntimeError("Game not started")

        tools = {tool.name: tool for tool in self.available_tools(agent_id)}
        if tool_name not in tools:
            return {"ok": False, "error": f"Tool {tool_name} not available"}

        result = tools[tool_name].handler(agent_id=agent_id, orchestrator=self, **kwargs)
        # Track in history
        entry = {"agent_id": agent_id, "tool": tool_name, "args": kwargs, "result": result}
        self.state.history.append(entry)

        # Evaluate termination
        terminal, outcome = self.game_spec.check_end_condition(self.state, self.env)
        self.state.terminal = terminal
        self.state.outcome = outcome

        return result
