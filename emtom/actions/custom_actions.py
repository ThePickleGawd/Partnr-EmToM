"""
Custom EMTOM Actions.

These actions provide rich interactions that can be affected by mechanics.
Each action has:
- A normal expected behavior
- Can be transformed by mechanics (inverse, remote control, counting, etc.)
- Produces observations that may differ per agent (theory of mind)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from habitat_llm.agent.env import EnvironmentInterface


@dataclass
class ActionResult:
    """Result of executing a custom action."""
    success: bool
    observation: str  # What the acting agent observes
    effect: Optional[str] = None  # What actually changed
    other_observations: Dict[str, str] = field(default_factory=dict)  # What other agents observe
    surprise_trigger: Optional[str] = None  # If this should trigger surprise detection


class EMTOMAction(ABC):
    """Base class for EMTOM custom actions."""

    name: str = "base_action"
    description: str = "Base action"

    @abstractmethod
    def execute(
        self,
        agent_id: str,
        target: Optional[str],
        env_interface: "EnvironmentInterface",
        world_state: Dict[str, Any],
    ) -> ActionResult:
        """
        Execute the action.

        Args:
            agent_id: The agent performing the action
            target: Optional target for the action
            env_interface: Habitat environment interface
            world_state: Current world state info

        Returns:
            ActionResult with observation and effects
        """
        pass

    def get_available_targets(
        self,
        env_interface: "EnvironmentInterface",
        world_state: Dict[str, Any],
    ) -> List[str]:
        """Get valid targets for this action in current state."""
        return []


class FlipLightsAction(EMTOMAction):
    """
    Flip the lights in the current room or a specific room.

    Normal behavior: Toggles all lights in the room.
    Can be affected by:
    - inverse_state: Lights toggle opposite direction
    - remote_control: Flipping lights here affects lights elsewhere
    - counting_state: Requires multiple flips to actually toggle
    """

    name = "FlipLights"
    description = "Toggle the lights in a room. Affects all lights in the specified room."

    def execute(
        self,
        agent_id: str,
        target: Optional[str],
        env_interface: "EnvironmentInterface",
        world_state: Dict[str, Any],
    ) -> ActionResult:
        room = target or world_state.get("agent_location", "current room")

        # Get current light state
        lights_on = world_state.get("room_lights", {}).get(room, False)
        new_state = not lights_on

        state_word = "on" if new_state else "off"
        observation = f"You flip the light switch. The lights in {room} turn {state_word}."

        return ActionResult(
            success=True,
            observation=observation,
            effect=f"lights_{room}={'on' if new_state else 'off'}",
        )

    def get_available_targets(
        self,
        env_interface: "EnvironmentInterface",
        world_state: Dict[str, Any],
    ) -> List[str]:
        return world_state.get("rooms", [])


class PressButtonAction(EMTOMAction):
    """
    Press a button in the environment.

    Normal behavior: Activates the button, may trigger connected systems.
    Can be affected by:
    - inverse_state: Button press has opposite effect
    - remote_control: Button controls something unexpected
    - counting_state: Requires multiple presses to activate
    """

    name = "PressButton"
    description = "Press a button. Buttons may control various systems in the environment."

    def execute(
        self,
        agent_id: str,
        target: Optional[str],
        env_interface: "EnvironmentInterface",
        world_state: Dict[str, Any],
    ) -> ActionResult:
        button = target or "the button"

        observation = f"You press {button}. You hear a soft click."

        return ActionResult(
            success=True,
            observation=observation,
            effect=f"button_pressed={button}",
        )

    def get_available_targets(
        self,
        env_interface: "EnvironmentInterface",
        world_state: Dict[str, Any],
    ) -> List[str]:
        # Find button-like objects
        buttons = []
        for entity in world_state.get("entities", []):
            if "button" in entity.get("type", "").lower() or "switch" in entity.get("name", "").lower():
                buttons.append(entity.get("name", entity.get("id")))
        return buttons


class PullLeverAction(EMTOMAction):
    """
    Pull a lever in the environment.

    Normal behavior: Toggles the lever position and activates connected systems.
    Can be affected by:
    - inverse_state: Lever has opposite effect
    - remote_control: Lever controls something in another room
    - counting_state: Must pull multiple times for full effect
    """

    name = "PullLever"
    description = "Pull a lever. Levers typically control mechanical systems or doors."

    def execute(
        self,
        agent_id: str,
        target: Optional[str],
        env_interface: "EnvironmentInterface",
        world_state: Dict[str, Any],
    ) -> ActionResult:
        lever = target or "the lever"

        # Get current lever state
        lever_state = world_state.get("lever_states", {}).get(lever, "up")
        new_state = "down" if lever_state == "up" else "up"

        observation = f"You pull {lever}. It moves from {lever_state} to {new_state} position."

        return ActionResult(
            success=True,
            observation=observation,
            effect=f"lever_{lever}={new_state}",
        )

    def get_available_targets(
        self,
        env_interface: "EnvironmentInterface",
        world_state: Dict[str, Any],
    ) -> List[str]:
        levers = []
        for entity in world_state.get("entities", []):
            if "lever" in entity.get("type", "").lower():
                levers.append(entity.get("name", entity.get("id")))
        return levers


class TurnDialAction(EMTOMAction):
    """
    Turn a dial to a specific setting.

    Normal behavior: Changes the dial to the specified value.
    Can be affected by:
    - inverse_state: Dial turns opposite direction
    - remote_control: Dial controls something unexpected
    - counting_state: Must turn multiple times to reach setting
    """

    name = "TurnDial"
    description = "Turn a dial to adjust a setting. Dials control values like temperature, volume, or intensity."

    def execute(
        self,
        agent_id: str,
        target: Optional[str],
        env_interface: "EnvironmentInterface",
        world_state: Dict[str, Any],
    ) -> ActionResult:
        dial = target or "the dial"

        # Simulate turning the dial
        current_value = world_state.get("dial_values", {}).get(dial, 5)
        # Turn clockwise by default
        new_value = min(10, current_value + 1)

        observation = f"You turn {dial} clockwise. The setting changes from {current_value} to {new_value}."

        return ActionResult(
            success=True,
            observation=observation,
            effect=f"dial_{dial}={new_value}",
        )

    def get_available_targets(
        self,
        env_interface: "EnvironmentInterface",
        world_state: Dict[str, Any],
    ) -> List[str]:
        dials = []
        for entity in world_state.get("entities", []):
            if "dial" in entity.get("type", "").lower() or "knob" in entity.get("name", "").lower():
                dials.append(entity.get("name", entity.get("id")))
        return dials


class InspectAction(EMTOMAction):
    """
    Carefully inspect an object to learn about its properties.

    Normal behavior: Reveals detailed information about the object.
    Can be affected by:
    - Mechanics may hide some information
    - Different agents may see different things
    """

    name = "Inspect"
    description = "Carefully examine an object to learn about its properties and state."

    def execute(
        self,
        agent_id: str,
        target: Optional[str],
        env_interface: "EnvironmentInterface",
        world_state: Dict[str, Any],
    ) -> ActionResult:
        if not target:
            return ActionResult(
                success=False,
                observation="You need to specify what to inspect.",
            )

        # Get entity info
        entity_info = world_state.get("entity_details", {}).get(target, {})

        if not entity_info:
            observation = f"You look closely at {target}. It appears to be a normal object."
        else:
            properties = entity_info.get("properties", {})
            states = entity_info.get("states", {})

            details = []
            if properties:
                for k, v in list(properties.items())[:3]:
                    details.append(f"{k}: {v}")
            if states:
                for k, v in list(states.items())[:3]:
                    state_word = "yes" if v else "no"
                    details.append(f"{k.replace('is_', '')}: {state_word}")

            if details:
                observation = f"You examine {target} closely. You notice: {', '.join(details)}."
            else:
                observation = f"You examine {target}. Nothing unusual stands out."

        return ActionResult(
            success=True,
            observation=observation,
        )

    def get_available_targets(
        self,
        env_interface: "EnvironmentInterface",
        world_state: Dict[str, Any],
    ) -> List[str]:
        # Can inspect any entity
        return [e.get("name", e.get("id")) for e in world_state.get("entities", [])]


class RingBellAction(EMTOMAction):
    """
    Ring a bell to make a sound.

    Normal behavior: Makes a sound that can be heard in nearby rooms.
    Useful for signaling other agents.
    """

    name = "RingBell"
    description = "Ring a bell to make a sound. Can be heard in adjacent rooms."

    def execute(
        self,
        agent_id: str,
        target: Optional[str],
        env_interface: "EnvironmentInterface",
        world_state: Dict[str, Any],
    ) -> ActionResult:
        bell = target or "the bell"
        current_room = world_state.get("agent_location", "unknown")

        observation = f"You ring {bell}. A clear tone echoes through the area."

        # Other agents hear the bell
        other_observations = {}
        for other_agent in world_state.get("other_agents", []):
            other_observations[other_agent] = f"You hear a bell ringing from {current_room}."

        return ActionResult(
            success=True,
            observation=observation,
            effect=f"bell_rang={current_room}",
            other_observations=other_observations,
        )


class CheckStatusAction(EMTOMAction):
    """
    Check the status of a system or device.

    Normal behavior: Reports current state of the target.
    """

    name = "CheckStatus"
    description = "Check the current status of a device or system."

    def execute(
        self,
        agent_id: str,
        target: Optional[str],
        env_interface: "EnvironmentInterface",
        world_state: Dict[str, Any],
    ) -> ActionResult:
        if not target:
            return ActionResult(
                success=False,
                observation="You need to specify what to check.",
            )

        # Get status from world state
        entity_info = world_state.get("entity_details", {}).get(target, {})
        states = entity_info.get("states", {})

        if states:
            status_parts = []
            for state, value in states.items():
                state_name = state.replace("is_", "")
                status_parts.append(f"{state_name}: {'yes' if value else 'no'}")
            observation = f"Status of {target}: {', '.join(status_parts)}"
        else:
            observation = f"You check {target}. It appears to be functioning normally."

        return ActionResult(
            success=True,
            observation=observation,
        )


# Registry of all EMTOM custom actions
EMTOM_ACTIONS: Dict[str, EMTOMAction] = {
    "FlipLights": FlipLightsAction(),
    "PressButton": PressButtonAction(),
    "PullLever": PullLeverAction(),
    "TurnDial": TurnDialAction(),
    "Inspect": InspectAction(),
    "RingBell": RingBellAction(),
    "CheckStatus": CheckStatusAction(),
}


class EMTOMActionExecutor:
    """
    Executor for EMTOM custom actions.

    Integrates custom actions with the Habitat environment and mechanics system.
    """

    def __init__(
        self,
        env_interface: "EnvironmentInterface",
        mechanics: Optional[List[Any]] = None,
    ):
        self.env = env_interface
        self.mechanics = mechanics or []
        self.actions = dict(EMTOM_ACTIONS)

    def get_available_actions(self, world_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get list of available custom actions with their targets."""
        available = []

        for name, action in self.actions.items():
            targets = action.get_available_targets(self.env, world_state)
            available.append({
                "name": name,
                "description": action.description,
                "targets": targets,
            })

        return available

    def execute(
        self,
        action_name: str,
        agent_id: str,
        target: Optional[str],
        world_state: Dict[str, Any],
    ) -> ActionResult:
        """
        Execute a custom action, applying any relevant mechanics.
        """
        if action_name not in self.actions:
            return ActionResult(
                success=False,
                observation=f"Unknown action: {action_name}",
            )

        action = self.actions[action_name]

        # Execute the base action
        result = action.execute(agent_id, target, self.env, world_state)

        # Apply mechanics transformations
        for mechanic in self.mechanics:
            if hasattr(mechanic, 'transform_action_result'):
                result = mechanic.transform_action_result(
                    action_name, agent_id, target, result, world_state
                )

        return result

    def register_action(self, action: EMTOMAction) -> None:
        """Register a new custom action."""
        self.actions[action.name] = action
