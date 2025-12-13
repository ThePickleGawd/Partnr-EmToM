"""
Core mechanic abstractions for EMTOM benchmark.

Mechanics define how actions produce effects that may differ from agent expectations,
enabling theory of mind reasoning through "unexpected behaviors".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

if TYPE_CHECKING:
    from emtom.core.world_state import TextWorldState


class MechanicCategory(Enum):
    """Categories of mechanics based on their unexpected behavior type."""

    INVERSE = "inverse"  # Actions have opposite effects
    HIDDEN_MAPPING = "hidden_mapping"  # Actions affect unexpected targets
    CONDITIONAL = "conditional"  # Effects depend on hidden state
    TIME_DELAYED = "time_delayed"  # Effects happen after a delay
    PER_AGENT = "per_agent"  # Different agents observe different effects
    COMPOUND = "compound"  # Multiple mechanics combined


@dataclass
class Effect:
    """
    Represents the result of an action on the world state.

    Effects can be immediate or delayed, and may be visible to different agents.
    """

    target: str  # Entity ID that was affected
    property_changed: str  # Which property changed
    old_value: Any  # Previous value
    new_value: Any  # New value
    visible_to: Set[str] = field(default_factory=set)  # Which agents can see this
    delay_steps: int = 0  # Steps until effect manifests (0 = immediate)
    description: str = ""  # Human-readable description of the effect

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "target": self.target,
            "property_changed": self.property_changed,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "visible_to": list(self.visible_to),
            "delay_steps": self.delay_steps,
            "description": self.description,
        }


@dataclass
class ActionResult:
    """
    Result of applying an action through the mechanics system.

    Contains immediate and delayed effects, per-agent observations,
    and triggers for surprise detection.
    """

    success: bool  # Whether the action succeeded
    effects: List[Effect] = field(default_factory=list)  # Immediate effects
    pending_effects: List[Effect] = field(default_factory=list)  # Delayed effects
    observations: Dict[str, str] = field(
        default_factory=dict
    )  # Per-agent observation text
    surprise_triggers: Dict[str, str] = field(
        default_factory=dict
    )  # Per-agent surprise explanations
    error_message: Optional[str] = None  # Error message if action failed

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "effects": [e.to_dict() for e in self.effects],
            "pending_effects": [e.to_dict() for e in self.pending_effects],
            "observations": self.observations,
            "surprise_triggers": self.surprise_triggers,
            "error_message": self.error_message,
        }


class Mechanic(ABC):
    """
    Base class for EMTOM mechanics.

    Each mechanic defines how actions produce effects that may differ from
    agent expectations. Subclasses implement specific "unexpected behaviors"
    that induce theory of mind reasoning.

    Mechanics are applied in order; the first matching mechanic handles the action.
    """

    name: str = "base_mechanic"
    category: MechanicCategory = MechanicCategory.INVERSE
    description: str = "Base mechanic - override in subclass"

    @abstractmethod
    def applies_to(
        self, action_name: str, target: str, world_state: "TextWorldState"
    ) -> bool:
        """
        Check if this mechanic should intercept this action.

        Args:
            action_name: Name of the action being performed (e.g., "open", "press")
            target: ID of the target entity
            world_state: Current world state

        Returns:
            True if this mechanic should handle the action
        """
        pass

    @abstractmethod
    def transform_effect(
        self,
        action_name: str,
        actor_id: str,
        target: str,
        intended_effect: Effect,
        world_state: "TextWorldState",
    ) -> ActionResult:
        """
        Transform the intended effect into the actual effect.

        This is where the "unexpected behavior" happens. The mechanic
        can modify, replace, or add to the intended effect.

        Args:
            action_name: Name of the action being performed
            actor_id: ID of the agent performing the action
            target: ID of the target entity
            intended_effect: What would normally happen
            world_state: Current world state

        Returns:
            ActionResult with actual effects and observations
        """
        pass

    def get_expected_effect_description(self, action_name: str, target: str) -> str:
        """
        Return a description of what an agent would normally expect from this action.

        Used for surprise detection - comparing expected vs actual outcomes.
        """
        return f"Performing {action_name} on {target}"

    def get_hidden_state_for_debug(self) -> Dict[str, Any]:
        """
        Return any hidden state for debugging/logging.

        This is NOT shown to agents but can be logged for analysis.
        """
        return {}

    def reset(self) -> None:
        """
        Reset any internal state of the mechanic.

        Called at the start of each episode.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, category={self.category.value})"


def create_default_effect(
    action_name: str,
    target: str,
    world_state: "TextWorldState",
) -> Effect:
    """
    Create the default/expected effect for a standard action.

    This is what would happen in a "normal" world without unexpected mechanics.
    """
    # Map action names to expected property changes
    action_effects = {
        "open": ("is_open", True),
        "close": ("is_open", False),
        "toggle": ("is_on", None),  # Toggle flips current value
        "press": ("is_pressed", True),
        "activate": ("is_active", True),
        "deactivate": ("is_active", False),
        "pick_up": ("is_held", True),
        "put_down": ("is_held", False),
        "turn_on": ("is_on", True),
        "turn_off": ("is_on", False),
    }

    if action_name in action_effects:
        prop, new_val = action_effects[action_name]
        old_val = world_state.get_property(target, prop, False)
        if new_val is None:  # Toggle
            new_val = not old_val
        return Effect(
            target=target,
            property_changed=prop,
            old_value=old_val,
            new_value=new_val,
            description=f"{action_name.replace('_', ' ')} {target}",
        )

    # Generic effect for unknown actions
    return Effect(
        target=target,
        property_changed="last_action",
        old_value=None,
        new_value=action_name,
        description=f"Performed {action_name} on {target}",
    )
