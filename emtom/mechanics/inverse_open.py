"""
Inverse Open/Close Mechanic.

Actions have the opposite effect: 'open' closes things, 'close' opens them.
Tests the agent's ability to discover and adapt to inverted action semantics.
"""

from typing import Any, Dict, List, Optional, Set

from emtom.core.mechanic import ActionResult, Effect, Mechanic, MechanicCategory
from emtom.core.world_state import TextWorldState
from emtom.mechanics.registry import register_mechanic


@register_mechanic("inverse_open")
class InverseOpenMechanic(Mechanic):
    """
    Open/close actions have inverse effects.

    When an agent tries to 'open' something, it closes instead.
    When an agent tries to 'close' something, it opens instead.

    This mechanic tests whether agents can:
    1. Detect the unexpected behavior
    2. Update their mental model
    3. Use the inverse action to achieve their goal
    """

    name = "inverse_open"
    category = MechanicCategory.INVERSE
    description = "Open/close actions have inverse effects"

    def __init__(
        self,
        affected_objects: Optional[List[str]] = None,
        affected_types: Optional[List[str]] = None,
    ):
        """
        Initialize the inverse open mechanic.

        Args:
            affected_objects: Specific object IDs that are affected.
                If None, all openable objects are affected.
            affected_types: Entity types that are affected (e.g., ["door", "cabinet"]).
                If None, defaults to ["door"].
        """
        self.affected_objects: Set[str] = (
            set(affected_objects) if affected_objects else set()
        )
        self.affected_types: Set[str] = (
            set(affected_types) if affected_types else {"door"}
        )

    def applies_to(
        self, action_name: str, target: str, world_state: TextWorldState
    ) -> bool:
        """Check if this mechanic should handle the action."""
        # Only applies to open/close actions
        if action_name not in ("open", "close"):
            return False

        # Check if specific objects are configured
        if self.affected_objects and target not in self.affected_objects:
            return False

        # Check entity type
        entity = world_state.get_entity(target)
        if entity is None:
            return False

        # If no specific objects configured, check by type
        if not self.affected_objects:
            if entity.entity_type not in self.affected_types:
                return False

        return True

    def transform_effect(
        self,
        action_name: str,
        actor_id: str,
        target: str,
        intended_effect: Effect,
        world_state: TextWorldState,
    ) -> ActionResult:
        """Transform the open/close action to have inverse effect."""
        current_state = world_state.get_property(target, "is_open", False)

        # Inverse the expected behavior
        if action_name == "open":
            # Instead of opening, it closes (or stays closed)
            new_state = False
            if current_state:
                observation = f"You try to open {target}, but it slams shut instead!"
            else:
                observation = f"You try to open {target}, but it remains firmly closed."
        else:  # close
            # Instead of closing, it opens
            new_state = True
            if current_state:
                observation = f"You try to close {target}, but it remains wide open."
            else:
                observation = f"You try to close {target}, but it swings open instead!"

        effect = Effect(
            target=target,
            property_changed="is_open",
            old_value=current_state,
            new_value=new_state,
            visible_to={actor_id},
            description=observation,
        )

        # Determine if this should trigger surprise
        # Surprise occurs when the result is opposite to intention
        expected_state = action_name == "open"
        surprise_triggers = {}

        if new_state != expected_state:
            surprise_triggers[actor_id] = (
                f"Expected {target} to be {'open' if expected_state else 'closed'}, "
                f"but it became {'open' if new_state else 'closed'}"
            )

        return ActionResult(
            success=True,
            effects=[effect],
            pending_effects=[],
            observations={actor_id: observation},
            surprise_triggers=surprise_triggers,
        )

    def get_expected_effect_description(self, action_name: str, target: str) -> str:
        """What an agent would normally expect."""
        if action_name == "open":
            return f"{target} should open"
        else:
            return f"{target} should close"

    def get_hidden_state_for_debug(self) -> Dict[str, Any]:
        """Return debug info about this mechanic."""
        return {
            "affected_objects": list(self.affected_objects),
            "affected_types": list(self.affected_types),
        }
