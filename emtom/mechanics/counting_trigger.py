"""
Counting Trigger Mechanic.

A button or object only activates after being interacted with N times.
Tests the agent's ability to discover hidden activation conditions.
"""

from typing import Any, Dict, List, Optional, Set

from emtom.core.mechanic import ActionResult, Effect, Mechanic, MechanicCategory
from emtom.core.world_state import TextWorldState
from emtom.mechanics.registry import register_mechanic


@register_mechanic("counting_trigger")
class CountingTriggerMechanic(Mechanic):
    """
    Objects require multiple interactions to activate.

    A button might only work after being pressed 3 times.
    The agent must discover this hidden threshold through experimentation.

    This mechanic tests whether agents can:
    1. Detect that an action has no immediate effect
    2. Persist in trying the same action
    3. Discover the threshold through trial and error
    4. Remember and apply this knowledge
    """

    name = "counting_trigger"
    category = MechanicCategory.CONDITIONAL
    description = "Objects require multiple interactions to activate"

    def __init__(
        self,
        required_count: int = 3,
        targets: Optional[List[str]] = None,
        target_types: Optional[List[str]] = None,
        show_progress: bool = False,
    ):
        """
        Initialize the counting trigger mechanic.

        Args:
            required_count: Number of interactions needed to activate.
            targets: Specific object IDs that have this behavior.
            target_types: Entity types affected (default: ["button"]).
            show_progress: If True, give hints about progress toward activation.
        """
        self.required_count = required_count
        self.targets: Set[str] = set(targets) if targets else set()
        self.target_types: Set[str] = (
            set(target_types) if target_types else {"button"}
        )
        self.show_progress = show_progress
        # Track press counts per target
        self._press_counts: Dict[str, int] = {}
        # Track which targets have been activated
        self._activated: Set[str] = set()

    def applies_to(
        self, action_name: str, target: str, world_state: TextWorldState
    ) -> bool:
        """Check if this mechanic should handle the action."""
        if action_name != "press":
            return False

        entity = world_state.get_entity(target)
        if entity is None:
            return False

        # Check specific targets first
        if self.targets and target not in self.targets:
            return False

        # Check by type if no specific targets
        if not self.targets and entity.entity_type not in self.target_types:
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
        """Transform press action with counting requirement."""
        # Increment press count
        count = self._press_counts.get(target, 0) + 1
        self._press_counts[target] = count

        # Check if already activated
        if target in self._activated:
            observation = f"You press {target}. It's already activated."
            return ActionResult(
                success=True,
                effects=[],
                pending_effects=[],
                observations={actor_id: observation},
                surprise_triggers={},
            )

        # Check if threshold reached
        if count < self.required_count:
            # Not yet activated
            if self.show_progress:
                remaining = self.required_count - count
                observation = (
                    f"You press {target}. Nothing happens, but you feel it "
                    f"give slightly. ({count}/{self.required_count})"
                )
            else:
                observation = f"You press {target}. Nothing happens."

            surprise_triggers = {
                actor_id: f"Expected {target} to do something, but nothing happened"
            }

            return ActionResult(
                success=True,
                effects=[],
                pending_effects=[],
                observations={actor_id: observation},
                surprise_triggers=surprise_triggers,
            )

        # Threshold reached - activate!
        self._activated.add(target)

        effect = Effect(
            target=target,
            property_changed="is_active",
            old_value=False,
            new_value=True,
            visible_to={actor_id},
            description=f"{target} activated after {count} presses",
        )

        observation = (
            f"You press {target} for the {self._ordinal(count)} time. "
            f"It finally activates with a satisfying click!"
        )

        # This is surprising because it finally worked
        surprise_triggers = {
            actor_id: f"After {count} presses, {target} finally activated! "
            f"It seems to require {self.required_count} presses."
        }

        return ActionResult(
            success=True,
            effects=[effect],
            pending_effects=[],
            observations={actor_id: observation},
            surprise_triggers=surprise_triggers,
        )

    def _ordinal(self, n: int) -> str:
        """Convert number to ordinal string (1st, 2nd, 3rd, etc.)."""
        if 11 <= n % 100 <= 13:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}"

    def get_expected_effect_description(self, action_name: str, target: str) -> str:
        """What an agent would normally expect."""
        return f"Pressing {target} should activate it immediately"

    def get_hidden_state_for_debug(self) -> Dict[str, Any]:
        """Return debug info about this mechanic."""
        return {
            "required_count": self.required_count,
            "press_counts": self._press_counts.copy(),
            "activated": list(self._activated),
        }

    def reset(self) -> None:
        """Reset per-episode state."""
        self._press_counts.clear()
        self._activated.clear()

    def get_press_count(self, target: str) -> int:
        """Get the current press count for a target (for testing)."""
        return self._press_counts.get(target, 0)

    def is_activated(self, target: str) -> bool:
        """Check if a target has been activated (for testing)."""
        return target in self._activated
