"""
Remote Switch Mechanic.

A switch in one location controls an object in a different location.
Tests the agent's ability to discover hidden cause-effect relationships.
"""

import random
from typing import Any, Dict, List, Optional, Set

from emtom.core.mechanic import ActionResult, Effect, Mechanic, MechanicCategory
from emtom.core.world_state import TextWorldState
from emtom.mechanics.registry import register_mechanic


@register_mechanic("remote_switch")
class RemoteSwitchMechanic(Mechanic):
    """
    Switches control objects in different locations.

    A switch in room A might control a light in room B.
    The agent must discover this hidden mapping through exploration.

    This mechanic tests whether agents can:
    1. Discover non-local cause-effect relationships
    2. Build a mental model of the hidden mapping
    3. Use this knowledge to achieve goals
    4. Communicate mappings to other agents (theory of mind)
    """

    name = "remote_switch"
    category = MechanicCategory.HIDDEN_MAPPING
    description = "Switches control objects in different locations"

    def __init__(
        self,
        mappings: Optional[Dict[str, str]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the remote switch mechanic.

        Args:
            mappings: Dict mapping switch_id -> controlled_object_id.
                If None, mappings are created randomly at initialization.
            seed: Random seed for generating mappings.
        """
        self.mappings: Dict[str, str] = mappings or {}
        self.seed = seed
        self._rng = random.Random(seed)
        self._initialized = bool(mappings)
        # Track which agents have discovered which mappings
        self._discovered: Dict[str, Set[str]] = {}

    def _initialize_mappings(self, world_state: TextWorldState) -> None:
        """Generate random mappings from switches to controllable objects."""
        if self._initialized:
            return

        switches = world_state.get_entities_by_type("switch")
        controllable = [
            e for e in world_state.entities.values()
            if e.entity_type in ("light", "door", "fan", "lamp")
        ]

        if not switches or not controllable:
            self._initialized = True
            return

        # Create random bijection (or partial mapping if unequal counts)
        self._rng.shuffle(controllable)
        for i, switch in enumerate(switches):
            if i < len(controllable):
                self.mappings[switch.id] = controllable[i].id

        self._initialized = True

    def applies_to(
        self, action_name: str, target: str, world_state: TextWorldState
    ) -> bool:
        """Check if this mechanic should handle the action."""
        # Initialize mappings on first check
        if not self._initialized:
            self._initialize_mappings(world_state)

        # Only applies to toggle action on switches
        if action_name != "toggle":
            return False

        entity = world_state.get_entity(target)
        if entity is None or entity.entity_type != "switch":
            return False

        # Only if this switch has a mapping
        return target in self.mappings

    def transform_effect(
        self,
        action_name: str,
        actor_id: str,
        target: str,
        intended_effect: Effect,
        world_state: TextWorldState,
    ) -> ActionResult:
        """Transform toggle action to affect a remote object."""
        controlled = self.mappings.get(target)
        if not controlled:
            return ActionResult(
                success=False,
                error_message=f"Switch {target} has no mapped target",
                observations={actor_id: f"You flip {target}. Nothing seems to happen."},
            )

        controlled_entity = world_state.get_entity(controlled)
        if not controlled_entity:
            return ActionResult(
                success=False,
                error_message=f"Controlled object {controlled} not found",
                observations={actor_id: f"You flip {target}. It clicks, but nothing happens."},
            )

        # Toggle the controlled object (in a different location)
        old_state = world_state.get_property(controlled, "is_on", False)
        new_state = not old_state

        # Effect on the distant object
        controlled_location = controlled_entity.location
        effect = Effect(
            target=controlled,
            property_changed="is_on",
            old_value=old_state,
            new_value=new_state,
            visible_to=self._get_observers_in_location(controlled_location, world_state),
            description=f"{controlled} turned {'on' if new_state else 'off'}",
        )

        # Actor only sees the switch flip, not necessarily the distant effect
        actor_location = world_state.get_agent_location(actor_id)
        actor_obs = f"You flip {target}. It clicks."

        # Build observations based on what each agent can see
        observations = {actor_id: actor_obs}
        surprise_triggers = {}

        # If actor is in same location as controlled object, they see the effect
        if actor_location == controlled_location:
            state_word = "on" if new_state else "off"
            observations[actor_id] = f"You flip {target}. The {controlled} turns {state_word}!"
            # This is surprising because the switch affected something else
            surprise_triggers[actor_id] = (
                f"Flipping {target} affected {controlled} in the same room"
            )
            self._record_discovery(actor_id, target)
        else:
            # Actor doesn't see immediate effect - potential future surprise
            surprise_triggers[actor_id] = (
                f"Switch {target} clicked but no immediate visible effect"
            )

        # Other agents in the controlled object's location see the effect
        for agent_id in self._get_agents_in_location(controlled_location, world_state):
            if agent_id != actor_id:
                state_word = "on" if new_state else "off"
                observations[agent_id] = f"The {controlled} suddenly turns {state_word}!"
                surprise_triggers[agent_id] = (
                    f"{controlled} changed state without anyone nearby touching it"
                )

        return ActionResult(
            success=True,
            effects=[effect],
            pending_effects=[],
            observations=observations,
            surprise_triggers=surprise_triggers,
        )

    def _get_observers_in_location(
        self, location: str, world_state: TextWorldState
    ) -> Set[str]:
        """Get IDs of agents who can observe changes in a location."""
        agents = world_state.get_entities_by_type("agent")
        return {
            a.id for a in agents
            if a.location == location
        }

    def _get_agents_in_location(
        self, location: str, world_state: TextWorldState
    ) -> List[str]:
        """Get list of agent IDs in a location."""
        agents = world_state.get_entities_by_type("agent")
        return [a.id for a in agents if a.location == location]

    def _record_discovery(self, agent_id: str, switch_id: str) -> None:
        """Record that an agent has discovered a switch mapping."""
        if agent_id not in self._discovered:
            self._discovered[agent_id] = set()
        self._discovered[agent_id].add(switch_id)

    def get_expected_effect_description(self, action_name: str, target: str) -> str:
        """What an agent would normally expect."""
        return f"Flipping {target} should toggle it on/off locally"

    def get_hidden_state_for_debug(self) -> Dict[str, Any]:
        """Return debug info about this mechanic."""
        return {
            "mappings": self.mappings.copy(),
            "discovered_by_agent": {
                k: list(v) for k, v in self._discovered.items()
            },
        }

    def reset(self) -> None:
        """Reset per-episode state."""
        self._discovered.clear()
        # Keep mappings if they were explicitly provided
        if self.seed is not None and not self.mappings:
            self._rng = random.Random(self.seed)
            self._initialized = False
