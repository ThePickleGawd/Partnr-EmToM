"""
Text-based world state for EMTOM benchmark.

This provides a lightweight simulation that doesn't require the full Habitat
simulator, enabling fast iteration on mechanics and exploration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

if TYPE_CHECKING:
    from emtom.core.mechanic import Effect


@dataclass
class Entity:
    """
    A thing in the world (room, object, agent, switch, etc.).

    Entities have properties that can be modified by actions and mechanics.
    """

    id: str  # Unique identifier (e.g., "door_kitchen", "switch_1", "agent_0")
    entity_type: str  # Type category ("room", "object", "agent", "switch", "button", etc.)
    properties: Dict[str, Any] = field(default_factory=dict)  # Mutable properties
    location: Optional[str] = None  # Room or container ID this entity is in

    def get_property(self, name: str, default: Any = None) -> Any:
        """Get a property value with optional default."""
        return self.properties.get(name, default)

    def set_property(self, name: str, value: Any) -> None:
        """Set a property value."""
        self.properties[name] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "entity_type": self.entity_type,
            "properties": self.properties.copy(),
            "location": self.location,
        }

    def __repr__(self) -> str:
        return f"Entity(id={self.id!r}, type={self.entity_type!r}, location={self.location!r})"


@dataclass
class TextWorldState:
    """
    Text-based world state for EMTOM.

    Does not require Habitat simulation - all state is represented as
    entities with properties that can be queried and modified.
    """

    entities: Dict[str, Entity] = field(default_factory=dict)
    pending_effects: List["Effect"] = field(default_factory=list)
    step_count: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)

    def add_entity(self, entity: Entity) -> None:
        """Add an entity to the world."""
        self.entities[entity.id] = entity

    def remove_entity(self, entity_id: str) -> Optional[Entity]:
        """Remove and return an entity from the world."""
        return self.entities.pop(entity_id, None)

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        return self.entities.get(entity_id)

    def get_property(
        self, entity_id: str, prop: str, default: Any = None
    ) -> Any:
        """Get a property from an entity."""
        entity = self.entities.get(entity_id)
        if entity:
            return entity.get_property(prop, default)
        return default

    def set_property(self, entity_id: str, prop: str, value: Any) -> bool:
        """
        Set a property on an entity.

        Returns True if successful, False if entity doesn't exist.
        """
        entity = self.entities.get(entity_id)
        if entity:
            entity.set_property(prop, value)
            return True
        return False

    def get_entities_by_type(self, entity_type: str) -> List[Entity]:
        """Get all entities of a specific type."""
        return [e for e in self.entities.values() if e.entity_type == entity_type]

    def get_entities_in_location(self, location_id: str) -> List[Entity]:
        """Get all entities in a specific location (room)."""
        return [e for e in self.entities.values() if e.location == location_id]

    def get_agent_location(self, agent_id: str) -> Optional[str]:
        """Get the current location of an agent."""
        agent = self.entities.get(agent_id)
        return agent.location if agent else None

    def move_entity(self, entity_id: str, new_location: str) -> bool:
        """Move an entity to a new location."""
        entity = self.entities.get(entity_id)
        if entity:
            entity.location = new_location
            return True
        return False

    def get_rooms(self) -> List[Entity]:
        """Get all room entities."""
        return self.get_entities_by_type("room")

    def get_room_ids(self) -> List[str]:
        """Get IDs of all rooms."""
        return [e.id for e in self.get_rooms()]

    def advance_step(self) -> List["Effect"]:
        """
        Advance time by one step and process pending delayed effects.

        Returns the effects that are now ready to be applied.
        """
        self.step_count += 1
        ready_effects = []
        remaining = []

        for effect in self.pending_effects:
            if effect.delay_steps <= 0:
                ready_effects.append(effect)
            else:
                effect.delay_steps -= 1
                remaining.append(effect)

        self.pending_effects = remaining
        return ready_effects

    def add_pending_effect(self, effect: "Effect") -> None:
        """Add a delayed effect to be processed later."""
        self.pending_effects.append(effect)

    def apply_effect(self, effect: "Effect") -> None:
        """Apply an effect to the world state."""
        self.set_property(effect.target, effect.property_changed, effect.new_value)

    def to_text(self, observer_id: str, include_hidden: bool = False) -> str:
        """
        Generate a text description of the world from an observer's perspective.

        Args:
            observer_id: ID of the observing agent
            include_hidden: If True, include all information (for debugging)

        Returns:
            Human-readable description of the world state
        """
        lines = []
        observer = self.entities.get(observer_id)
        observer_location = observer.location if observer else None

        lines.append(f"Step {self.step_count}")
        lines.append("")

        # Describe current room
        if observer_location:
            room = self.entities.get(observer_location)
            room_name = room.get_property("name", observer_location) if room else observer_location
            lines.append(f"You are in: {room_name}")

            # List entities in the room
            room_entities = self.get_entities_in_location(observer_location)
            if room_entities:
                lines.append("You see:")
                for entity in room_entities:
                    if entity.id == observer_id:
                        continue  # Don't list self
                    desc = self._describe_entity(entity, observer_id, include_hidden)
                    if desc:
                        lines.append(f"  - {desc}")
            lines.append("")

        # List available rooms
        rooms = self.get_rooms()
        if rooms:
            other_rooms = [r for r in rooms if r.id != observer_location]
            if other_rooms:
                room_names = [r.get_property("name", r.id) for r in other_rooms]
                lines.append(f"Other rooms: {', '.join(room_names)}")

        return "\n".join(lines)

    def _describe_entity(
        self, entity: Entity, observer_id: str, include_hidden: bool = False
    ) -> str:
        """Generate a description of a single entity."""
        parts = [entity.id]

        # Add relevant property descriptions
        if entity.entity_type == "door":
            is_open = entity.get_property("is_open", False)
            parts.append("(open)" if is_open else "(closed)")

        elif entity.entity_type == "switch":
            is_on = entity.get_property("is_on", False)
            parts.append("(on)" if is_on else "(off)")

        elif entity.entity_type == "button":
            is_pressed = entity.get_property("is_pressed", False)
            is_active = entity.get_property("is_active", False)
            if is_active:
                parts.append("(activated)")
            elif is_pressed:
                parts.append("(pressed)")

        elif entity.entity_type == "light":
            is_on = entity.get_property("is_on", False)
            parts.append("(lit)" if is_on else "(dark)")

        elif entity.entity_type == "object":
            # Generic object description
            pass

        elif entity.entity_type == "agent":
            agent_name = entity.get_property("name", entity.id)
            parts = [agent_name]

        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "step_count": self.step_count,
            "entities": {k: v.to_dict() for k, v in self.entities.items()},
            "pending_effects_count": len(self.pending_effects),
        }

    def snapshot(self) -> Dict[str, Any]:
        """Create a complete snapshot of the world state."""
        return {
            "step_count": self.step_count,
            "entities": {k: v.to_dict() for k, v in self.entities.items()},
            "pending_effects": [
                e.to_dict() for e in self.pending_effects
            ],
        }

    @classmethod
    def from_snapshot(cls, snapshot: Dict[str, Any]) -> "TextWorldState":
        """Restore world state from a snapshot."""
        world = cls()
        world.step_count = snapshot["step_count"]
        for entity_data in snapshot["entities"].values():
            entity = Entity(
                id=entity_data["id"],
                entity_type=entity_data["entity_type"],
                properties=entity_data["properties"],
                location=entity_data.get("location"),
            )
            world.add_entity(entity)
        # Note: pending_effects would need Effect import to restore
        return world


def create_simple_world(
    rooms: List[str],
    agents: List[str],
    objects: Optional[Dict[str, Dict[str, Any]]] = None,
) -> TextWorldState:
    """
    Create a simple world with rooms, agents, and objects.

    Args:
        rooms: List of room IDs
        agents: List of agent IDs
        objects: Dict mapping object_id -> {type, location, properties}

    Returns:
        Initialized TextWorldState
    """
    world = TextWorldState()

    # Add rooms
    for room_id in rooms:
        world.add_entity(
            Entity(
                id=room_id,
                entity_type="room",
                properties={"name": room_id.replace("_", " ").title()},
            )
        )

    # Add agents (all start in first room)
    start_room = rooms[0] if rooms else None
    for i, agent_id in enumerate(agents):
        world.add_entity(
            Entity(
                id=agent_id,
                entity_type="agent",
                properties={"name": f"Agent {i}"},
                location=start_room,
            )
        )

    # Add objects
    if objects:
        for obj_id, obj_data in objects.items():
            world.add_entity(
                Entity(
                    id=obj_id,
                    entity_type=obj_data.get("type", "object"),
                    properties=obj_data.get("properties", {}),
                    location=obj_data.get("location"),
                )
            )

    return world
