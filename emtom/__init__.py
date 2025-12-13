"""
EMTOM: Embodied Theory of Mind Benchmark

A framework for testing theory of mind reasoning through mechanics
with "unexpected behaviors" that induce surprise and require mental modeling.
"""

from emtom.core.mechanic import Mechanic, MechanicCategory, Effect, ActionResult
from emtom.core.world_state import TextWorldState, Entity
from emtom.mechanics.registry import MechanicRegistry, register_mechanic

__all__ = [
    "Mechanic",
    "MechanicCategory",
    "Effect",
    "ActionResult",
    "TextWorldState",
    "Entity",
    "MechanicRegistry",
    "register_mechanic",
]
