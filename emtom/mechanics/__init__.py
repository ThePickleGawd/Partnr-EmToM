"""Mechanics library for EMTOM benchmark."""

from emtom.mechanics.registry import MechanicRegistry, register_mechanic

# Scene-aware mechanics (work with any Habitat scene)
from emtom.mechanics.inverse_state import InverseStateMechanic
from emtom.mechanics.remote_control import RemoteControlMechanic
from emtom.mechanics.counting_state import CountingStateMechanic

__all__ = [
    "MechanicRegistry",
    "register_mechanic",
    # Scene-aware mechanics
    "InverseStateMechanic",
    "RemoteControlMechanic",
    "CountingStateMechanic",
]
