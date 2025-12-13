"""Mechanics library for EMTOM benchmark."""

from emtom.mechanics.registry import MechanicRegistry, register_mechanic

# Import all mechanics to trigger registration
# Original (type-specific) mechanics
from emtom.mechanics.inverse_open import InverseOpenMechanic
from emtom.mechanics.remote_switch import RemoteSwitchMechanic
from emtom.mechanics.counting_trigger import CountingTriggerMechanic

# New object-agnostic (scene-aware) mechanics
from emtom.mechanics.inverse_state import InverseStateMechanic
from emtom.mechanics.remote_control import RemoteControlMechanic
from emtom.mechanics.counting_state import CountingStateMechanic

__all__ = [
    "MechanicRegistry",
    "register_mechanic",
    # Original mechanics (retained for backwards compatibility)
    "InverseOpenMechanic",
    "RemoteSwitchMechanic",
    "CountingTriggerMechanic",
    # New scene-aware mechanics
    "InverseStateMechanic",
    "RemoteControlMechanic",
    "CountingStateMechanic",
]
