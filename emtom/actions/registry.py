"""
Action registry for EMTOM benchmark.

Provides a decorator-based registration system for custom actions,
enabling plug-and-play action selection via configuration.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from emtom.actions.custom_actions import EMTOMAction

# Global registry of action classes
_ACTION_REGISTRY: Dict[str, Type["EMTOMAction"]] = {}


def register_action(name: Optional[str] = None):
    """
    Decorator to register a custom action class.

    Usage:
        @register_action("Hide")
        class HideAction(EMTOMAction):
            ...

        # Or use the class's name attribute:
        @register_action()
        class HideAction(EMTOMAction):
            name = "Hide"
    """

    def decorator(cls: Type["EMTOMAction"]) -> Type["EMTOMAction"]:
        action_name = name or getattr(cls, "name", cls.__name__)
        if action_name in _ACTION_REGISTRY:
            raise ValueError(
                f"Action '{action_name}' is already registered "
                f"(by {_ACTION_REGISTRY[action_name].__name__})"
            )
        _ACTION_REGISTRY[action_name] = cls
        return cls

    return decorator


class ActionRegistry:
    """
    Central registry for all available custom actions.

    Provides methods to query, instantiate, and compose actions.
    """

    @staticmethod
    def get(name: str) -> Type["EMTOMAction"]:
        """Get an action class by name."""
        if name not in _ACTION_REGISTRY:
            available = ", ".join(sorted(_ACTION_REGISTRY.keys()))
            raise KeyError(
                f"Unknown action: '{name}'. Available: {available}"
            )
        return _ACTION_REGISTRY[name]

    @staticmethod
    def list_all() -> List[str]:
        """List all registered action names."""
        return sorted(_ACTION_REGISTRY.keys())

    @staticmethod
    def is_registered(name: str) -> bool:
        """Check if an action is registered."""
        return name in _ACTION_REGISTRY

    @staticmethod
    def instantiate(name: str, **params) -> "EMTOMAction":
        """Create an instance of an action."""
        cls = ActionRegistry.get(name)
        return cls(**params)

    @staticmethod
    def instantiate_all() -> Dict[str, "EMTOMAction"]:
        """Instantiate all registered actions."""
        return {name: cls() for name, cls in _ACTION_REGISTRY.items()}

    @staticmethod
    def get_info(name: str) -> Dict[str, Any]:
        """Get information about a registered action."""
        cls = ActionRegistry.get(name)
        return {
            "name": name,
            "description": getattr(cls, "description", ""),
            "class": cls.__name__,
        }

    @staticmethod
    def describe_all() -> str:
        """Get a human-readable description of all registered actions."""
        lines = ["Registered Actions:", "=" * 40]
        for name in sorted(_ACTION_REGISTRY.keys()):
            info = ActionRegistry.get_info(name)
            desc = info["description"][:60] + "..." if len(info["description"]) > 60 else info["description"]
            lines.append(f"  - {name}: {desc}")
        return "\n".join(lines)


def clear_registry() -> None:
    """Clear all registered actions (useful for testing)."""
    _ACTION_REGISTRY.clear()


def get_registry() -> Dict[str, Type["EMTOMAction"]]:
    """Get the raw registry dict (useful for testing)."""
    return _ACTION_REGISTRY
