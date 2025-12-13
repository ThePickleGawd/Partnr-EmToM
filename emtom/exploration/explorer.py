"""
Main exploration loop for EMTOM benchmark.

Orchestrates agents exploring a world with mechanics,
detecting surprises, and logging trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from emtom.core.mechanic import (
    ActionResult,
    Effect,
    Mechanic,
    SceneAwareMechanic,
    create_default_effect,
)
from emtom.core.world_state import TextWorldState
from emtom.exploration.curiosity import ActionChoice, CuriosityModel, RandomCuriosityModel
from emtom.exploration.surprise_detector import (
    SurpriseAssessment,
    SurpriseDetector,
    RuleBasedSurpriseDetector,
)
from emtom.exploration.trajectory_logger import SurpriseRecord, TrajectoryLogger


@dataclass
class ExplorationConfig:
    """Configuration for the exploration loop."""

    max_steps: int = 100
    agent_ids: List[str] = field(default_factory=lambda: ["agent_0"])
    log_path: str = "data/trajectories/emtom"
    snapshot_frequency: int = 0  # 0 = no snapshots
    stop_on_terminal: bool = True  # Stop if world reaches terminal state


@dataclass
class StepResult:
    """Result of a single exploration step."""

    step: int
    agent_actions: Dict[str, ActionChoice]
    action_results: Dict[str, ActionResult]
    surprises: List[SurpriseRecord]
    is_terminal: bool = False


class ExplorationLoop:
    """
    Main exploration loop where agents freely interact with mechanics.

    The loop:
    1. Gets world description for each agent
    2. Uses curiosity model to select actions
    3. Executes actions through mechanics
    4. Detects surprises via LLM self-report
    5. Logs everything for later task generation
    """

    def __init__(
        self,
        world_state: TextWorldState,
        mechanics: List[Mechanic],
        curiosity_model: Union[CuriosityModel, RandomCuriosityModel],
        surprise_detector: Union[SurpriseDetector, RuleBasedSurpriseDetector],
        config: Optional[ExplorationConfig] = None,
    ):
        """
        Initialize the exploration loop.

        Args:
            world_state: Initial world state
            mechanics: List of mechanics to apply
            curiosity_model: Model for action selection
            surprise_detector: Model for surprise detection
            config: Exploration configuration
        """
        self.world = world_state
        self.mechanics = mechanics
        self.curiosity = curiosity_model
        self.surprise_detector = surprise_detector
        self.config = config or ExplorationConfig()

        self.logger = TrajectoryLogger(
            output_dir=self.config.log_path,
            snapshot_frequency=self.config.snapshot_frequency,
        )

        self.step_count = 0
        self.surprise_moments: List[SurpriseRecord] = []
        self._is_running = False
        # Track active mechanics (those that successfully bound to the scene)
        self._active_mechanics: List[Mechanic] = []

    def run(self, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the full exploration loop.

        Args:
            metadata: Additional metadata to include in logs

        Returns:
            Complete episode data including statistics
        """
        # Start episode
        mechanic_names = [m.name for m in self.mechanics]
        self.logger.start_episode(
            agent_ids=self.config.agent_ids,
            mechanics_active=mechanic_names,
            metadata=metadata,
        )

        self._is_running = True
        self.step_count = 0

        # Reset and bind mechanics to the scene
        self._active_mechanics = []
        for mechanic in self.mechanics:
            mechanic.reset()

            # For scene-aware mechanics, attempt to bind to the current scene
            if isinstance(mechanic, SceneAwareMechanic):
                if mechanic.bind_to_scene(self.world):
                    self._active_mechanics.append(mechanic)
                    # Log what the mechanic bound to
                    debug_state = mechanic.get_hidden_state_for_debug()
                    self.logger.log_message(
                        f"Mechanic '{mechanic.name}' bound to targets: {debug_state.get('bound_targets', [])}"
                    )
                else:
                    # Mechanic couldn't find suitable objects - it stays inactive
                    self.logger.log_message(
                        f"Mechanic '{mechanic.name}' found no suitable objects - inactive"
                    )
            else:
                # Non-scene-aware mechanics are always active
                self._active_mechanics.append(mechanic)

        # Log active mechanics
        active_names = [m.name for m in self._active_mechanics]
        self.logger.log_message(f"Active mechanics for this episode: {active_names}")

        # Main loop
        while self._is_running and self.step_count < self.config.max_steps:
            step_result = self._run_step()

            if step_result.is_terminal and self.config.stop_on_terminal:
                break

            self.step_count += 1

        # Finalize and return
        return self.logger.finalize_episode()

    def _run_step(self) -> StepResult:
        """Execute a single exploration step."""
        agent_actions: Dict[str, ActionChoice] = {}
        action_results: Dict[str, ActionResult] = {}
        step_surprises: List[SurpriseRecord] = []

        # Each agent takes an action
        for agent_id in self.config.agent_ids:
            # Get world description for this agent
            world_text = self.world.to_text(agent_id)
            available_actions = self._get_available_actions(agent_id)
            recent_history = self.logger.get_recent_actions(agent_id, n=5)

            # Select action via curiosity
            action_choice = self.curiosity.select_action(
                agent_id=agent_id,
                world_description=world_text,
                available_actions=available_actions,
                exploration_history=recent_history,
            )
            agent_actions[agent_id] = action_choice

            # Execute action through mechanics
            result = self._execute_action(agent_id, action_choice)
            action_results[agent_id] = result

            # Detect surprise
            if result.surprise_triggers.get(agent_id):
                surprise_assessment = self.surprise_detector.assess_surprise(
                    agent_id=agent_id,
                    action=action_choice.action,
                    target=action_choice.target,
                    expected=action_choice.expected_outcome,
                    actual=result.observations.get(agent_id, ""),
                    trigger=result.surprise_triggers.get(agent_id),
                )

                if surprise_assessment.is_surprised:
                    surprise_record = SurpriseRecord(
                        step=self.step_count,
                        agent_id=agent_id,
                        action=action_choice.action,
                        target=action_choice.target,
                        surprise_level=surprise_assessment.level,
                        explanation=surprise_assessment.explanation,
                        hypothesis=surprise_assessment.hypothesis,
                    )
                    step_surprises.append(surprise_record)
                    self.surprise_moments.append(surprise_record)

        # Process delayed effects
        ready_effects = self.world.advance_step()
        for effect in ready_effects:
            self.world.apply_effect(effect)

        # Log the step
        self.logger.log_step(
            step=self.step_count,
            agent_actions={
                aid: {
                    "action": ac.action,
                    "target": ac.target,
                    "reasoning": ac.reasoning,
                    "expected_outcome": ac.expected_outcome,
                }
                for aid, ac in agent_actions.items()
            },
            effects=[e.to_dict() for r in action_results.values() for e in r.effects],
            observations={
                aid: r.observations.get(aid, "")
                for aid, r in action_results.items()
            },
            surprises=step_surprises,
            world_snapshot=(
                self.world.snapshot()
                if self.config.snapshot_frequency > 0
                and self.step_count % self.config.snapshot_frequency == 0
                else None
            ),
        )

        return StepResult(
            step=self.step_count,
            agent_actions=agent_actions,
            action_results=action_results,
            surprises=step_surprises,
            is_terminal=False,  # Could add terminal state detection
        )

    def _execute_action(
        self, agent_id: str, action_choice: ActionChoice
    ) -> ActionResult:
        """Execute an action, applying mechanics."""
        action_name = action_choice.action
        target = action_choice.target

        # Handle special actions that don't go through mechanics
        if action_name == "wait":
            return ActionResult(
                success=True,
                observations={agent_id: "You wait and observe."},
            )

        if action_name == "move" and target:
            return self._handle_move(agent_id, target)

        if action_name == "look":
            description = self.world.to_text(agent_id)
            return ActionResult(
                success=True,
                observations={agent_id: description},
            )

        # Check if target exists
        if target and not self.world.get_entity(target):
            return ActionResult(
                success=False,
                error_message=f"Target '{target}' not found",
                observations={agent_id: f"You don't see any '{target}' here."},
            )

        # Find applicable mechanic (only check active mechanics)
        for mechanic in self._active_mechanics:
            if mechanic.applies_to(action_name, target, self.world):
                # Create intended effect
                intended = create_default_effect(action_name, target, self.world)
                # Let mechanic transform it
                result = mechanic.transform_effect(
                    action_name, agent_id, target, intended, self.world
                )
                # Apply immediate effects
                for effect in result.effects:
                    self.world.apply_effect(effect)
                # Queue delayed effects
                for effect in result.pending_effects:
                    self.world.add_pending_effect(effect)
                return result

        # No mechanic applied - use default behavior
        return self._default_action(agent_id, action_name, target)

    def _default_action(
        self, agent_id: str, action_name: str, target: Optional[str]
    ) -> ActionResult:
        """Handle action with default behavior (no mechanic)."""
        if not target:
            return ActionResult(
                success=False,
                observations={agent_id: f"You need to specify a target for {action_name}."},
            )

        effect = create_default_effect(action_name, target, self.world)
        self.world.apply_effect(effect)

        observation = effect.description or f"You {action_name} the {target}."
        return ActionResult(
            success=True,
            effects=[effect],
            observations={agent_id: observation},
        )

    def _handle_move(self, agent_id: str, destination: str) -> ActionResult:
        """Handle agent movement between rooms."""
        rooms = self.world.get_room_ids()

        # Check if destination is valid
        if destination not in rooms:
            return ActionResult(
                success=False,
                error_message=f"Unknown room: {destination}",
                observations={agent_id: f"You don't know how to get to '{destination}'."},
            )

        # Move agent
        old_location = self.world.get_agent_location(agent_id)
        self.world.move_entity(agent_id, destination)

        effect = Effect(
            target=agent_id,
            property_changed="location",
            old_value=old_location,
            new_value=destination,
            visible_to={agent_id},
            description=f"Moved from {old_location} to {destination}",
        )

        room = self.world.get_entity(destination)
        room_name = room.get_property("name", destination) if room else destination
        observation = f"You move to {room_name}.\n{self.world.to_text(agent_id)}"

        return ActionResult(
            success=True,
            effects=[effect],
            observations={agent_id: observation},
        )

    def _get_available_actions(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get list of available actions for an agent."""
        actions = []

        # Always available
        actions.append({
            "name": "wait",
            "description": "Do nothing for one step",
            "targets": [],
        })
        actions.append({
            "name": "look",
            "description": "Look around and observe the environment",
            "targets": [],
        })

        # Movement
        rooms = self.world.get_room_ids()
        current_room = self.world.get_agent_location(agent_id)
        other_rooms = [r for r in rooms if r != current_room]
        if other_rooms:
            actions.append({
                "name": "move",
                "description": "Move to another room",
                "targets": other_rooms,
            })

        # Get interactable objects in current room
        if current_room:
            room_entities = self.world.get_entities_in_location(current_room)
            for entity in room_entities:
                if entity.id == agent_id:
                    continue

                # Add appropriate actions based on entity type
                if entity.entity_type == "door":
                    is_open = entity.get_property("is_open", False)
                    actions.append({
                        "name": "close" if is_open else "open",
                        "description": f"{'Close' if is_open else 'Open'} the {entity.id}",
                        "targets": [entity.id],
                    })

                elif entity.entity_type == "switch":
                    actions.append({
                        "name": "toggle",
                        "description": f"Flip the {entity.id}",
                        "targets": [entity.id],
                    })

                elif entity.entity_type == "button":
                    actions.append({
                        "name": "press",
                        "description": f"Press the {entity.id}",
                        "targets": [entity.id],
                    })

                elif entity.entity_type in ("light", "lamp"):
                    is_on = entity.get_property("is_on", False)
                    actions.append({
                        "name": "turn_off" if is_on else "turn_on",
                        "description": f"Turn {'off' if is_on else 'on'} the {entity.id}",
                        "targets": [entity.id],
                    })

                elif entity.entity_type == "object":
                    # Generic object interactions
                    actions.append({
                        "name": "examine",
                        "description": f"Examine the {entity.id}",
                        "targets": [entity.id],
                    })

        return actions

    def stop(self):
        """Stop the exploration loop."""
        self._is_running = False

    def get_surprise_count(self) -> int:
        """Get total number of surprises detected."""
        return len(self.surprise_moments)

    def get_step_count(self) -> int:
        """Get current step count."""
        return self.step_count


def run_exploration(
    world_state: TextWorldState,
    mechanics: List[Mechanic],
    llm_client: Any,
    config: Optional[ExplorationConfig] = None,
    curiosity_type: str = "llm",
    surprise_type: str = "llm",
) -> Dict[str, Any]:
    """
    Convenience function to run exploration with standard setup.

    Args:
        world_state: Initial world state
        mechanics: List of mechanics to apply
        llm_client: LLM client for curiosity and surprise detection
        config: Exploration configuration
        curiosity_type: "llm" or "random"
        surprise_type: "llm", "rules", or "hybrid"

    Returns:
        Episode data from the exploration
    """
    from emtom.exploration.curiosity import create_curiosity_model
    from emtom.exploration.surprise_detector import create_surprise_detector

    curiosity = create_curiosity_model(curiosity_type, llm_client=llm_client)
    surprise = create_surprise_detector(surprise_type, llm_client=llm_client)

    explorer = ExplorationLoop(
        world_state=world_state,
        mechanics=mechanics,
        curiosity_model=curiosity,
        surprise_detector=surprise,
        config=config,
    )

    return explorer.run()
