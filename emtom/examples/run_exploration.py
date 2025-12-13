#!/usr/bin/env python3
"""
Example script for running EMTOM exploration.

Demonstrates the exploration loop with sample mechanics,
random action selection, and rule-based surprise detection.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from emtom.core.world_state import Entity, TextWorldState, create_simple_world
from emtom.exploration import (
    ExplorationConfig,
    ExplorationLoop,
    RandomCuriosityModel,
    RuleBasedSurpriseDetector,
    ScriptedCuriosityModel,
)
from emtom.mechanics.registry import MechanicRegistry

# Import mechanics to register them
from emtom.mechanics import inverse_open, remote_switch, counting_trigger


def create_test_world() -> TextWorldState:
    """Create a simple test world with various objects."""
    world = create_simple_world(
        rooms=["living_room", "kitchen", "bedroom"],
        agents=["agent_0"],
    )

    # Add doors
    world.add_entity(Entity(
        id="front_door",
        entity_type="door",
        properties={"is_open": False, "name": "Front Door"},
        location="living_room",
    ))
    world.add_entity(Entity(
        id="kitchen_door",
        entity_type="door",
        properties={"is_open": False, "name": "Kitchen Door"},
        location="kitchen",
    ))

    # Add switches
    world.add_entity(Entity(
        id="switch_1",
        entity_type="switch",
        properties={"is_on": False, "name": "Wall Switch 1"},
        location="living_room",
    ))
    world.add_entity(Entity(
        id="switch_2",
        entity_type="switch",
        properties={"is_on": False, "name": "Wall Switch 2"},
        location="kitchen",
    ))

    # Add lights
    world.add_entity(Entity(
        id="living_light",
        entity_type="light",
        properties={"is_on": False, "name": "Living Room Light"},
        location="living_room",
    ))
    world.add_entity(Entity(
        id="kitchen_light",
        entity_type="light",
        properties={"is_on": False, "name": "Kitchen Light"},
        location="kitchen",
    ))

    # Add button
    world.add_entity(Entity(
        id="red_button",
        entity_type="button",
        properties={"is_pressed": False, "is_active": False, "name": "Red Button"},
        location="bedroom",
    ))

    return world


def run_random_exploration(
    steps: int = 20,
    seed: int = 42,
    output_dir: str = "data/trajectories/emtom",
):
    """Run exploration with random action selection."""
    print("=== EMTOM Random Exploration Demo ===\n")

    # Create world
    world = create_test_world()
    print("Created test world with:")
    print(f"  Rooms: {world.get_room_ids()}")
    print(f"  Entities: {list(world.entities.keys())}")
    print()

    # Setup mechanics
    mechanics = [
        MechanicRegistry.instantiate("inverse_open"),
        MechanicRegistry.instantiate("remote_switch", mappings={
            "switch_1": "kitchen_light",  # switch in living room controls kitchen light
            "switch_2": "living_light",   # switch in kitchen controls living room light
        }),
        MechanicRegistry.instantiate("counting_trigger", required_count=3),
    ]
    print("Active mechanics:")
    for m in mechanics:
        print(f"  - {m.name}: {m.description}")
    print()

    # Setup exploration
    curiosity = RandomCuriosityModel(seed=seed)
    surprise = RuleBasedSurpriseDetector()

    config = ExplorationConfig(
        max_steps=steps,
        agent_ids=["agent_0"],
        log_path=output_dir,
    )

    explorer = ExplorationLoop(
        world_state=world,
        mechanics=mechanics,
        curiosity_model=curiosity,
        surprise_detector=surprise,
        config=config,
    )

    # Run exploration
    print(f"Running exploration for {steps} steps...\n")
    episode_data = explorer.run(metadata={"seed": seed, "mode": "random"})

    # Print summary
    print("\n=== Exploration Summary ===")
    stats = episode_data["statistics"]
    print(f"Total steps: {stats['total_steps']}")
    print(f"Total surprises: {stats['total_surprises']}")
    print(f"Actions per agent: {stats['actions_per_agent']}")
    print(f"Unique actions: {stats['unique_actions']}")

    if episode_data["surprise_summary"]:
        print("\nSurprise moments:")
        for surprise in episode_data["surprise_summary"]:
            print(f"  Step {surprise['step']}: {surprise['action']} on {surprise['target']}")
            print(f"    Level: {surprise['surprise_level']}/5")
            print(f"    Explanation: {surprise['explanation']}")
            if surprise.get("hypothesis"):
                print(f"    Hypothesis: {surprise['hypothesis']}")

    # Print trajectory file location
    print(f"\nTrajectory saved to: {output_dir}/trajectory_{episode_data['episode_id']}.json")

    return episode_data


def run_scripted_exploration(output_dir: str = "data/trajectories/emtom"):
    """Run exploration with a scripted action sequence to demonstrate mechanics."""
    print("=== EMTOM Scripted Exploration Demo ===\n")

    # Create world
    world = create_test_world()

    # Setup mechanics
    mechanics = [
        MechanicRegistry.instantiate("inverse_open"),
        MechanicRegistry.instantiate("remote_switch", mappings={
            "switch_1": "kitchen_light",
            "switch_2": "living_light",
        }),
        MechanicRegistry.instantiate("counting_trigger", required_count=3),
    ]

    # Define scripted actions to demonstrate each mechanic
    script = [
        # Test inverse open mechanic
        {"action": "open", "target": "front_door", "expected": "door opens"},
        {"action": "close", "target": "front_door", "expected": "door closes"},

        # Test remote switch mechanic
        {"action": "toggle", "target": "switch_1", "expected": "local effect"},
        {"action": "move", "target": "kitchen"},
        {"action": "look", "target": None},  # Observe the kitchen light

        # Test counting trigger
        {"action": "move", "target": "bedroom"},
        {"action": "press", "target": "red_button", "expected": "button activates"},
        {"action": "press", "target": "red_button", "expected": "button activates"},
        {"action": "press", "target": "red_button", "expected": "button activates"},
    ]

    curiosity = ScriptedCuriosityModel(script)
    surprise = RuleBasedSurpriseDetector()

    config = ExplorationConfig(
        max_steps=len(script),
        agent_ids=["agent_0"],
        log_path=output_dir,
    )

    explorer = ExplorationLoop(
        world_state=world,
        mechanics=mechanics,
        curiosity_model=curiosity,
        surprise_detector=surprise,
        config=config,
    )

    print("Running scripted exploration to demonstrate mechanics...\n")
    episode_data = explorer.run(metadata={"mode": "scripted"})

    # Print step-by-step results
    print("\n=== Step-by-Step Results ===")
    for step in episode_data["steps"]:
        agent_action = step["agent_actions"].get("agent_0", {})
        action = agent_action.get("action", "unknown")
        target = agent_action.get("target", "")
        observation = step["observations"].get("agent_0", "")

        print(f"\nStep {step['step']}: {action}" + (f" on {target}" if target else ""))
        print(f"  Result: {observation[:100]}..." if len(observation) > 100 else f"  Result: {observation}")

        for surprise in step.get("surprises", []):
            print(f"  [SURPRISE] Level {surprise['surprise_level']}: {surprise['explanation']}")

    return episode_data


def main():
    parser = argparse.ArgumentParser(description="Run EMTOM exploration demo")
    parser.add_argument(
        "--mode",
        choices=["random", "scripted"],
        default="scripted",
        help="Exploration mode (default: scripted)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Number of exploration steps (for random mode)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (for random mode)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/trajectories/emtom",
        help="Output directory for trajectory logs",
    )
    args = parser.parse_args()

    if args.mode == "random":
        run_random_exploration(
            steps=args.steps,
            seed=args.seed,
            output_dir=args.output_dir,
        )
    else:
        run_scripted_exploration(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
