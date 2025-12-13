#!/usr/bin/env python3
"""
Example script for generating collaborative challenge tasks from EMTOM trajectories.

Demonstrates the full pipeline:
1. Run exploration to discover mechanics
2. Analyze trajectory for patterns
3. Generate collaborative challenge tasks with success/failure conditions
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
    ScriptedCuriosityModel,
    RuleBasedSurpriseDetector,
)
from emtom.mechanics.registry import MechanicRegistry
from emtom.task_gen import (
    TrajectoryAnalyzer,
    TaskGenerator,
)

# Import mechanics to register them
from emtom.mechanics import inverse_open, remote_switch, counting_trigger


def create_test_world() -> TextWorldState:
    """Create a test world with various objects."""
    world = create_simple_world(
        rooms=["living_room", "kitchen", "bedroom"],
        agents=["agent_0", "agent_1"],
    )

    # Add doors
    world.add_entity(Entity(
        id="front_door",
        entity_type="door",
        properties={"is_open": False, "name": "Front Door"},
        location="living_room",
    ))

    # Add switches
    world.add_entity(Entity(
        id="switch_1",
        entity_type="switch",
        properties={"is_on": False, "name": "Wall Switch"},
        location="living_room",
    ))

    # Add lights
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


def print_task(task, index: int):
    """Pretty print a generated task."""
    print(f"\n{'='*60}")
    print(f"TASK {index}: {task.title}")
    print(f"{'='*60}")
    print(f"Category: {task.category.value}")
    print(f"Difficulty: {task.difficulty}/5")
    print(f"Estimated steps: {task.estimated_steps}")
    print(f"Theory of Mind required: {task.theory_of_mind_required}")
    print(f"Communication required: {task.communication_required}")

    print(f"\nDESCRIPTION:")
    print(f"  {task.description}")

    print(f"\nAGENT ROLES ({task.num_agents} agents):")
    for agent_id, role in task.agent_roles.items():
        print(f"  {agent_id}: {role}")

    print(f"\nAGENT STARTING KNOWLEDGE:")
    for agent_id, knowledge in task.agent_knowledge.items():
        if knowledge:
            print(f"  {agent_id}:")
            for k in knowledge:
                print(f"    - {k}")
        else:
            print(f"  {agent_id}: (no special knowledge)")

    print(f"\nSUBTASKS ({len(task.subtasks)}):")
    for subtask in task.subtasks:
        deps = f" [depends on: {', '.join(subtask.depends_on)}]" if subtask.depends_on else ""
        assigned = f" [assigned to: {subtask.assigned_agent}]" if subtask.assigned_agent else ""
        print(f"  - {subtask.subtask_id}: {subtask.description}{deps}{assigned}")
        if subtask.hints:
            for hint in subtask.hints:
                print(f"      Hint: {hint}")

    print(f"\nSUCCESS CONDITION:")
    print(f"  {task.success_condition.description}")
    if task.success_condition.time_limit:
        print(f"  Time limit: {task.success_condition.time_limit} steps")
    print(f"  Required states:")
    for state in task.success_condition.required_states:
        print(f"    - {state}")

    print(f"\nFAILURE CONDITIONS:")
    for fc in task.failure_conditions:
        print(f"  - {fc.description}")
        if fc.max_failed_attempts:
            print(f"    Max failed attempts: {fc.max_failed_attempts}")

    print(f"\nREQUIRED MECHANICS: {task.required_mechanics}")


def run_exploration_and_generate_tasks():
    """Run the full pipeline: exploration -> analysis -> task generation."""
    print("=" * 60)
    print("EMTOM Collaborative Task Generation Pipeline")
    print("=" * 60)

    # Step 1: Create world and mechanics
    print("\n[Step 1] Setting up world and mechanics...")
    world = create_test_world()

    mechanics = [
        MechanicRegistry.instantiate("inverse_open"),
        MechanicRegistry.instantiate("remote_switch", mappings={
            "switch_1": "kitchen_light",
        }),
        MechanicRegistry.instantiate("counting_trigger", required_count=3),
    ]

    print(f"  World has {len(world.entities)} entities")
    print(f"  Active mechanics: {[m.name for m in mechanics]}")

    # Step 2: Run scripted exploration
    print("\n[Step 2] Running exploration to discover mechanics...")

    script = [
        {"action": "open", "target": "front_door"},
        {"action": "close", "target": "front_door"},
        {"action": "toggle", "target": "switch_1"},
        {"action": "move", "target": "kitchen"},
        {"action": "look", "target": None},
        {"action": "move", "target": "bedroom"},
        {"action": "press", "target": "red_button"},
        {"action": "press", "target": "red_button"},
        {"action": "press", "target": "red_button"},
    ]

    curiosity = ScriptedCuriosityModel(script)
    surprise_detector = RuleBasedSurpriseDetector()

    config = ExplorationConfig(
        max_steps=len(script),
        agent_ids=["agent_0"],
        log_path="data/trajectories/emtom",
    )

    explorer = ExplorationLoop(
        world_state=world,
        mechanics=mechanics,
        curiosity_model=curiosity,
        surprise_detector=surprise_detector,
        config=config,
    )

    trajectory = explorer.run(metadata={"mode": "task_generation"})
    print(f"  Completed {trajectory['statistics']['total_steps']} steps")
    print(f"  Detected {trajectory['statistics']['total_surprises']} surprises")

    # Step 3: Analyze trajectory
    print("\n[Step 3] Analyzing trajectory for mechanic patterns...")
    analyzer = TrajectoryAnalyzer()
    analysis = analyzer.analyze(trajectory)

    print(f"  Discovered {len(analysis.discovered_mechanics)} mechanics:")
    for m in analysis.discovered_mechanics:
        print(f"    - {m.mechanic_type}: {m.description}")

    # Step 4: Generate collaborative tasks
    print("\n[Step 4] Generating collaborative challenge tasks...")
    generator = TaskGenerator()
    tasks = generator.generate_tasks(
        trajectory=trajectory,
        analysis=analysis,
        num_agents=2,
        max_tasks=5,
    )

    print(f"  Generated {len(tasks)} collaborative tasks")

    # Step 5: Display tasks
    for i, task in enumerate(tasks, 1):
        print_task(task, i)

    # Step 6: Save tasks
    output_file = f"data/tasks/emtom_challenges_{trajectory['episode_id']}.json"
    Path("data/tasks").mkdir(parents=True, exist_ok=True)

    tasks_data = {
        "source_trajectory": trajectory["episode_id"],
        "discovered_mechanics": [m.mechanic_type for m in analysis.discovered_mechanics],
        "tasks": [task.to_dict() for task in tasks],
    }

    with open(output_file, "w") as f:
        json.dump(tasks_data, f, indent=2)

    print(f"\n[Step 6] Tasks saved to: {output_file}")

    return trajectory, analysis, tasks


def main():
    parser = argparse.ArgumentParser(
        description="Generate EMTOM collaborative challenge tasks"
    )
    parser.add_argument(
        "--output-dir",
        default="data/tasks",
        help="Output directory for generated tasks",
    )
    args = parser.parse_args()

    run_exploration_and_generate_tasks()


if __name__ == "__main__":
    main()
