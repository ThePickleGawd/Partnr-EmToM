"""
Task generator for EMTOM benchmark.

Generates collaborative challenges where agents must work together
to achieve goal states, leveraging discovered mechanics that create
opportunities for theory of mind reasoning.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from emtom.task_gen.trajectory_analyzer import TrajectoryAnalysis, DiscoveredMechanic


class TaskCategory(Enum):
    """Categories of collaborative tasks."""

    COORDINATION = "coordination"  # Agents must coordinate actions
    KNOWLEDGE_ASYMMETRY = "knowledge_asymmetry"  # One agent knows something others don't
    COMMUNICATION = "communication"  # Agents must share information to succeed
    SEQUENTIAL = "sequential"  # Tasks must be done in order
    RESOURCE_SHARING = "resource_sharing"  # Agents share limited resources/abilities


@dataclass
class Subtask:
    """A subtask within a larger challenge."""

    subtask_id: str
    description: str
    success_condition: Dict[str, Any]  # e.g., {"entity": "light_1", "property": "is_on", "value": True}
    assigned_agent: Optional[str] = None  # None = any agent can do it
    depends_on: List[str] = field(default_factory=list)  # IDs of prerequisite subtasks
    hints: List[str] = field(default_factory=list)  # Optional hints about mechanics

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Subtask":
        return cls(
            subtask_id=data["subtask_id"],
            description=data["description"],
            success_condition=data["success_condition"],
            assigned_agent=data.get("assigned_agent"),
            depends_on=data.get("depends_on", []),
            hints=data.get("hints", []),
        )


@dataclass
class SuccessCondition:
    """Defines what success looks like for a task."""

    description: str
    required_states: List[Dict[str, Any]]  # List of entity states that must be true
    time_limit: Optional[int] = None  # Max steps allowed
    all_agents_must_survive: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SuccessCondition":
        return cls(
            description=data["description"],
            required_states=data["required_states"],
            time_limit=data.get("time_limit"),
            all_agents_must_survive=data.get("all_agents_must_survive", True),
        )


@dataclass
class FailureCondition:
    """Defines what causes task failure."""

    description: str
    failure_states: List[Dict[str, Any]] = field(default_factory=list)
    max_failed_attempts: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FailureCondition":
        return cls(
            description=data["description"],
            failure_states=data.get("failure_states", []),
            max_failed_attempts=data.get("max_failed_attempts"),
        )


@dataclass
class GeneratedTask:
    """A collaborative challenge task."""

    task_id: str
    title: str
    category: TaskCategory
    description: str  # High-level description of the challenge

    # World setup
    initial_world_state: Dict[str, Any]  # Entity states at start
    required_mechanics: List[str]  # Which mechanics must be active

    # Agent setup
    num_agents: int
    agent_roles: Dict[str, str]  # agent_id -> role description
    agent_knowledge: Dict[str, List[str]]  # agent_id -> what they know at start

    # Task structure
    subtasks: List[Subtask]
    success_condition: SuccessCondition
    failure_conditions: List[FailureCondition]

    # Metadata
    difficulty: int  # 1-5
    estimated_steps: int
    theory_of_mind_required: bool
    communication_required: bool
    source_trajectory: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d["category"] = self.category.value
        return d

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeneratedTask":
        """Create task from dictionary."""
        return cls(
            task_id=data["task_id"],
            title=data["title"],
            category=TaskCategory(data["category"]),
            description=data["description"],
            initial_world_state=data["initial_world_state"],
            required_mechanics=data["required_mechanics"],
            num_agents=data["num_agents"],
            agent_roles=data["agent_roles"],
            agent_knowledge=data["agent_knowledge"],
            subtasks=[Subtask.from_dict(s) for s in data["subtasks"]],
            success_condition=SuccessCondition.from_dict(data["success_condition"]),
            failure_conditions=[FailureCondition.from_dict(f) for f in data["failure_conditions"]],
            difficulty=data["difficulty"],
            estimated_steps=data["estimated_steps"],
            theory_of_mind_required=data.get("theory_of_mind_required", False),
            communication_required=data.get("communication_required", False),
            source_trajectory=data.get("source_trajectory"),
        )


class TaskGenerator:
    """
    Generates collaborative challenge tasks from trajectory analyses.

    Tasks are designed to require:
    - Multiple agents working together
    - Theory of mind reasoning about what others know
    - Communication to share discovered mechanic behaviors
    - Planning around unexpected world behaviors
    """

    def __init__(self, llm_client: Any = None):
        self.llm = llm_client

    def generate_tasks(
        self,
        trajectory: Dict[str, Any],
        analysis: TrajectoryAnalysis,
        num_agents: int = 2,
        max_tasks: int = 5,
    ) -> List[GeneratedTask]:
        """
        Generate collaborative tasks from a trajectory analysis.

        Args:
            trajectory: Original trajectory dict
            analysis: Analysis with discovered mechanics
            num_agents: Number of agents for tasks
            max_tasks: Maximum tasks to generate

        Returns:
            List of generated challenge tasks
        """
        tasks = []
        mechanics = analysis.discovered_mechanics

        # Generate tasks based on discovered mechanics
        for mechanic in mechanics:
            if len(tasks) >= max_tasks:
                break

            if mechanic.mechanic_type == "inverse":
                tasks.append(self._generate_inverse_task(
                    trajectory, analysis, mechanic, num_agents
                ))

            elif mechanic.mechanic_type == "hidden_mapping":
                tasks.append(self._generate_mapping_task(
                    trajectory, analysis, mechanic, num_agents
                ))

            elif mechanic.mechanic_type == "conditional":
                tasks.append(self._generate_conditional_task(
                    trajectory, analysis, mechanic, num_agents
                ))

            elif mechanic.mechanic_type == "delayed":
                tasks.append(self._generate_timing_task(
                    trajectory, analysis, mechanic, num_agents
                ))

        # Generate combination tasks if multiple mechanics
        if len(mechanics) >= 2 and len(tasks) < max_tasks:
            tasks.append(self._generate_combination_task(
                trajectory, analysis, mechanics[:2], num_agents
            ))

        # Generate knowledge asymmetry task
        if len(tasks) < max_tasks and mechanics:
            tasks.append(self._generate_knowledge_asymmetry_task(
                trajectory, analysis, mechanics[0], num_agents
            ))

        return tasks[:max_tasks]

    def _generate_inverse_task(
        self,
        trajectory: Dict,
        analysis: TrajectoryAnalysis,
        mechanic: DiscoveredMechanic,
        num_agents: int,
    ) -> GeneratedTask:
        """Generate task around inverse mechanics."""

        return GeneratedTask(
            task_id=f"task_inverse_{uuid.uuid4().hex[:8]}",
            title="The Backwards House",
            category=TaskCategory.KNOWLEDGE_ASYMMETRY,
            description=(
                "All doors in this house work backwards - 'open' closes them and "
                "'close' opens them. Agent A has discovered this through trial and error. "
                "Agent B just arrived and doesn't know. Together, they must open all doors "
                "to let fresh air through the house."
            ),
            initial_world_state={
                "doors": [
                    {"id": "door_1", "location": "room_a", "is_open": False},
                    {"id": "door_2", "location": "room_b", "is_open": False},
                    {"id": "door_3", "location": "room_c", "is_open": False},
                ],
                "agent_positions": {
                    "agent_0": "room_a",
                    "agent_1": "room_b",
                },
            },
            required_mechanics=["inverse_open"],
            num_agents=num_agents,
            agent_roles={
                "agent_0": "Experienced - has learned doors work backwards",
                "agent_1": "Newcomer - assumes normal door behavior",
            },
            agent_knowledge={
                "agent_0": [
                    "Doors in this house work backwards",
                    "Use 'close' to open a door",
                ],
                "agent_1": [],  # Knows nothing special
            },
            subtasks=[
                Subtask(
                    subtask_id="open_door_1",
                    description="Open door_1 in room_a",
                    success_condition={"entity": "door_1", "property": "is_open", "value": True},
                    hints=["This door works backwards"],
                ),
                Subtask(
                    subtask_id="open_door_2",
                    description="Open door_2 in room_b",
                    success_condition={"entity": "door_2", "property": "is_open", "value": True},
                    hints=["Agent B might try 'open' first - that won't work"],
                ),
                Subtask(
                    subtask_id="open_door_3",
                    description="Open door_3 in room_c",
                    success_condition={"entity": "door_3", "property": "is_open", "value": True},
                    depends_on=["open_door_1"],  # Need door_1 open to access room_c
                ),
            ],
            success_condition=SuccessCondition(
                description="All three doors are open",
                required_states=[
                    {"entity": "door_1", "property": "is_open", "value": True},
                    {"entity": "door_2", "property": "is_open", "value": True},
                    {"entity": "door_3", "property": "is_open", "value": True},
                ],
                time_limit=20,
            ),
            failure_conditions=[
                FailureCondition(
                    description="Too many failed attempts",
                    max_failed_attempts=10,
                ),
            ],
            difficulty=2,
            estimated_steps=10,
            theory_of_mind_required=True,
            communication_required=True,
            source_trajectory=analysis.episode_id,
        )

    def _generate_mapping_task(
        self,
        trajectory: Dict,
        analysis: TrajectoryAnalysis,
        mechanic: DiscoveredMechanic,
        num_agents: int,
    ) -> GeneratedTask:
        """Generate task around hidden mapping mechanics."""

        return GeneratedTask(
            task_id=f"task_mapping_{uuid.uuid4().hex[:8]}",
            title="Remote Control Chaos",
            category=TaskCategory.COMMUNICATION,
            description=(
                "The house has a strange wiring system - switches don't control "
                "the lights in their own room! Agent A is in the living room with a switch. "
                "Agent B is in the kitchen and can see the kitchen light. "
                "They need to figure out the mapping and turn on all lights."
            ),
            initial_world_state={
                "switches": [
                    {"id": "switch_living", "location": "living_room", "is_on": False, "controls": "light_kitchen"},
                    {"id": "switch_kitchen", "location": "kitchen", "is_on": False, "controls": "light_living"},
                ],
                "lights": [
                    {"id": "light_living", "location": "living_room", "is_on": False},
                    {"id": "light_kitchen", "location": "kitchen", "is_on": False},
                ],
                "agent_positions": {
                    "agent_0": "living_room",
                    "agent_1": "kitchen",
                },
            },
            required_mechanics=["remote_switch"],
            num_agents=num_agents,
            agent_roles={
                "agent_0": "In living room - can operate switch_living, can see light_living",
                "agent_1": "In kitchen - can operate switch_kitchen, can see light_kitchen",
            },
            agent_knowledge={
                "agent_0": ["There's a switch here but I don't know what it controls"],
                "agent_1": ["There's a switch here but I don't know what it controls"],
            },
            subtasks=[
                Subtask(
                    subtask_id="discover_mapping_1",
                    description="Discover what switch_living controls",
                    success_condition={"discovered": "switch_living -> light_kitchen"},
                    hints=["Try the switch and have Agent B report what happens"],
                ),
                Subtask(
                    subtask_id="discover_mapping_2",
                    description="Discover what switch_kitchen controls",
                    success_condition={"discovered": "switch_kitchen -> light_living"},
                    hints=["Agent B should try their switch while Agent A watches"],
                ),
                Subtask(
                    subtask_id="turn_on_all",
                    description="Turn on both lights",
                    success_condition={"all_lights_on": True},
                    depends_on=["discover_mapping_1", "discover_mapping_2"],
                ),
            ],
            success_condition=SuccessCondition(
                description="Both lights are on",
                required_states=[
                    {"entity": "light_living", "property": "is_on", "value": True},
                    {"entity": "light_kitchen", "property": "is_on", "value": True},
                ],
                time_limit=15,
            ),
            failure_conditions=[
                FailureCondition(
                    description="Agents give up without turning on lights",
                ),
            ],
            difficulty=3,
            estimated_steps=8,
            theory_of_mind_required=True,
            communication_required=True,
            source_trajectory=analysis.episode_id,
        )

    def _generate_conditional_task(
        self,
        trajectory: Dict,
        analysis: TrajectoryAnalysis,
        mechanic: DiscoveredMechanic,
        num_agents: int,
    ) -> GeneratedTask:
        """Generate task around conditional/counting mechanics."""

        return GeneratedTask(
            task_id=f"task_conditional_{uuid.uuid4().hex[:8]}",
            title="The Stubborn Button",
            category=TaskCategory.COORDINATION,
            description=(
                "There's a security button that requires exactly 3 presses to activate. "
                "But there's a catch - only one agent can be in the button room at a time, "
                "and the button resets if too much time passes between presses. "
                "Agents must coordinate their entries to press the button in sequence."
            ),
            initial_world_state={
                "buttons": [
                    {"id": "security_button", "location": "vault_room", "press_count": 0, "is_active": False, "required_presses": 3},
                ],
                "rooms": [
                    {"id": "waiting_room", "capacity": 2},
                    {"id": "vault_room", "capacity": 1},  # Only one agent at a time
                ],
                "agent_positions": {
                    "agent_0": "waiting_room",
                    "agent_1": "waiting_room",
                },
            },
            required_mechanics=["counting_trigger"],
            num_agents=num_agents,
            agent_roles={
                "agent_0": "First responder - goes in first",
                "agent_1": "Backup - enters when agent_0 leaves",
            },
            agent_knowledge={
                "agent_0": ["The button needs multiple presses"],
                "agent_1": ["The button needs multiple presses"],
            },
            subtasks=[
                Subtask(
                    subtask_id="first_press",
                    description="Agent 0 enters vault and presses button once",
                    success_condition={"entity": "security_button", "property": "press_count", "value": 1},
                    assigned_agent="agent_0",
                ),
                Subtask(
                    subtask_id="swap_agents",
                    description="Agent 0 leaves, Agent 1 enters",
                    success_condition={"agent_1_in_vault": True, "agent_0_in_waiting": True},
                    depends_on=["first_press"],
                ),
                Subtask(
                    subtask_id="second_press",
                    description="Agent 1 presses button",
                    success_condition={"entity": "security_button", "property": "press_count", "value": 2},
                    assigned_agent="agent_1",
                    depends_on=["swap_agents"],
                ),
                Subtask(
                    subtask_id="third_press",
                    description="One more press to activate",
                    success_condition={"entity": "security_button", "property": "is_active", "value": True},
                    depends_on=["second_press"],
                ),
            ],
            success_condition=SuccessCondition(
                description="Security button is activated",
                required_states=[
                    {"entity": "security_button", "property": "is_active", "value": True},
                ],
                time_limit=12,
            ),
            failure_conditions=[
                FailureCondition(
                    description="Button resets due to timeout between presses",
                    failure_states=[{"entity": "security_button", "property": "press_count", "value": 0}],
                ),
                FailureCondition(
                    description="Both agents try to enter vault at once",
                ),
            ],
            difficulty=3,
            estimated_steps=10,
            theory_of_mind_required=False,
            communication_required=True,
            source_trajectory=analysis.episode_id,
        )

    def _generate_timing_task(
        self,
        trajectory: Dict,
        analysis: TrajectoryAnalysis,
        mechanic: DiscoveredMechanic,
        num_agents: int,
    ) -> GeneratedTask:
        """Generate task around delayed effect mechanics."""

        return GeneratedTask(
            task_id=f"task_timing_{uuid.uuid4().hex[:8]}",
            title="Delayed Reactions",
            category=TaskCategory.SEQUENTIAL,
            description=(
                "In this house, some actions take time to have effect. "
                "When you flip a switch, the light doesn't turn on immediately - "
                "it takes 3 seconds. Agents must learn this timing and coordinate "
                "to turn on lights in the correct sequence for a security system."
            ),
            initial_world_state={
                "switches": [
                    {"id": "switch_1", "location": "room_a", "delay": 3},
                    {"id": "switch_2", "location": "room_b", "delay": 3},
                ],
                "lights": [
                    {"id": "light_1", "location": "room_a", "is_on": False, "must_be_on_before": "light_2"},
                    {"id": "light_2", "location": "room_b", "is_on": False},
                ],
                "agent_positions": {
                    "agent_0": "room_a",
                    "agent_1": "room_b",
                },
            },
            required_mechanics=["delayed_effect"],
            num_agents=num_agents,
            agent_roles={
                "agent_0": "Controls switch_1 for light_1",
                "agent_1": "Controls switch_2 for light_2",
            },
            agent_knowledge={
                "agent_0": [],
                "agent_1": [],
            },
            subtasks=[
                Subtask(
                    subtask_id="discover_delay",
                    description="Discover that switches have delayed effects",
                    success_condition={"discovered": "delay_mechanic"},
                ),
                Subtask(
                    subtask_id="coordinate_timing",
                    description="Agent 0 flips switch first, waits, then Agent 1 flips",
                    success_condition={"light_1_on_before_light_2": True},
                    depends_on=["discover_delay"],
                ),
            ],
            success_condition=SuccessCondition(
                description="Lights turn on in correct sequence (light_1 before light_2)",
                required_states=[
                    {"entity": "light_1", "property": "is_on", "value": True},
                    {"entity": "light_2", "property": "is_on", "value": True},
                    {"sequence_correct": True},
                ],
                time_limit=20,
            ),
            failure_conditions=[
                FailureCondition(
                    description="Lights turn on in wrong order",
                    failure_states=[{"sequence_correct": False}],
                ),
            ],
            difficulty=4,
            estimated_steps=12,
            theory_of_mind_required=False,
            communication_required=True,
            source_trajectory=analysis.episode_id,
        )

    def _generate_combination_task(
        self,
        trajectory: Dict,
        analysis: TrajectoryAnalysis,
        mechanics: List[DiscoveredMechanic],
        num_agents: int,
    ) -> GeneratedTask:
        """Generate task combining multiple mechanics."""

        mechanic_names = [m.mechanic_type for m in mechanics]

        return GeneratedTask(
            task_id=f"task_combo_{uuid.uuid4().hex[:8]}",
            title="The Puzzle House",
            category=TaskCategory.COORDINATION,
            description=(
                f"This house combines multiple strange behaviors: {', '.join(mechanic_names)}. "
                "Agents must work together, sharing what they learn about each mechanic, "
                "to successfully navigate the house and reach the goal room."
            ),
            initial_world_state={
                "mechanics_active": mechanic_names,
                "goal": {"reach_room": "treasure_room"},
                "agent_positions": {
                    "agent_0": "start_room",
                    "agent_1": "start_room",
                },
            },
            required_mechanics=[m.mechanic_type for m in mechanics],
            num_agents=num_agents,
            agent_roles={
                "agent_0": "Explorer - tries actions and reports findings",
                "agent_1": "Planner - coordinates based on discoveries",
            },
            agent_knowledge={
                "agent_0": [],
                "agent_1": [],
            },
            subtasks=[
                Subtask(
                    subtask_id=f"discover_{m.mechanic_type}",
                    description=f"Discover and understand the {m.mechanic_type} mechanic",
                    success_condition={"discovered": m.mechanic_type},
                )
                for m in mechanics
            ] + [
                Subtask(
                    subtask_id="reach_goal",
                    description="Both agents reach the treasure room",
                    success_condition={"all_agents_in": "treasure_room"},
                    depends_on=[f"discover_{m.mechanic_type}" for m in mechanics],
                ),
            ],
            success_condition=SuccessCondition(
                description="Both agents reach the treasure room",
                required_states=[
                    {"entity": "agent_0", "property": "location", "value": "treasure_room"},
                    {"entity": "agent_1", "property": "location", "value": "treasure_room"},
                ],
                time_limit=30,
            ),
            failure_conditions=[
                FailureCondition(
                    description="Agents get stuck or separated",
                    max_failed_attempts=15,
                ),
            ],
            difficulty=4,
            estimated_steps=20,
            theory_of_mind_required=True,
            communication_required=True,
            source_trajectory=analysis.episode_id,
        )

    def _generate_knowledge_asymmetry_task(
        self,
        trajectory: Dict,
        analysis: TrajectoryAnalysis,
        mechanic: DiscoveredMechanic,
        num_agents: int,
    ) -> GeneratedTask:
        """Generate task where agents have different knowledge."""

        return GeneratedTask(
            task_id=f"task_asymmetry_{uuid.uuid4().hex[:8]}",
            title="The Expert and the Novice",
            category=TaskCategory.KNOWLEDGE_ASYMMETRY,
            description=(
                f"Agent A has spent time in this house and learned about the "
                f"{mechanic.mechanic_type} mechanic. Agent B just arrived and knows nothing. "
                "Agent B must complete a task, but can only succeed if Agent A shares "
                "their knowledge. Agent A cannot directly help - only communicate."
            ),
            initial_world_state={
                "mechanic": mechanic.mechanic_type,
                "target_object": "mystery_device",
                "agent_positions": {
                    "agent_0": "observation_room",  # Can see but not touch
                    "agent_1": "action_room",  # Can act but doesn't know how
                },
            },
            required_mechanics=[mechanic.mechanic_type],
            num_agents=num_agents,
            agent_roles={
                "agent_0": "Expert observer - knows the mechanic, cannot act directly",
                "agent_1": "Novice actor - must perform actions based on guidance",
            },
            agent_knowledge={
                "agent_0": [
                    f"The {mechanic.mechanic_type} mechanic: {mechanic.description}",
                    "You can see Agent B but cannot enter their room",
                    "You must guide them through communication",
                ],
                "agent_1": [
                    "There's a device you need to activate",
                    "Agent A might know something useful",
                ],
            },
            subtasks=[
                Subtask(
                    subtask_id="establish_communication",
                    description="Agent A contacts Agent B",
                    success_condition={"communication_established": True},
                ),
                Subtask(
                    subtask_id="share_knowledge",
                    description="Agent A explains the mechanic to Agent B",
                    success_condition={"knowledge_transferred": True},
                    depends_on=["establish_communication"],
                ),
                Subtask(
                    subtask_id="apply_knowledge",
                    description="Agent B uses the knowledge to activate the device",
                    success_condition={"entity": "mystery_device", "property": "is_active", "value": True},
                    assigned_agent="agent_1",
                    depends_on=["share_knowledge"],
                ),
            ],
            success_condition=SuccessCondition(
                description="Agent B activates the mystery device using learned knowledge",
                required_states=[
                    {"entity": "mystery_device", "property": "is_active", "value": True},
                ],
                time_limit=15,
            ),
            failure_conditions=[
                FailureCondition(
                    description="Agent B acts without understanding",
                    max_failed_attempts=5,
                ),
                FailureCondition(
                    description="Agents fail to communicate effectively",
                ),
            ],
            difficulty=3,
            estimated_steps=10,
            theory_of_mind_required=True,
            communication_required=True,
            source_trajectory=analysis.episode_id,
        )
