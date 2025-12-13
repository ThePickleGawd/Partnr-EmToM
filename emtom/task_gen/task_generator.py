"""
LLM-based task generator for EMTOM benchmark.

Generates collaborative challenges by feeding trajectory surprises to an LLM
and having it create tasks that leverage the discovered mechanics.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional

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
    success_condition: Dict[str, Any]
    assigned_agent: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)
    hints: List[str] = field(default_factory=list)

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
    required_states: List[Dict[str, Any]]
    time_limit: Optional[int] = None
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
    description: str

    initial_world_state: Dict[str, Any]
    required_mechanics: List[str]

    num_agents: int
    agent_roles: Dict[str, str]
    agent_knowledge: Dict[str, List[str]]

    subtasks: List[Subtask]
    success_condition: SuccessCondition
    failure_conditions: List[FailureCondition]

    difficulty: int
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


TASK_GENERATION_PROMPT = '''You are a task designer for a multi-agent collaboration benchmark.

Given the following exploration trajectory where an agent discovered surprising behaviors in a simulated home environment, generate a collaborative task that leverages these discoveries.

## Trajectory Summary
Scene: {scene_id}
Total Steps: {total_steps}
Active Mechanics: {mechanics}

## Surprises Discovered
{surprises}

## Task Requirements
- Design a task for {num_agents} agents working together
- The task should require agents to understand and use the discovered mechanics
- One agent should have knowledge about the mechanics, the other should not (theory of mind)
- Agents must communicate to succeed
- Use the ACTUAL objects and rooms from the trajectory (e.g., {example_objects})

## Output Format
Respond with a JSON object:
{{
    "title": "Short descriptive title",
    "category": "knowledge_asymmetry" or "coordination" or "communication",
    "description": "2-3 sentence description of the task scenario",
    "initial_world_state": {{
        "objects": ["list of objects involved"],
        "agent_positions": {{"agent_0": "room_name", "agent_1": "room_name"}}
    }},
    "required_mechanics": ["mechanic_names from surprises"],
    "agent_roles": {{
        "agent_0": "Role description - what they know/can do",
        "agent_1": "Role description - what they know/can do"
    }},
    "agent_knowledge": {{
        "agent_0": ["List of things agent_0 knows at start"],
        "agent_1": ["List of things agent_1 knows at start (less than agent_0)"]
    }},
    "subtasks": [
        {{
            "subtask_id": "unique_id",
            "description": "What needs to be done",
            "success_condition": {{"entity": "object_name", "state": "target_state"}}
        }}
    ],
    "success_condition": {{
        "description": "What success looks like",
        "required_states": [{{"entity": "name", "property": "prop", "value": true}}]
    }},
    "failure_conditions": [
        {{"description": "What causes failure"}}
    ],
    "difficulty": 3,
    "estimated_steps": 15
}}

Generate the task JSON:'''


class TaskGenerator:
    """
    LLM-based task generator that creates collaborative challenges
    from trajectory surprises.
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
        Generate collaborative tasks from a trajectory using LLM.

        Args:
            trajectory: Original trajectory dict with steps and surprises
            analysis: Analysis with discovered mechanics
            num_agents: Number of agents for tasks
            max_tasks: Maximum tasks to generate

        Returns:
            List of generated challenge tasks
        """
        if self.llm is None:
            raise ValueError("LLM client required for task generation")

        # Extract surprises from trajectory
        surprises = trajectory.get("surprise_summary", [])
        if not surprises:
            # Also check steps for surprises
            for step in trajectory.get("steps", []):
                surprises.extend(step.get("surprises", []))

        if not surprises:
            print("  WARNING: No surprises found in trajectory")
            return []

        # Get scene info
        scene_id = trajectory.get("metadata", {}).get("scene_id", "unknown")
        total_steps = trajectory.get("statistics", {}).get("total_steps", 0)
        mechanics = trajectory.get("mechanics_active", [])

        # Extract objects from trajectory
        objects_seen = set()
        for step in trajectory.get("steps", []):
            for action in step.get("agent_actions", {}).values():
                if action.get("target"):
                    objects_seen.add(action["target"])
        example_objects = list(objects_seen)[:5]

        # Format surprises for prompt
        surprise_text = self._format_surprises(surprises)

        # Generate tasks
        tasks = []
        for i in range(min(max_tasks, len(surprises))):
            try:
                task = self._generate_single_task(
                    scene_id=scene_id,
                    total_steps=total_steps,
                    mechanics=mechanics,
                    surprises=surprise_text,
                    example_objects=example_objects,
                    num_agents=num_agents,
                    episode_id=trajectory.get("episode_id", "unknown"),
                )
                if task:
                    tasks.append(task)
            except Exception as e:
                print(f"  Failed to generate task {i+1}: {e}")

        return tasks

    def _format_surprises(self, surprises: List[Dict[str, Any]]) -> str:
        """Format surprises for the prompt."""
        lines = []
        for i, s in enumerate(surprises, 1):
            lines.append(f"{i}. Action: {s.get('action', 'unknown')} on {s.get('target', 'unknown')}")
            lines.append(f"   Surprise Level: {s.get('surprise_level', 'N/A')}/5")
            lines.append(f"   What happened: {s.get('explanation', 'No explanation')}")
            lines.append(f"   Hypothesis: {s.get('hypothesis', 'No hypothesis')}")
            lines.append("")
        return "\n".join(lines)

    def _generate_single_task(
        self,
        scene_id: str,
        total_steps: int,
        mechanics: List[str],
        surprises: str,
        example_objects: List[str],
        num_agents: int,
        episode_id: str,
    ) -> Optional[GeneratedTask]:
        """Generate a single task using LLM."""
        prompt = TASK_GENERATION_PROMPT.format(
            scene_id=scene_id,
            total_steps=total_steps,
            mechanics=", ".join(mechanics) if mechanics else "unknown",
            surprises=surprises,
            num_agents=num_agents,
            example_objects=", ".join(example_objects) if example_objects else "various objects",
        )

        response = self.llm.generate(prompt)

        # Parse JSON from response
        try:
            # Find JSON in response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                task_data = json.loads(json_match.group())
                return self._parse_task_response(task_data, episode_id, num_agents)
        except json.JSONDecodeError as e:
            print(f"  Failed to parse LLM response as JSON: {e}")
            print(f"  Response: {response[:500]}...")

        return None

    def _parse_task_response(
        self,
        data: Dict[str, Any],
        episode_id: str,
        num_agents: int,
    ) -> GeneratedTask:
        """Parse LLM response into GeneratedTask."""
        task_id = f"task_{uuid.uuid4().hex[:8]}"

        # Parse subtasks
        subtasks = []
        for st in data.get("subtasks", []):
            subtasks.append(Subtask(
                subtask_id=st.get("subtask_id", f"subtask_{len(subtasks)}"),
                description=st.get("description", ""),
                success_condition=st.get("success_condition", {}),
                assigned_agent=st.get("assigned_agent"),
                depends_on=st.get("depends_on", []),
                hints=st.get("hints", []),
            ))

        # Parse success condition
        sc_data = data.get("success_condition", {})
        success_condition = SuccessCondition(
            description=sc_data.get("description", "Complete the task"),
            required_states=sc_data.get("required_states", []),
            time_limit=sc_data.get("time_limit", 30),
        )

        # Parse failure conditions
        failure_conditions = []
        for fc in data.get("failure_conditions", []):
            failure_conditions.append(FailureCondition(
                description=fc.get("description", "Task failed"),
                failure_states=fc.get("failure_states", []),
                max_failed_attempts=fc.get("max_failed_attempts"),
            ))

        # Default failure condition if none provided
        if not failure_conditions:
            failure_conditions.append(FailureCondition(
                description="Too many failed attempts",
                max_failed_attempts=10,
            ))

        return GeneratedTask(
            task_id=task_id,
            title=data.get("title", "Untitled Task"),
            category=TaskCategory(data.get("category", "coordination")),
            description=data.get("description", ""),
            initial_world_state=data.get("initial_world_state", {}),
            required_mechanics=data.get("required_mechanics", []),
            num_agents=num_agents,
            agent_roles=data.get("agent_roles", {}),
            agent_knowledge=data.get("agent_knowledge", {}),
            subtasks=subtasks,
            success_condition=success_condition,
            failure_conditions=failure_conditions,
            difficulty=data.get("difficulty", 3),
            estimated_steps=data.get("estimated_steps", 15),
            theory_of_mind_required=True,
            communication_required=True,
            source_trajectory=episode_id,
        )
