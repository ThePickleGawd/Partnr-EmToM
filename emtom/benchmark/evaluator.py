"""
Benchmark evaluator for EMTOM.

Aggregates results from multiple task runs and computes benchmark metrics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from emtom.benchmark.task_runner import TaskResult, TaskStatus


@dataclass
class TaskMetrics:
    """Metrics for a single task across multiple runs."""
    task_id: str
    task_title: str
    difficulty: int
    category: str

    # Aggregate metrics
    total_runs: int = 0
    successes: int = 0
    failures: int = 0
    timeouts: int = 0

    # Performance metrics
    avg_steps: float = 0.0
    min_steps: int = 0
    max_steps: int = 0
    avg_time_seconds: float = 0.0

    # Subtask metrics
    avg_subtasks_completed: float = 0.0

    # Communication metrics
    avg_messages_per_run: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return self.successes / self.total_runs if self.total_runs > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_title": self.task_title,
            "difficulty": self.difficulty,
            "category": self.category,
            "total_runs": self.total_runs,
            "successes": self.successes,
            "failures": self.failures,
            "timeouts": self.timeouts,
            "success_rate": self.success_rate,
            "avg_steps": self.avg_steps,
            "min_steps": self.min_steps,
            "max_steps": self.max_steps,
            "avg_time_seconds": self.avg_time_seconds,
            "avg_subtasks_completed": self.avg_subtasks_completed,
            "avg_messages_per_run": self.avg_messages_per_run,
        }


@dataclass
class BenchmarkResults:
    """Aggregated results from a full benchmark run."""
    benchmark_id: str
    timestamp: str
    agent_model: str

    # Overall metrics
    total_tasks: int = 0
    total_runs: int = 0
    overall_success_rate: float = 0.0

    # Per-category success rates
    category_success_rates: Dict[str, float] = field(default_factory=dict)

    # Per-difficulty success rates
    difficulty_success_rates: Dict[int, float] = field(default_factory=dict)

    # Theory of mind metrics
    tom_required_success_rate: float = 0.0
    tom_not_required_success_rate: float = 0.0

    # Communication metrics
    comm_required_success_rate: float = 0.0
    avg_messages_per_task: float = 0.0

    # Per-task metrics
    task_metrics: List[TaskMetrics] = field(default_factory=list)

    # Raw results
    all_results: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "timestamp": self.timestamp,
            "agent_model": self.agent_model,
            "total_tasks": self.total_tasks,
            "total_runs": self.total_runs,
            "overall_success_rate": self.overall_success_rate,
            "category_success_rates": self.category_success_rates,
            "difficulty_success_rates": self.difficulty_success_rates,
            "tom_required_success_rate": self.tom_required_success_rate,
            "tom_not_required_success_rate": self.tom_not_required_success_rate,
            "comm_required_success_rate": self.comm_required_success_rate,
            "avg_messages_per_task": self.avg_messages_per_task,
            "task_metrics": [t.to_dict() for t in self.task_metrics],
        }

    def save(self, path: str) -> None:
        """Save results to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "BenchmarkResults":
        """Load results from JSON file."""
        with open(path) as f:
            data = json.load(f)

        results = cls(
            benchmark_id=data["benchmark_id"],
            timestamp=data["timestamp"],
            agent_model=data["agent_model"],
            total_tasks=data["total_tasks"],
            total_runs=data["total_runs"],
            overall_success_rate=data["overall_success_rate"],
            category_success_rates=data["category_success_rates"],
            difficulty_success_rates={int(k): v for k, v in data["difficulty_success_rates"].items()},
            tom_required_success_rate=data.get("tom_required_success_rate", 0.0),
            tom_not_required_success_rate=data.get("tom_not_required_success_rate", 0.0),
            comm_required_success_rate=data.get("comm_required_success_rate", 0.0),
            avg_messages_per_task=data.get("avg_messages_per_task", 0.0),
        )

        # Reconstruct task metrics
        for tm_data in data.get("task_metrics", []):
            results.task_metrics.append(TaskMetrics(
                task_id=tm_data["task_id"],
                task_title=tm_data["task_title"],
                difficulty=tm_data["difficulty"],
                category=tm_data["category"],
                total_runs=tm_data["total_runs"],
                successes=tm_data["successes"],
                failures=tm_data["failures"],
                timeouts=tm_data["timeouts"],
                avg_steps=tm_data["avg_steps"],
                min_steps=tm_data["min_steps"],
                max_steps=tm_data["max_steps"],
                avg_time_seconds=tm_data["avg_time_seconds"],
                avg_subtasks_completed=tm_data.get("avg_subtasks_completed", 0.0),
                avg_messages_per_run=tm_data.get("avg_messages_per_run", 0.0),
            ))

        return results


class BenchmarkEvaluator:
    """
    Evaluates EMTOM benchmark results.

    Aggregates results from multiple task runs and computes metrics
    for analysis and comparison.
    """

    def __init__(self, agent_model: str = "unknown"):
        self.agent_model = agent_model
        self._results: List[TaskResult] = []
        self._task_info: Dict[str, Dict[str, Any]] = {}  # task_id -> task metadata

    def add_task_info(
        self,
        task_id: str,
        title: str,
        difficulty: int,
        category: str,
        theory_of_mind_required: bool = False,
        communication_required: bool = False,
        num_subtasks: int = 0,
    ) -> None:
        """Register task metadata for evaluation."""
        self._task_info[task_id] = {
            "title": title,
            "difficulty": difficulty,
            "category": category,
            "theory_of_mind_required": theory_of_mind_required,
            "communication_required": communication_required,
            "num_subtasks": num_subtasks,
        }

    def add_result(self, result: TaskResult) -> None:
        """Add a task result for evaluation."""
        self._results.append(result)

    def evaluate(self, benchmark_id: Optional[str] = None) -> BenchmarkResults:
        """
        Evaluate all collected results and compute metrics.

        Returns:
            BenchmarkResults with aggregated metrics
        """
        if benchmark_id is None:
            benchmark_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        results = BenchmarkResults(
            benchmark_id=benchmark_id,
            timestamp=datetime.now().isoformat(),
            agent_model=self.agent_model,
        )

        # Group results by task
        task_results: Dict[str, List[TaskResult]] = {}
        for r in self._results:
            if r.task_id not in task_results:
                task_results[r.task_id] = []
            task_results[r.task_id].append(r)

        results.total_tasks = len(task_results)
        results.total_runs = len(self._results)

        # Calculate per-task metrics
        total_successes = 0
        category_stats: Dict[str, Dict[str, int]] = {}  # category -> {successes, total}
        difficulty_stats: Dict[int, Dict[str, int]] = {}  # difficulty -> {successes, total}
        tom_stats = {"required": {"successes": 0, "total": 0}, "not_required": {"successes": 0, "total": 0}}
        comm_stats = {"required": {"successes": 0, "total": 0}}
        total_messages = 0

        for task_id, runs in task_results.items():
            info = self._task_info.get(task_id, {})
            category = info.get("category", "unknown")
            difficulty = info.get("difficulty", 0)
            tom_required = info.get("theory_of_mind_required", False)
            comm_required = info.get("communication_required", False)

            # Compute task metrics
            tm = TaskMetrics(
                task_id=task_id,
                task_title=info.get("title", task_id),
                difficulty=difficulty,
                category=category,
            )

            steps_list = []
            times_list = []
            subtasks_list = []
            messages_list = []

            for run in runs:
                tm.total_runs += 1

                if run.status == TaskStatus.SUCCESS:
                    tm.successes += 1
                    total_successes += 1
                elif run.status == TaskStatus.FAILURE:
                    tm.failures += 1
                else:  # TIMEOUT
                    tm.timeouts += 1

                steps_list.append(run.total_steps)
                times_list.append(run.time_elapsed_seconds)
                subtasks_list.append(len(run.subtasks_completed))

                # Count messages
                run_messages = sum(
                    m.get("messages_sent", 0)
                    for m in run.agent_metrics.values()
                )
                messages_list.append(run_messages)
                total_messages += run_messages

            if steps_list:
                tm.avg_steps = sum(steps_list) / len(steps_list)
                tm.min_steps = min(steps_list)
                tm.max_steps = max(steps_list)

            if times_list:
                tm.avg_time_seconds = sum(times_list) / len(times_list)

            if subtasks_list:
                tm.avg_subtasks_completed = sum(subtasks_list) / len(subtasks_list)

            if messages_list:
                tm.avg_messages_per_run = sum(messages_list) / len(messages_list)

            results.task_metrics.append(tm)

            # Update category stats
            if category not in category_stats:
                category_stats[category] = {"successes": 0, "total": 0}
            category_stats[category]["successes"] += tm.successes
            category_stats[category]["total"] += tm.total_runs

            # Update difficulty stats
            if difficulty not in difficulty_stats:
                difficulty_stats[difficulty] = {"successes": 0, "total": 0}
            difficulty_stats[difficulty]["successes"] += tm.successes
            difficulty_stats[difficulty]["total"] += tm.total_runs

            # Update ToM stats
            if tom_required:
                tom_stats["required"]["successes"] += tm.successes
                tom_stats["required"]["total"] += tm.total_runs
            else:
                tom_stats["not_required"]["successes"] += tm.successes
                tom_stats["not_required"]["total"] += tm.total_runs

            # Update communication stats
            if comm_required:
                comm_stats["required"]["successes"] += tm.successes
                comm_stats["required"]["total"] += tm.total_runs

        # Calculate overall success rate
        if results.total_runs > 0:
            results.overall_success_rate = total_successes / results.total_runs

        # Calculate category success rates
        for cat, stats in category_stats.items():
            if stats["total"] > 0:
                results.category_success_rates[cat] = stats["successes"] / stats["total"]

        # Calculate difficulty success rates
        for diff, stats in difficulty_stats.items():
            if stats["total"] > 0:
                results.difficulty_success_rates[diff] = stats["successes"] / stats["total"]

        # Calculate ToM success rates
        if tom_stats["required"]["total"] > 0:
            results.tom_required_success_rate = (
                tom_stats["required"]["successes"] / tom_stats["required"]["total"]
            )
        if tom_stats["not_required"]["total"] > 0:
            results.tom_not_required_success_rate = (
                tom_stats["not_required"]["successes"] / tom_stats["not_required"]["total"]
            )

        # Calculate communication success rate
        if comm_stats["required"]["total"] > 0:
            results.comm_required_success_rate = (
                comm_stats["required"]["successes"] / comm_stats["required"]["total"]
            )

        # Calculate average messages
        if results.total_runs > 0:
            results.avg_messages_per_task = total_messages / results.total_runs

        # Store raw results
        results.all_results = [r.to_dict() for r in self._results]

        return results

    def print_summary(self, results: BenchmarkResults) -> None:
        """Print a human-readable summary of benchmark results."""
        print("\n" + "=" * 60)
        print("EMTOM BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Benchmark ID: {results.benchmark_id}")
        print(f"Agent Model: {results.agent_model}")
        print(f"Timestamp: {results.timestamp}")
        print()

        print(f"OVERALL METRICS:")
        print(f"  Total Tasks: {results.total_tasks}")
        print(f"  Total Runs: {results.total_runs}")
        print(f"  Overall Success Rate: {results.overall_success_rate:.1%}")
        print()

        if results.category_success_rates:
            print("SUCCESS BY CATEGORY:")
            for cat, rate in sorted(results.category_success_rates.items()):
                print(f"  {cat}: {rate:.1%}")
            print()

        if results.difficulty_success_rates:
            print("SUCCESS BY DIFFICULTY:")
            for diff, rate in sorted(results.difficulty_success_rates.items()):
                print(f"  Difficulty {diff}: {rate:.1%}")
            print()

        print("THEORY OF MIND IMPACT:")
        print(f"  ToM Required: {results.tom_required_success_rate:.1%}")
        print(f"  ToM Not Required: {results.tom_not_required_success_rate:.1%}")
        print()

        print("COMMUNICATION METRICS:")
        print(f"  Comm Required Success Rate: {results.comm_required_success_rate:.1%}")
        print(f"  Avg Messages per Task: {results.avg_messages_per_task:.1f}")
        print()

        print("PER-TASK BREAKDOWN:")
        print("-" * 60)
        for tm in results.task_metrics:
            status_bar = f"[{'#' * tm.successes}{'.' * (tm.total_runs - tm.successes)}]"
            print(f"  {tm.task_title}")
            print(f"    {status_bar} {tm.success_rate:.0%} ({tm.successes}/{tm.total_runs})")
            print(f"    Avg steps: {tm.avg_steps:.1f} | Avg time: {tm.avg_time_seconds:.2f}s")
        print()

    def reset(self) -> None:
        """Reset all collected results."""
        self._results.clear()
        self._task_info.clear()
