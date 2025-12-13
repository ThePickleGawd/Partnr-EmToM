#!/bin/bash
# EMTOM Benchmark Pipeline
# Usage: ./emtom/run_emtom.sh [exploration|generate|benchmark|all]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Default values
MAX_SIM_STEPS=1000
MAX_LLM_CALLS=20
EXPLORATION_STEPS=50

print_usage() {
    echo "EMTOM Benchmark Pipeline"
    echo ""
    echo "Usage: ./emtom/run_emtom.sh <command> [options]"
    echo ""
    echo "Commands:"
    echo "  exploration    Run LLM-guided exploration in Habitat (generates video)"
    echo "  generate       Generate tasks from exploration trajectories"
    echo "  benchmark      Run the Habitat benchmark with video recording"
    echo "  all            Run the full pipeline (exploration -> generate -> benchmark)"
    echo ""
    echo "Options:"
    echo "  --max-sim-steps N    Maximum simulation steps for benchmark (default: $MAX_SIM_STEPS)"
    echo "  --max-llm-calls N    Maximum LLM calls per agent (default: $MAX_LLM_CALLS)"
    echo "  --steps N            Exploration steps (default: $EXPLORATION_STEPS)"
    echo ""
    echo "Examples:"
    echo "  ./emtom/run_emtom.sh exploration --steps 100"
    echo "  ./emtom/run_emtom.sh benchmark --max-sim-steps 500"
    echo "  ./emtom/run_emtom.sh all"
}

run_exploration() {
    echo "=============================================="
    echo "Running EMTOM Exploration (Habitat Backend)"
    echo "=============================================="
    echo "Mode: LLM-guided"
    echo "Steps: $EXPLORATION_STEPS"
    echo "=============================================="
    echo ""

    # Use Hydra config system - pass parameters as config overrides
    python emtom/examples/run_habitat_exploration.py \
        --config-name examples/planner_multi_agent_demo_config \
        +exploration_steps=$EXPLORATION_STEPS \
        evaluation.save_video=true
}

run_generate() {
    echo "=============================================="
    echo "Running EMTOM Task Generation"
    echo "=============================================="
    echo "This generates tasks from exploration trajectories."
    echo ""
    python emtom/examples/generate_tasks.py \
        --trajectory-dir data/emtom/trajectories \
        --output-dir data/emtom/tasks
}

run_benchmark() {
    echo "=============================================="
    echo "Running EMTOM Habitat Benchmark"
    echo "=============================================="
    echo "Max simulation steps: $MAX_SIM_STEPS"
    echo "Max LLM calls per agent: $MAX_LLM_CALLS"
    echo "=============================================="

    python emtom/examples/run_habitat_benchmark.py \
        --config-name examples/emtom_two_robots \
        habitat.environment.max_episode_steps=$MAX_SIM_STEPS \
        evaluation.agents.agent_0.planner.plan_config.replanning_threshold=$MAX_LLM_CALLS \
        evaluation.agents.agent_1.planner.plan_config.replanning_threshold=$MAX_LLM_CALLS
}

run_all() {
    echo "=============================================="
    echo "Running Full EMTOM Pipeline"
    echo "=============================================="
    run_exploration
    run_generate
    run_benchmark
}

# Parse command line arguments
COMMAND=""
while [[ $# -gt 0 ]]; do
    case $1 in
        exploration|generate|benchmark|all)
            COMMAND=$1
            shift
            ;;
        --max-sim-steps)
            MAX_SIM_STEPS=$2
            shift 2
            ;;
        --max-llm-calls)
            MAX_LLM_CALLS=$2
            shift 2
            ;;
        --steps)
            EXPLORATION_STEPS=$2
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

if [ -z "$COMMAND" ]; then
    print_usage
    exit 1
fi

case $COMMAND in
    exploration)
        run_exploration
        ;;
    generate)
        run_generate
        ;;
    benchmark)
        run_benchmark
        ;;
    all)
        run_all
        ;;
esac

echo ""
echo "=============================================="
echo "Done!"
echo "=============================================="
