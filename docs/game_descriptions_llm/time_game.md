# TimeShift Puzzle Game --- Specification

## Overview

**TimeShift** is a two-agent temporal reasoning puzzle.

Two agents, **Past (agent_0)** and **Future (agent_1)**, act
*simultaneously* in different temporal versions of the same world:

-   **Past** operates in an earlier timeline.
-   **Future** operates in a later timeline.
-   **Past's actions immediately propagate into Future's world**, but
    Past cannot observe the consequences.
-   **Future has no access to Past's switches**, only their resulting
    effects.
-   **Full communication is allowed**, but observability differs for
    each agent.

The episode's goal is for **Future** to correctly identify and retrieve
a hidden target object using a **deterministic code** produced by
**Past's switch toggles**.\
The mapping from switches → observable effects is defined by the game
manager and **unknown to both agents** initially.

Success requires **Theory of Mind**: - Past must predict how Future will
interpret ambiguous downstream effects. - Future must infer what Past
intended when toggling switches. - Both must collaboratively build a
shared model of how the temporal system works.

------------------------------------------------------------------------

## Core Game Mechanics

### Asymmetric Observability

-   Past sees the pristine earlier world and has access to **switches**
    that cause temporal effects.
-   Future sees only the downstream effects and transformed environment.
-   Past **never** sees the effects and can only infer the results
    through communication.
-   Future **cannot** manipulate past switches and must interpret the
    world state.

### Hidden Target Encoding

-   A hidden target object is selected at episode start.
-   Past's private prompt describes the **code pattern** required to
    encode this target via switch toggles.
-   Future's private prompt describes how **effect patterns** map to
    specific objects.
-   Switch toggles produce deterministic effects (lights, runes,
    symbols, door states, etc.)

### Completion Condition

Future must: 1. Interpret the effect patterns caused by Past's toggles,\
2. Infer the correct target object,\
3. Retrieve it using PARTNR-native navigation/manipulation tools,\
4. Submit it via:

    submit_target_item[target_name, justification]

------------------------------------------------------------------------

# Game-Specific Tools

## Past Agent Tools (agent_0)

### `toggle_switch[name]`

**Description:**\
Flips a named temporal switch in the Past timeline.

**Effect:**\
Triggers a deterministic effect in the Future world.\
Past cannot observe the effect and must rely on Future's reports.

**Syntax:**

    Agent_0_Action: toggle_switch[switch_A]

------------------------------------------------------------------------

## Future Agent Tools (agent_1)

### `query_effect_state[]`

**Description:**\
Returns the current effect-state in the Future world, such as:

-   LightPanel readings\
-   RuneWall indicators\
-   DoorState changes\
-   Symbol patterns

**Syntax:**

    Agent_1_Action: query_effect_state[]

### `submit_target_item[name, justification]`

**Description:**\
Future submits the final answer.

**Syntax:**

    Agent_1_Action: submit_target_item[golden_mug, "Code matched BLUE-GREEN-RED + Rune_3-1"]
# Time Game — Design Summary

## Overview
Two Spot robots play a time-separated secret-code puzzle. Agent_0 (Writer) receives a hidden 5-digit code and must inscribe it on an object. Some objects rust (code immediately destroyed); some are stolen (Agent_1 never sees them). Agent_1 (Seeker) must find, read, and submit the code within limited attempts.

## Setup & Observability
- Agents: agent_0, agent_1; identical Spot sensors.
- Config: `conf/game/time_game.yaml`; partial_obs=False; agent_asymmetry=False.
- Cameras/FPV: jaw cameras via `trajectory.agent_names=[agent_0,agent_1]`, `camera_prefixes=[articulated_agent_jaw,articulated_agent_jaw]`.

## Tools
- Base tools (via game agent configs): `Navigate`, `Explore`, `Wait`, `CommunicationTool`, `FindRoomTool`, `FindObjectTool`, `FindReceptacleTool`, `FindAgentActionTool`.
- Game tools (runtime):
  - `write_secret_code` (Writer): write code on an object; rusted objects destroy it; stolen objects become invisible to Seeker.
  - `read_secret_code` (Seeker): read code from the inscribed object; fails if rusted/stolen/wrong object.
  - `submit_secret_code` (Seeker): submit the 5-digit code; limited attempts (default 3).

## Flow
- Writer picks an object and writes the code; if rusted, code is destroyed; if stolen, Seeker can’t interact.
|- Writer may communicate object choice/outcome.
- Seeker locates the viable object, reads the code, and submits it within the allowed attempts.

## Clear Condition
- Success when Seeker submits the correct code before attempts expire. Failure if attempts run out or code cannot be recovered (rusted/stolen) and not submitted correctly.

## Theory-of-Mind Aspect
- Time separation and hidden hazards (rust, stolen) force Writer to anticipate Seeker’s view and communicate; Seeker must trust/interpret Writer’s reports and handle missing/invalid objects to recover the code.
