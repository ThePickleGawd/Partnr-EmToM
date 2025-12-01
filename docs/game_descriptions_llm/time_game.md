# Time Game — Design Summary

## Overview
Two Spot robots play a time-separated secret-code puzzle. Agent_0 (Writer) receives a hidden 5-digit code and must inscribe it on an object. Some objects rust (code immediately destroyed); some are stolen (Agent_1 never sees them). Agent_1 (Seeker) must find, read, and submit the code within limited attempts.

## Setup & Observability
- Agents: agent_0, agent_1; identical Spot sensors.
- Config: `conf/game/time_game.yaml`; partial_obs=False; agent_asymmetry=False.
- Cameras/FPV: jaw cameras via `trajectory.agent_names=[agent_0,agent_1]`, `camera_prefixes=[articulated_agent_jaw, articulated_agent_jaw]`.

## Tools
- Base tools (via game agent configs): `Navigate`, `Explore`, `Wait`, `CommunicationTool`, `FindRoomTool`, `FindObjectTool`, `FindReceptacleTool`, `FindAgentActionTool`.
- Game tools (runtime):
  - `write_secret_code` (Writer): write code on an object; rusted objects destroy it; stolen objects become invisible to Seeker. Writer must pick up the object before writing.
  - `read_secret_code` (Seeker): read code from the inscribed object; fails if rusted/stolen/wrong object.
  - `submit_secret_code` (Seeker): submit the 5-digit code; limited attempts (default 3).

## Flow
- Writer picks up an object and writes the code; if rusted, code is destroyed; if stolen, Seeker can’t interact.
- Writer may communicate object choice/outcome.
- Seeker locates the viable object, reads the code, and submits it within the allowed attempts.

## Clear Condition
- Success when Seeker submits the correct code before attempts expire. Failure if attempts run out or code cannot be recovered (rusted/stolen) and not submitted correctly.

## Theory-of-Mind Aspect
- Time separation and hidden hazards (rust, stolen) force Writer to anticipate Seeker’s view and communicate; Seeker must trust/interpret Writer’s reports and handle missing/invalid objects to recover the code.
