# Bomb Game — Design Summary

## Overview
Two Spot robots play a bomb-defusal scenario in a house. Agent_0 (Guide) privately knows the bomb room; Agent_1 (Seeker) does not. There is no bomb object; success requires navigation and the game tool to defuse.

## Setup & Observability
- Agents: agent_0, agent_1; identical Spot sensors.
- Config: `conf/game/bomb_game.yaml`; partial_obs=True; agent_asymmetry=True.
- Cameras/FPV: jaw cameras configured via `trajectory.agent_names=[agent_0,agent_1]`, `camera_prefixes=[articulated_agent_jaw,articulated_agent_jaw]`.

## Tools
- Base tools (via game agent configs): `Navigate`, `Explore`, `Wait`, `CommunicationTool`, `FindRoomTool`, `FindObjectTool`, `FindReceptacleTool`, `FindAgentActionTool`.
- Game tool (runtime): `DefuseBomb` (alias `DefuseBombTool`) — available only when in the bomb room and the bomb is not defused/failed.

## Flow
- Game state tracks `bomb_room`, attempts left, defused/failed, optional auto-defuse on enter.
- Guide communicates the bomb room; both navigate; when in the correct room, call `DefuseBomb`.

## Clear Condition
- Success when `DefuseBomb` is executed in the bomb room before attempts run out. Failure when attempts are exhausted or state marked failed.

## Theory-of-Mind Aspect
- Asymmetric knowledge: Guide knows the bomb room; Seeker does not. Seeker must interpret Guide’s communication and actions to find/defuse efficiently under partial observability.
