"""
Time-separated code game:
- Agent_0 (Writer) receives a secret 5-digit code and must write it on an object.
- Some objects rust: any code written on them is immediately destroyed.
- Some objects are stolen: Agent_1 (Seeker) will never see or interact with them.
- Agent_1 must find and submit the code with at most 3 submissions.
Game-specific tools enforce the write/read/submit flow; embodiment is handled by PARTNR tools.
"""

from __future__ import annotations

import random
import string
from typing import Dict, List, Optional, Tuple

from habitat.sims.habitat_simulator.sim_utilities import get_obj_from_id

from game.game import AgentRole, EnvironmentAdapter, GameSpec, GameState, ToolDescriptor


class TimeGameSpec(GameSpec):
    name = "time_game"

    def __init__(self, max_submissions: int = 3, seed: Optional[int] = None) -> None:
        self.max_submissions = max_submissions
        self.rng = random.Random(seed)
        self._catalog = [
            "steel_wrench",
            "paper_map",
            "wooden_crate",
            "fabric_banner",
            "glass_bottle",
            "battery_pack",
            "rope_coil",
            "kettle",
            "toy_airplane",
            "phone_stand",
            "mug",
        ]

    # --- GameSpec hooks --------------------------------------------------
    def initialize(
        self, agent_ids: List[str], env: EnvironmentAdapter
    ) -> GameState:
        if len(agent_ids) < 2:
            raise ValueError("TimeGameSpec requires two agents (writer and seeker).")

        writer_id = agent_ids[0]
        seeker_id = agent_ids[1]
        roles: Dict[str, AgentRole] = {
            writer_id: AgentRole(name="Writer", private_info={}),
            seeker_id: AgentRole(name="Seeker", private_info={}),
        }

        rooms = env.list_rooms() if hasattr(env, "list_rooms") else []
        objects = self._choose_objects(env)
        rusted = set(self.rng.sample(objects, k=max(1, len(objects) // 4)))
        remaining = [o for o in objects if o not in rusted]
        stolen = set(self.rng.sample(remaining, k=max(1, len(remaining) // 5))) if remaining else set()

        secret_code = self._generate_code()
        roles[writer_id].private_info["secret_code"] = secret_code
        roles[writer_id].private_info["objects"] = objects

        public_info = {
            "game": "time_game",
            "rooms": rooms,
            "visible_objects": [o for o in objects if o not in stolen],
            "rules": (
                "Writer knows a 5-digit code and must inscribe it on an object. "
                "Some objects rust (code destroyed immediately). Some are stolen (Seeker never sees them). "
                "Seeker must read and submit the code within 3 submissions."
            ),
        }

        secret_state = {
            "objects": objects,
            "rusted": rusted,
            "stolen": stolen,
            "secret_code": secret_code,
            "code_location": None,
            "code_destroyed": False,
            "code_written": False,
            "code_read": False,
            "submissions_left": self.max_submissions,
            "outcome": None,
        }

        return GameState(
            agent_roles=roles,
            public_info=public_info,
            secret_state=secret_state,
        )

    def get_tools_for_agent(
        self, agent_id: str, state: GameState, env: EnvironmentAdapter
    ) -> List[ToolDescriptor]:
        role = state.agent_roles.get(agent_id)
        if role is None:
            return []

        tools: List[ToolDescriptor] = []
        if role.name == "Writer":
            tools.append(
                ToolDescriptor(
                    name="write_secret_code",
                    description=(
                        "Write your secret 5-digit code on an object you are holding, then place it so the Seeker can read it. "
                        "Include the code explicitly. Example: write_secret_code[kettle_3, 42742]"
                    ),
                    parameters={"object_name": "string", "code": "string"},
                    handler=self._write_secret_code,
                )
            )
        if role.name == "Seeker":
            tools.append(
                ToolDescriptor(
                    name="read_secret_code",
                    description="Read the code written on the specified object (must have picked it up first). Example: read_secret_code[kettle_3]",
                    parameters={"object_name": "string"},
                    handler=self._read_secret_code,
                )
            )
            tools.append(
                ToolDescriptor(
                    name="submit_secret_code",
                    description="Submit the 5-digit code. You have limited attempts. Example: submit_secret_code[12345]",
                    parameters={"code": "string"},
                    handler=self._submit_secret_code,
                )
            )
        return tools

    def check_end_condition(
        self, state: GameState, env: EnvironmentAdapter
    ) -> Tuple[bool, Optional[str]]:
        outcome = state.secret_state.get("outcome")
        if outcome is not None:
            return True, outcome
        if state.secret_state.get("submissions_left", 0) <= 0:
            return True, "failure"
        return False, None

    def render_public_context(self, state: GameState) -> str:
        return (
            "Game: Time-separated secret code. Writer inscribes a 5-digit code on an object; "
            "some objects rust and destroy the code; some are stolen and invisible to Seeker. "
            "Seeker must read and submit the code within 3 attempts."
        )

    def render_private_context(self, agent_id: str, state: GameState) -> str:
        role = state.agent_roles.get(agent_id)
        if not role:
            return ""
        if role.name == "Writer":
            code = role.private_info.get("secret_code", "")
            objs = role.private_info.get("objects", [])
            return (
                f"You are the Writer. Secret code: {code}. "
                f"Objects available: {objs}. Pick up one object, use write_secret_code to inscribe the code, "
                "then place it back down (and tell the Seeker where)."
            )
        if role.name == "Seeker":
            visible = state.public_info.get("visible_objects", [])
            return (
                f"You are the Seeker. You cannot see stolen objects. Visible objects: {visible}. "
                "Find and read the code, then submit using submit_secret_code. You have 3 attempts."
            )
        return ""

    def debug_summary(self, state: GameState, env: EnvironmentAdapter) -> list:
        return [
            f"Secret code: {state.secret_state.get('secret_code')}",
            f"Rusted objects: {sorted(state.secret_state.get('rusted', []))}",
            f"Stolen objects: {sorted(state.secret_state.get('stolen', []))}",
        ]

    # --- Tool handlers ----------------------------------------------------
    def _write_secret_code(
        self, agent_id: str, orchestrator, object_name: str = "", code: str = "", **_: Dict
    ) -> Dict[str, any]:
        state = orchestrator.state
        if state.secret_state.get("code_written"):
            return {"ok": False, "error": "Code already written."}
        if not object_name:
            return {"ok": False, "error": "Specify an object to write on."}
        objs = state.secret_state.get("objects", [])
        if object_name not in objs:
            return {"ok": False, "error": f"Unknown object '{object_name}'. Valid: {objs}"}
        # Require that the writer is actually holding the object.
        held = set()
        # Prefer held objects in game state
        held_map = {}
        game_state = getattr(orchestrator, "state", None)
        if game_state and hasattr(game_state, "secret_state"):
            hm = game_state.secret_state.get("held_objects", {})
            if isinstance(hm, dict):
                held_map.update(hm)
        # Also look at the underlying env_interface game_state if present
        try:
            adapter = getattr(orchestrator, "env", None)
            env_iface = getattr(adapter, "env", None)
            gm = getattr(env_iface, "game_state", {})
            if isinstance(gm, dict) and "held_objects" in gm and isinstance(
                gm["held_objects"], dict
            ):
                held_map.update(gm["held_objects"])
            # Query the simulator directly for ground truth grasped handles (covers manual planner stepping internally)
            env_if = getattr(getattr(orchestrator, "env", None), "env", None)
            sim = getattr(env_if, "sim", None)
            wg = getattr(env_if, "full_world_graph", None)
            if sim is not None and hasattr(sim, "agents_mgr"):
                for agent_name in getattr(sim.agents_mgr, "agent_names", []):
                    try:
                        agent_uid = int(agent_name.split("_")[1])
                    except Exception:
                        agent_uid = 0
                    try:
                        gmgr = sim.agents_mgr[agent_uid].grasp_mgr
                        grasped = getattr(gmgr, "is_grasped", False)
                        snap_idx = getattr(gmgr, "snap_idx", None)
                        handle = None
                        resolved_name = None
                        if grasped and snap_idx is not None:
                            obj = get_obj_from_id(sim, int(snap_idx))
                            handle = getattr(obj, "handle", None)
                            try:
                                if wg is not None and wg.has_node_with_sim_handle(handle):
                                    resolved_name = wg.get_node_from_sim_handle(handle).name
                            except Exception:
                                resolved_name = None
                        print(
                            f"[TimeGame] grasp probe agent={agent_uid} is_grasped={grasped} snap_idx={snap_idx} handle={handle} name={resolved_name}"
                        )
                        chosen = resolved_name or handle
                        if chosen:
                            held_map[str(agent_uid)] = chosen
                    except Exception as exc:
                        print(f"[TimeGame] grasp probe error agent={agent_name}: {exc}")
        except Exception as exc:
            print(f"[TimeGame] grasp probe outer error: {exc}")
        except Exception:
            pass
        print(f"[TimeGame] held_map={held_map}")
        if game_state and hasattr(game_state, "secret_state"):
            game_state.secret_state["held_objects"] = held_map
        held_val = held_map.get(agent_id) or held_map.get(str(agent_id))
        if held_val:
            if isinstance(held_val, list):
                held = set(held_val)
            else:
                held = {held_val}
        # Normalize held entries to names if they are sim handles
        try:
            env_if = getattr(getattr(orchestrator, "env", None), "env", None)
            wg = getattr(env_if, "full_world_graph", None)
            if wg is not None:
                normalized = set()
                for h in held:
                    if wg.has_node_with_sim_handle(h):
                        try:
                            normalized.add(wg.get_node_from_sim_handle(h).name)
                            continue
                        except Exception:
                            pass
                    normalized.add(h)
                held = normalized
        except Exception:
            pass
        if object_name not in held:
            return {
                "ok": False,
                "error": f"You must pick up {object_name} before writing the code on it. Held_map={held_map}",
            }
        if not code:
            code = state.secret_state.get("secret_code", "")
        state.secret_state["code_written"] = True
        state.secret_state["code_location"] = object_name
        state.secret_state["inscribed_code"] = code
        if object_name in state.secret_state.get("rusted", set()):
            state.secret_state["code_destroyed"] = True
            return {
                "ok": True,
                "message": (
                    f"Wrote the code on {object_name}, but it rusted immediately. "
                    "The code is destroyed."
                ),
            }
        if object_name in state.secret_state.get("stolen", set()):
            return {
                "ok": True,
                "message": (
                    f"Wrote the code on {object_name}, but it will be stolen in the future. "
                    "Seeker will not be able to find it."
                ),
            }
        return {"ok": True, "message": f"Wrote the code '{code}' on {object_name}."}

    def _read_secret_code(
        self, agent_id: str, orchestrator, object_name: str = "", **_: Dict
    ) -> Dict[str, any]:
        state = orchestrator.state
        if not object_name:
            return {"ok": False, "error": "Specify an object to read."}
        if object_name in state.secret_state.get("stolen", set()):
            return {"ok": False, "error": f"Object '{object_name}' is missing; cannot interact with it."}
        if not state.secret_state.get("code_written"):
            return {"ok": False, "error": "No code has been written yet."}
        if state.secret_state.get("code_location") != object_name:
            return {"ok": False, "error": f"No code found on {object_name}."}
        if state.secret_state.get("code_destroyed"):
            return {"ok": False, "error": f"The code on {object_name} rusted away and is unreadable."}
        state.secret_state["code_read"] = True
        return {"ok": True, "code": state.secret_state.get("inscribed_code", state.secret_state.get("secret_code"))}

    def _submit_secret_code(
        self, agent_id: str, orchestrator, code: str = "", **_: Dict
    ) -> Dict[str, any]:
        state = orchestrator.state
        remaining = state.secret_state.get("submissions_left", 0)
        if remaining <= 0:
            state.secret_state["outcome"] = "failure"
            return {"ok": False, "error": "No submissions left."}
        if not code:
            return {"ok": False, "error": "Provide a 5-digit code to submit."}
        state.secret_state["submissions_left"] = remaining - 1
        if code.strip() == state.secret_state.get("secret_code"):
            state.secret_state["outcome"] = "success"
            return {"ok": True, "message": "Correct code submitted. You win!"}
        if state.secret_state["submissions_left"] == 0:
            state.secret_state["outcome"] = "failure"
        return {
            "ok": True,
            "message": f"Incorrect code. Submissions left: {state.secret_state['submissions_left']}",
        }

    # --- Helpers ---------------------------------------------------------
    def _generate_code(self) -> str:
        return "".join(self.rng.choice(string.digits) for _ in range(5))

    def _choose_objects(self, env: EnvironmentAdapter, k: int = 6) -> List[str]:
        objs: List[str] = []
        try:
            if hasattr(env, "list_objects"):
                objs = env.list_objects()
        except Exception:
            objs = []
        pool = objs if len(objs) >= 3 else self._catalog
        if len(pool) < 3:
            pool = self._catalog
        return self.rng.sample(pool, k=min(k, len(pool)))
