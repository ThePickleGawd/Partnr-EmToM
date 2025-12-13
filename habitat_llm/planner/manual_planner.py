from __future__ import annotations

import time
from typing import Any, Dict, Tuple, Optional, List, DefaultDict
from collections import defaultdict

import os

from habitat_llm.planner.planner import Planner
from habitat_llm.llm import instantiate_llm
from habitat_llm.tools import PerceptionTool
from habitat_llm.examples.example_utils import DebugVideoUtil


class ManualPlanner(Planner):
    """
    Minimal planner that prompts the user on stdin for each step.
    Input format: Tool[arg1, arg2]. Use empty input for Wait.
    Type 'done' to end the episode for this agent.
    """

    def __init__(self, plan_config, env_interface) -> None:
        super().__init__(plan_config, env_interface)
        self._obs_counter: Dict[int, int] = {}
        self._printed_world = False
        self._last_action_sig: Optional[str] = None
        self._obs_counter: Dict[int, int] = {}
        self._manual_fpv_keys: Optional[Dict[str, str]] = None
        self.llm = self._hardcode_openai_llm()

    def _hardcode_openai_llm(self):
        """
        Always use the OpenAI API backend for manual tools.
        """
        try:
            llm = instantiate_llm("openai_chat")
            print(f"[Manual CLI] Using OpenAI LLM for manual planner: {type(llm).__name__}")
            return llm
        except Exception as exc:
            print(f"[Manual CLI] Failed to initialize OpenAI LLM: {exc}")
            return None

    @Planner.agents.setter
    def agents(self, agents):
        self._agents = agents
        # If an LLM is available, pass it to all agent tools that accept it.
        if self.llm is not None:
            for agent in self._agents:
                try:
                    agent.pass_llm_to_tools(self.llm)
                except Exception as exc:
                    print(f"[Manual CLI] Failed to attach LLM to agent_{agent.uid} tools: {exc}")

    def _prompt_action(self, agent_uid: int) -> Tuple[str, str, str]:
        agent_obj = next(a for a in self.agents if a.uid == agent_uid)
        available = sorted(list(agent_obj.tools.keys()))
        header = f"\nAgent_{agent_uid} > Tools: {available}"
        prompt = f"{header}\nAgent_{agent_uid} > "
        while True:
            try:
                raw = input(prompt).strip()
            except EOFError:
                # If stdin is closed (non-interactive run), keep waiting instead of crashing.
                time.sleep(1.0)
                continue
            except KeyboardInterrupt:
                self.is_done = True
                return "Done", "", None
            break
        if raw.lower() == "done":
            self.is_done = True
            return "Done", "", None
        if raw == "":
            return "Wait", "", None
        # Expect format Tool[args]
        if "[" in raw and raw.endswith("]"):
            name, rest = raw.split("[", 1)
            args = rest[:-1]  # drop trailing ]
        else:
            name = raw
            args = ""
        name = name.strip()
        # Validate tool name
        if name not in available and name not in {"Wait", "Done"}:
            print(f"[Warning] Invalid tool '{name}'. Valid: {available}. Try again.")
            return self._prompt_action(agent_uid)
        return name, args, None

    def get_next_action(
        self,
        instruction: str,
        observations: Dict[str, Any],
        world_graph: Dict[int, Any],
        verbose: bool = False,
    ):
        if not hasattr(self, "_printed_world"):
            # Print the full world graph once for context.
            try:
                world_desc = world_graph[self.agents[0].uid].get_world_descr(
                    is_human_wg=False
                )
                print(world_desc)
            except Exception:
                pass
            self._printed_world = True

        if self.is_done:
            info = {
                "high_level_actions": {},
                "responses": {},
                "replanned": {agent.uid: False for agent in self.agents},
                "replan_required": {agent.uid: False for agent in self.agents},
                "is_done": {agent.uid: True for agent in self.agents},
            }
            return {}, info, True

        hl_actions: Dict[int, Tuple[str, str, str]] = {}
        responses: Dict[int, str] = {}
        manual_frames: Dict[int, List[Any]] = {}
        manual_fpv_frames: Dict[str, List[Any]] = {}
        for agent in self.agents:
            obs_agent = self.filter_obs_space(observations, agent.uid)
            name, args, err = self._prompt_action(agent.uid)
            if name == "Done":
                self.is_done = True
                break
            hl_actions[agent.uid] = (name, args, err)
            # Execute immediately: perception tools single-step, motor skills run to completion.
            tool = agent.get_tool_from_name(name)
            if isinstance(tool, PerceptionTool):
                _, resp = self.process_high_level_actions(
                    {agent.uid: hl_actions[agent.uid]}, observations
                )
                responses.update(resp)
            else:
                resp, frames, fpv_frames = self._run_skill_with_empty_allowed(
                    {agent.uid: hl_actions[agent.uid]}
                )
                responses.update(resp)
                if frames:
                    # keep frames for both per-action save and downstream cumulative video
                    manual_frames.setdefault(agent.uid, []).extend(frames)
                if fpv_frames:
                    for agent_name, f_list in fpv_frames.items():
                        manual_fpv_frames.setdefault(agent_name, []).extend(f_list)
                final_obs = self.env_interface.get_observations()
                final_obs_agent = self.filter_obs_space(final_obs, agent.uid)
                self._popup_media(
                    [self._extract_rgb(final_obs_agent)],
                    agent_uid=agent.uid,
                    popup=self.planner_config.get("manual_obs_popup", False),
                )
            # Refresh observations after executing to reflect new state
            observations = self.env_interface.get_observations()

        if self.is_done:
            # short-circuit without invoking tools (avoids "Tool 'Done' not found")
            hl_actions = {
                agent.uid: ("Done", "", None) for agent in self.agents
            }
            responses = {
                agent.uid: "Episode ended by user (Done)." for agent in self.agents
            }
            low_level_actions = {}
        else:
            # Actions already executed; no low-level actions to return.
            low_level_actions = {}

        # Fill empty responses to avoid downstream null issues
        for agent in self.agents:
            if agent.uid not in responses or not responses[agent.uid]:
                responses[agent.uid] = "Manual action acknowledged."
            print(f"Agent_{agent.uid} Response: {responses[agent.uid]}")
            # Print current room if available
            try:
                room = self.env_interface.get_agent_room(agent.uid)
                print(f"Agent_{agent.uid} is in room: {room}")
            except Exception:
                pass

        info = {
            "high_level_actions": hl_actions,
            "responses": responses,
            "replanned": {agent.uid: True for agent in self.agents},
            "replan_required": {agent.uid: False for agent in self.agents},
            "is_done": {agent.uid: self.is_done for agent in self.agents},
        }
        if manual_frames:
            # Pass collected frames to the evaluation runner so it can build a cumulative video.
            info["manual_video_frames"] = manual_frames
        if manual_fpv_frames:
            info["manual_fpv_frames"] = manual_fpv_frames
        return low_level_actions, info, self.is_done

    def reset(self) -> None:
        """Reset manual planner state between episodes."""
        self.is_done = False
        self._printed_world = False
        self._last_action_sig = None
        self._obs_counter = {}
    def _popup_media(self, frames, agent_uid: int, popup: bool = False) -> None:
        """
        Show a final observation frame if requested.
        """
        if not popup or not frames:
            return
        try:
            import matplotlib
            matplotlib.rcParams["toolbar"] = "none"
            if matplotlib.get_backend().lower().startswith("qt"):
                try:
                    matplotlib.use("TkAgg", force=True)
                except Exception:
                    pass
            import matplotlib.pyplot as plt

            fig = plt.figure(f"Agent_{agent_uid}_action")
            ax = fig.gca()
            ax.axis("off")
            ax.imshow(frames[-1])
            fig.canvas.draw()
            # Block until user closes the window to avoid instant disappearance.
            plt.show(block=True)
            plt.close(fig)
        except Exception as exc:
            print(f"[Manual CLI] Popup failed: {exc}")

    def _run_skill_with_empty_allowed(self, hl_actions: Dict[int, Tuple[str, str, str]], make_video: bool = True):
        """
        Run a motor skill to completion, tolerating empty low-level actions if a response is returned.
        This mirrors execute_skill but allows immediate-completion tools (e.g., defuse_bomb).
        """
        # Use a fresh DebugVideoUtil per action to avoid overwriting the same video file.
        dvu = DebugVideoUtil(
            self.env_interface,
            self.env_interface.conf.paths.results_dir,
            unique_postfix=True,
        )
        observations = self.env_interface.get_observations()
        fpv_frames: DefaultDict[str, List[Any]] = defaultdict(list)
        self._capture_fpv_frames(observations, fpv_frames)
        agent_idx = list(hl_actions.keys())[0]
        action_name = hl_actions[agent_idx][0].lower()
        skill_steps = 0
        max_skill_steps = 1500
        # Capture the initial frame so short/empty runs still yield a video
        if make_video:
            try:
                dvu._store_for_video(observations, hl_actions)
            except Exception:
                pass
        # Fast path for Wait: do a single iteration and return
        if action_name == "wait":
            responses = {agent_idx: "Wait completed."}
            if make_video:
                dvu._store_for_video(observations, hl_actions)
                dvu._make_video(play=False)
            return responses, dvu.frames, fpv_frames

        while True:
            low_level_actions, responses = self.process_high_level_actions(
                hl_actions, observations
            )
            # If no actions but we have a response, consider it done
            if len(low_level_actions) == 0:
                if any(responses.values()):
                    break
                else:
                    # nothing to do; avoid infinite loop
                    break
            obs, reward, done, info = self.env_interface.step(low_level_actions)
            observations = self.env_interface.parse_observations(obs)
            self._capture_fpv_frames(observations, fpv_frames)
            if make_video:
                dvu._store_for_video(observations, hl_actions)
            skill_steps += 1
            if any(responses.values()):
                break
            if skill_steps >= max_skill_steps:
                responses = {agent_idx: "Skill timed out."}
                break
        if make_video and dvu.frames:
            dvu._make_video(play=False)
        return responses, dvu.frames, fpv_frames

    def _extract_rgb(self, obs: Dict[str, Any]) -> Optional[Any]:
        """
        Extract ONLY the first-person head_rgb. If missing, return None (no fallbacks).
        """
        key = "head_rgb"
        if key not in obs:
            print(f"[Manual CLI] head_rgb not found. Keys: {list(obs.keys())}")
            return None
        arr_raw = obs[key]
        arr = self._to_np(arr_raw)
        if arr is None:
            print(
                f"[Manual CLI] head_rgb present but could not convert to array. "
                f"type={type(arr_raw)}"
            )
            return None
        arr = self._squeeze_batch_dim(arr)
        arr = self._ensure_hwc(arr)
        if arr is None:
            shape = getattr(arr_raw, "shape", None)
            dtype = getattr(arr_raw, "dtype", None)
            print(
                f"[Manual CLI] head_rgb present but not valid image. "
                f"raw_shape={shape}, raw_dtype={dtype}"
            )
        return arr

    def _to_np(self, arr: Any):
        try:
            import numpy as np
            import torch
        except Exception:
            import numpy as np  # type: ignore
            torch = None  # type: ignore
        try:
            if torch is not None and isinstance(arr, torch.Tensor):
                arr = arr.detach().cpu().numpy()
            if hasattr(arr, "shape"):
                return arr
        except Exception:
            return None
        return None

    def _squeeze_batch_dim(self, arr: Any):
        """
        Drop a leading batch dimension of size 1 if present.
        """
        try:
            import numpy as np

            a = np.array(arr)
            if a.ndim == 4 and a.shape[0] == 1:
                return a[0]
            return a
        except Exception:
            return arr

    def _ensure_hwc(self, arr: Any):
        """
        Convert array to HWC 3-channel uint8 if possible.
        """
        import numpy as np

        if arr is None or not hasattr(arr, "shape"):
            return None
        a = np.array(arr)
        if a.ndim == 4 and a.shape[0] == 1:
            a = a[0]
        if a.ndim != 3:
            return None
        # If channel-first (C,H,W)
        if a.shape[0] in (3, 4):
            a = np.transpose(a, (1, 2, 0))
        # Require at least 3 channels
        if a.shape[2] < 3:
            return None
        a = a[:, :, :3]
        # normalize/convert to uint8
        if a.dtype != np.uint8:
            a = np.clip(a, 0, 255)
            if a.max() <= 1.0:
                a = a * 255.0
            a = a.astype(np.uint8)
        return a

    # --- First-person capture helpers ---------------------------------
    def _get_fpv_keys(self) -> Dict[str, str]:
        """
        Build observation keys for FPV capture based on trajectory config.
        """
        if self._manual_fpv_keys is not None:
            return self._manual_fpv_keys
        traj_conf = getattr(self.env_interface.conf, "trajectory", None)
        agent_names = list(getattr(traj_conf, "agent_names", [])) if traj_conf else []
        cam_prefixes = list(getattr(traj_conf, "camera_prefixes", [])) if traj_conf else []
        if not agent_names or len(agent_names) != len(cam_prefixes):
            self._manual_fpv_keys = {}
            return self._manual_fpv_keys
        keys: Dict[str, str] = {}
        for agent_name, cam_prefix in zip(agent_names, cam_prefixes):
            if getattr(self.env_interface, "_single_agent_mode", False):
                keys[agent_name] = f"{cam_prefix}_rgb"
            else:
                keys[agent_name] = f"{agent_name}_{cam_prefix}_rgb"
        self._manual_fpv_keys = keys
        return self._manual_fpv_keys

    def _capture_fpv_frames(
        self, observations: Dict[str, Any], fpv_frames: DefaultDict[str, List[Any]]
    ) -> None:
        """
        Collect first-person frames for the configured cameras into fpv_frames.
        """
        keys = self._get_fpv_keys()
        if not keys:
            return
        for agent_name, obs_key in keys.items():
            if obs_key not in observations:
                continue
            frame = self._ensure_hwc(self._to_np(observations[obs_key]))
            if frame is not None:
                fpv_frames[agent_name].append(frame)
