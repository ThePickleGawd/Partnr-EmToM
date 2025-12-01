#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from typing import Any, Dict, List, Tuple

import cv2
import imageio
import numpy as np

from habitat_llm.agent.env import EnvironmentInterface


class DebugVideoUtil:
    """
    This class provides an interface wrapper for creating, saving, and viewing third person videos of individual skill runs using the EnvironmentInterface API.

    For example, see `execute_skill` function below.
    NOTE: This code was largely adapted from the evaluation_runner.py
    """

    def __init__(
        self, env_interface_arg: EnvironmentInterface, output_dir: str
    ) -> None:
        """
        Construct the DebugVideoUtil instance from an EnvironmentInterface.

        :param env_interface_arg: The EnvironmentInterface instance.
        :param output_dir: The desired directory for saving output frames and videos.
        """

        self.env_interface = env_interface_arg

        # Declare container to store frames used for generating video
        self.frames: List[Any] = []

        self.output_dir = output_dir

        self.num_agents = 0
        for _agent_conf in self.env_interface.conf.evaluation.agents.values():
            self.num_agents += 1

    def __get_combined_frames(self, batch: Dict[str, Any]) -> np.ndarray:
        """
        For each agent, extract the observation from the "third_rgb" sensor and merge them into a single split-screen image.

        :param batch: A dict mapping observation names to values.
        :return: The composite image as a numpy array.
        """
        # Extract first agent frame
        images = []
        for obs_name, obs_value in batch.items():
            if "third_rgb" in obs_name:
                if self.num_agents == 1:
                    if "0" in obs_name or "main_agent" in obs_name:
                        images.append(obs_value)
                else:
                    images.append(obs_value)

        if not images:
            raise ValueError(
                f"No third_rgb observations found; keys: {list(batch.keys())}"
            )
        # Extract dimensions of the first image
        first = images[0]
        if hasattr(first, "cpu"):
            first = first.cpu().numpy()
        first_np = np.array(first)
        if first_np.shape[0] in (3, 4):
            first_np = np.transpose(first_np, (1, 2, 0))
        height, width = first_np.shape[:2]

        # Create an empty canvas to hold the concatenated images
        concat_image = np.zeros((height, width * len(images), 3), dtype=np.uint8)

        # Iterate through the images and concatenate them horizontally
        for i, image in enumerate(images):
            if hasattr(image, "cpu"):
                image_np = image.cpu().numpy()
            else:
                image_np = np.array(image)
            if image_np.shape[0] in (3, 4):
                image_np = np.transpose(image_np, (1, 2, 0))
            image_np = np.ascontiguousarray(image_np)
            concat_image[:, i * width : (i + 1) * width] = image_np[:, :, :3]

        return concat_image

    def _store_for_video(
        self,
        observations: Dict[str, Any],
        hl_actions: Dict[int, Any],
        popup_images: Dict[int, str] = None,
    ) -> None:
        """
        Store a video with observations and text from an observation dict and an agent to action metadata dict.
        NOTE: Could probably go into utils?

        :param observations: A dict mapping observation names to values.
        :param hl_actions: A dict mapping agent action indices to actions.
        """
        frames_concat = self.__get_combined_frames(observations)
        frames_concat = np.ascontiguousarray(frames_concat)

        for idx, action in hl_actions.items():
            # text = f"Agent_{id}:{action[0]}[{action[1]}]"
            agent_name = "Human" if str(idx) == "1" else "Robot"
            text = f"{agent_name}: {action[0]}[{action[1]}]"
            frames_concat = cv2.putText(
                frames_concat,
                text,
                (20, (int(idx) + 1) * 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )

        # Overlay popups if provided (per agent, left/right).
        if popup_images:
            # we assume two agents max for overlay placement
            for agent_idx, path in popup_images.items():
                try:
                    popup = cv2.imread(path)
                    if popup is None:
                        continue
                    # resize popup to fit in the corner
                    ph, pw = popup.shape[:2]
                    scale = 0.3
                    popup = cv2.resize(popup, (int(pw * scale), int(ph * scale)))
                    ph, pw = popup.shape[:2]
                    if int(agent_idx) == 0:
                        y0, y1 = 10, 10 + ph
                        x0, x1 = 10, 10 + pw
                    else:
                        y0, y1 = 10, 10 + ph
                        x1 = frames_concat.shape[1] - 10
                        x0 = x1 - pw
                    frames_concat[y0:y1, x0:x1] = popup
                except Exception:
                    continue

        self.frames.append(frames_concat)
        return

    def _make_video(self, play: bool = True, postfix: str = "") -> None:
        """
        Makes a video from a pre-processed set of frames using imageio and saves it to the output directory.

        :param play: Whether or not to play the video immediately.
        :param postfix: An optional postfix for the video file name.
        """
        out_file = f"{self.output_dir}/videos/video-{postfix}.mp4"
        print(f"Saving video to {out_file}")
        os.makedirs(f"{self.output_dir}/videos", exist_ok=True)
        writer = imageio.get_writer(
            out_file,
            fps=30,
            quality=4,
        )
        for frame in self.frames:
            arr = np.array(frame)
            # Squeeze any singleton dimensions to reduce to 2D/3D.
            if arr.ndim > 3 or 1 in arr.shape:
                arr = np.squeeze(arr)
            print(f"[VideoDebug] raw frame shape after squeeze: {arr.shape}, dtype={arr.dtype}")
            # Handle channel-first tensors
            if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[2] not in (3, 4):
                arr = np.transpose(arr, (1, 2, 0))
            # If 2D, expand to 3 channels
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            # If single/dual channel, pad to 3
            if arr.ndim == 3 and arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=2)
            elif arr.ndim == 3 and arr.shape[2] == 2:
                arr = np.concatenate([arr, arr[:, :, :1]], axis=2)
            # Truncate extra channels
            if arr.ndim == 3 and arr.shape[2] > 3:
                arr = arr[:, :, :3]
            if arr.ndim != 3 or arr.shape[2] not in (1, 2, 3):
                print(f"[Video] Skipping frame with invalid shape {arr.shape}")
                continue
            arr = np.ascontiguousarray(arr)
            print(f"[VideoDebug] writing frame shape {arr.shape}, dtype={arr.dtype}")
            writer.append_data(arr.astype(np.uint8))

        writer.close()
        if play:
            print("     ...playing video, press 'q' to continue...")
            self.play_video(out_file)

    def play_video(self, filename: str) -> None:
        """
        Play and loop video from a filepath with cv2.

        :param filename: The filepath of the video.
        """
        cap = cv2.VideoCapture(filename)
        last_time = time.time()
        while cap.isOpened():
            if time.time() - last_time > 1.0 / 30:
                last_time = time.time()
                ret, frame = cap.read()
                # cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
                # cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

                if ret:
                    cv2.imshow("Image", frame)
                else:
                    # looping
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        cv2.destroyAllWindows()


def execute_skill(
    high_level_skill_actions: Dict[Any, Any],
    llm_env,
    make_video: bool = True,
    vid_postfix: str = "",
    play_video: bool = True,
) -> Tuple[Dict[Any, Any], Dict[Any, Any], List[Any]]:
    """
    Execute a high-level skill from a string (e.g. as produced by the planner).
    Can create and display a video of the running skill.

    :param high_level_skill_actions: The map of agent indices to actions. TODO: typing
    :param llm_env: The planner instance. TODO: typing
    :param make_video: whether or not to create, save, and display a video of the skill.
    :param vid_postfix: An optional postfix for the video file. For example, the action name.
    :param play_video: Whether or not to immediately play the generated video.
    :return: A tuple with two dict(the first contains responses per-agent skill, the second contains the number of skill steps taken) and a list of frames.
    """
    dvu = DebugVideoUtil(
        llm_env.env_interface, llm_env.env_interface.conf.paths.results_dir
    )

    # Get the env observations
    observations = llm_env.env_interface.get_observations()
    agent_idx = list(high_level_skill_actions.keys())[0]
    skill_name = high_level_skill_actions[agent_idx][0]

    # Set up the variables
    skill_steps = 0
    max_skill_steps = 1500
    skill_done = None

    # While loop for executing skills
    while not skill_done:
        # Check if the maximum number of steps is reached
        assert (
            skill_steps < max_skill_steps
        ), f"Maximum number of steps reached: {skill_name} skill fails."

        # Get low level actions and responses
        low_level_actions, responses = llm_env.process_high_level_actions(
            high_level_skill_actions, observations
        )

        assert (
            len(low_level_actions) > 0
        ), f"No low level actions returned. Response: {responses.values()}"

        # Check if the agent finishes
        if any(responses.values()):
            skill_done = True

        # Get the observations
        obs, reward, done, info = llm_env.env_interface.step(low_level_actions)
        observations = llm_env.env_interface.parse_observations(obs)

        if make_video:
            dvu._store_for_video(observations, high_level_skill_actions)

        # Increase steps
        skill_steps += 1

    if make_video and skill_steps > 1:
        dvu._make_video(postfix=vid_postfix, play=play_video)

    return responses, {"skill_steps": skill_steps}, dvu.frames
