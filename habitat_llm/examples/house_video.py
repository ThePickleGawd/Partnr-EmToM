#!/usr/bin/env python3
from __future__ import annotations
"""
Generate a smooth, non-interactive flythrough video of a Habitat scene.

The script builds a navigation path across the navmesh, marches the agent along
it, and writes the color sensor frames to disk as a video. No keyboard or mouse
input is required.
"""

import argparse
import atexit
import json
import math
import os
import tempfile
from typing import Iterable, List, Sequence, Tuple

import imageio
import numpy as np

import habitat_sim
from habitat_sim.logging import LoggingContext, logger
from habitat_sim.utils.common import quat_from_angle_axis
from habitat_sim.utils.settings import default_sim_settings, make_cfg

_TEMP_SCENE_FILES: List[str] = []


def _cleanup_temp_scenes() -> None:
    for path in _TEMP_SCENE_FILES:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass


atexit.register(_cleanup_temp_scenes)


def heading_for_direction(
    direction: np.ndarray, fallback: np.ndarray
) -> habitat_sim.geo.Quaternion:
    """Compute a yaw-only quaternion that faces the supplied direction."""
    planar = np.array([direction[0], 0.0, direction[2]], dtype=np.float32)
    if np.linalg.norm(planar) < 1e-4:
        planar = np.array([fallback[0], 0.0, fallback[2]], dtype=np.float32)
    norm = np.linalg.norm(planar)
    if norm < 1e-4:
        planar = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    else:
        planar /= norm
    yaw = math.atan2(planar[0], planar[2])
    return quat_from_angle_axis(yaw, np.array([0.0, 1.0, 0.0], dtype=np.float32))


def positions_from_route(
    route: Sequence[np.ndarray], step_m: float
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """
    Yield densely sampled positions and directions along the provided route at
    roughly constant arc length spacing.
    """
    if not route:
        return

    last_direction = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    for start, end in zip(route, route[1:]):
        delta = end - start
        distance = np.linalg.norm(delta)
        if distance < 1e-4:
            continue
        direction = delta / distance
        last_direction = direction

        traveled = 0.0
        while traveled < distance:
            yield start + direction * traveled, direction
            traveled += step_m

    # Ensure the final waypoint is emitted at least once.
    yield np.array(route[-1], dtype=np.float32), last_direction


def strip_articulated_instances(scene_path: str) -> str:
    """
    Some scene_instance files reference articulated objects, which require Bullet.
    When Bullet isn't available, strip them out into a temp file so rendering
    still works.
    """
    if not scene_path.endswith(".scene_instance.json"):
        return scene_path
    try:
        with open(scene_path, "r") as f:
            data = json.load(f)
    except OSError:
        return scene_path

    if not data.get("articulated_object_instances"):
        return scene_path

    data = dict(data)
    data["articulated_object_instances"] = []
    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=".scene_instance.json", prefix="hab_video_"
    )
    with open(tmp.name, "w") as f:
        json.dump(data, f)
    _TEMP_SCENE_FILES.append(tmp.name)
    logger.info(
        "Stripped articulated objects from scene instance (Bullet not available): %s -> %s",
        scene_path,
        tmp.name,
    )
    return tmp.name


def should_strip_articulated(mode: str) -> bool:
    """
    Decide whether to strip articulated objects based on mode and build.
    """
    built_with_bullet = bool(getattr(habitat_sim, "built_with_bullet", False))
    if mode == "strip":
        return True
    if mode == "keep":
        if not built_with_bullet:
            logger.warning(
                "Requested to keep articulated objects, but Bullet is not available; they may fail to load."
            )
        return False
    # auto
    return not built_with_bullet


def farthest_nav_points(
    pathfinder: habitat_sim.nav.PathFinder,
    target_count: int,
    candidates_per_iter: int = 64,
) -> List[np.ndarray]:
    """
    Farthest-point sampling on the navmesh to cover the house broadly.
    """
    if target_count <= 0:
        return []
    first = pathfinder.get_random_navigable_point()
    points: List[np.ndarray] = [np.array(first, dtype=np.float32)]

    for _ in range(target_count - 1):
        best = None
        best_dist = -1.0
        for _ in range(candidates_per_iter):
            candidate = np.array(pathfinder.get_random_navigable_point(), dtype=np.float32)
            dists = [np.linalg.norm(candidate - p) for p in points]
            min_dist = min(dists) if dists else 0.0
            if min_dist > best_dist:
                best = candidate
                best_dist = min_dist
        if best is not None:
            points.append(best)
    return points


def order_waypoints_nearest(points: Sequence[np.ndarray]) -> List[np.ndarray]:
    """
    Order waypoints with a greedy nearest-neighbor pass to keep travel reasonable.
    """
    if not points:
        return []
    remaining = [np.array(p, dtype=np.float32) for p in points]
    ordered = [remaining.pop(0)]
    while remaining:
        curr = ordered[-1]
        next_idx = int(
            np.argmin([np.linalg.norm(curr - cand) for cand in remaining])
        )
        ordered.append(remaining.pop(next_idx))
    return ordered


def stitch_route_from_waypoints(
    pathfinder: habitat_sim.nav.PathFinder,
    waypoints: Sequence[np.ndarray],
    min_segment_m: float,
) -> Tuple[List[np.ndarray], float]:
    """
    Connect waypoints with shortest paths on the navmesh.
    Returns the concatenated route and total geodesic length.
    """
    if len(waypoints) < 2:
        return [np.array(wp, dtype=np.float32) for wp in waypoints], 0.0

    route: List[np.ndarray] = [np.array(waypoints[0], dtype=np.float32)]
    total_len = 0.0
    path = habitat_sim.ShortestPath()
    for start, end in zip(waypoints, waypoints[1:]):
        path.requested_start = start
        path.requested_end = end
        if not pathfinder.find_path(path):
            continue
        if path.geodesic_distance < min_segment_m or len(path.points) < 2:
            continue
        for idx in range(1, len(path.points)):
            p_prev = np.array(path.points[idx - 1], dtype=np.float32)
            p_curr = np.array(path.points[idx], dtype=np.float32)
            seg_len = np.linalg.norm(p_curr - p_prev)
            if seg_len < 1e-4:
                continue
            route.append(p_curr)
            total_len += seg_len
    return route, total_len


def build_sim(args: argparse.Namespace) -> habitat_sim.Simulator:
    sim_settings = dict(default_sim_settings)
    strip_flag = should_strip_articulated(args.articulated_mode)
    sim_settings["scene"] = (
        strip_articulated_instances(args.scene) if strip_flag else args.scene
    )
    sim_settings["scene_dataset_config_file"] = args.dataset
    sim_settings["enable_physics"] = False
    sim_settings["sensor_height"] = args.camera_height
    sim_settings["width"] = args.width
    sim_settings["height"] = args.height
    sim_settings["color_sensor"] = True
    sim_settings["depth_sensor"] = False
    sim_settings["semantic_sensor"] = False

    cfg = make_cfg(sim_settings)
    sim = habitat_sim.Simulator(cfg)

    # Navmesh is sometimes missing; build a simple one so we can move safely.
    if not sim.pathfinder.is_loaded and cfg.sim_cfg.scene_id.lower() != "none":
        nav_settings = habitat_sim.NavMeshSettings()
        nav_settings.set_defaults()
        nav_settings.agent_height = cfg.agents[0].height
        nav_settings.agent_radius = cfg.agents[0].radius
        nav_settings.include_static_objects = True
        sim.recompute_navmesh(sim.pathfinder, nav_settings)

    return sim


def render_pan_video(args: argparse.Namespace) -> None:
    LoggingContext.reinitialize_from_env()
    if args.seed is not None:
        np.random.seed(args.seed)

    sim = build_sim(args)
    agent = sim.get_agent(0)

    if not sim.pathfinder.is_loaded:
        raise RuntimeError("Navmesh failed to load; cannot plan coverage path.")

    fps = args.fps
    step_m = args.speed / fps

    bounds = sim.pathfinder.get_bounds()
    nav_extent = np.linalg.norm(np.array(bounds[1]) - np.array(bounds[0]))
    auto_waypoints = max(80, int(nav_extent * 6))
    waypoint_count = args.waypoints or auto_waypoints

    cover_points = farthest_nav_points(sim.pathfinder, waypoint_count)
    ordered_points = order_waypoints_nearest(cover_points)
    route, route_length = stitch_route_from_waypoints(
        sim.pathfinder, ordered_points, args.min_segment
    )

    if route_length <= 0.0:
        raise RuntimeError("Planned route is empty; cannot render video.")

    poses = list(positions_from_route(route, step_m))
    total_frames = (
        max(1, int(math.ceil(args.duration * fps)))
        if args.duration and args.duration > 0
        else len(poses)
    )

    if len(poses) < total_frames:
        logger.warning(
            "Only generated %d positions for %d frames; repeating the last pose.",
            len(poses),
            total_frames,
        )
        if poses:
            poses.extend([poses[-1]] * (total_frames - len(poses)))
    poses = poses[:total_frames]

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    logger.info(
        "Coverage plan: %d waypoints -> route %.1fm (~%.1fs at %.2fm/s), rendering %d frames (~%.1fs).",
        waypoint_count,
        route_length,
        route_length / args.speed,
        args.speed,
        total_frames,
        total_frames / fps,
    )

    logger.info(
        "Writing video to %s (%d frames at %dfps, speed %.2fm/s)",
        args.output,
        total_frames,
        fps,
        args.speed,
    )

    last_direction = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    with imageio.get_writer(args.output, fps=fps, macro_block_size=1) as writer:
        for idx, (position, direction) in enumerate(poses):
            if np.linalg.norm(direction) > 1e-4:
                last_direction = direction
            heading = heading_for_direction(direction, fallback=last_direction)

            state = agent.get_state()
            state.position = position
            state.rotation = heading
            agent.set_state(state, reset_sensors=True)

            observations = sim.get_sensor_observations()
            frame = observations.get("color_sensor")
            if frame is None:
                raise RuntimeError("color_sensor observations not found in simulator.")

            frame_rgb = np.asarray(frame)[..., :3]
            writer.append_data(frame_rgb)

            if (idx + 1) % fps == 0 or idx + 1 == total_frames:
                logger.info(
                    "Rendered %.1fs (%d/%d frames)",
                    (idx + 1) / fps,
                    idx + 1,
                    total_frames,
                )

    sim.close()
    logger.info("Done. Video saved to %s", args.output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a flythrough video of a Habitat scene.")
    parser.add_argument(
        "--scene",
        default="./data/test_assets/scenes/simple_room.glb",
        type=str,
        help='Scene or stage file to load (default: "./data/test_assets/scenes/simple_room.glb").',
    )
    parser.add_argument(
        "--dataset",
        default="default",
        type=str,
        metavar="DATASET",
        help='Dataset configuration file to use (default: "default").',
    )
    parser.add_argument(
        "--output",
        default="house_video.mp4",
        type=str,
        help='Video file to write (default: "house_video.mp4").',
    )
    parser.add_argument(
        "--duration",
        default=0.0,
        type=float,
        help="Duration of the output video in seconds. If <=0, duration is set to cover the full planned path automatically.",
    )
    parser.add_argument(
        "--fps",
        default=30,
        type=int,
        help="Frames per second for the output video (default: 30).",
    )
    parser.add_argument(
        "--speed",
        default=0.75,
        type=float,
        help="Camera speed in meters per second along the navmesh path (default: 0.75).",
    )
    parser.add_argument(
        "--min-segment",
        dest="min_segment",
        default=1.5,
        type=float,
        help="Minimum segment length when stitching the route, in meters (default: 1.5).",
    )
    parser.add_argument(
        "--waypoints",
        default=0,
        type=int,
        help="Number of coverage waypoints to sample on the navmesh. If <=0, a heuristic based on house size is used.",
    )
    parser.add_argument(
        "--width",
        default=800,
        type=int,
        help="Horizontal resolution of the rendered frames (default: 800).",
    )
    parser.add_argument(
        "--height",
        default=600,
        type=int,
        help="Vertical resolution of the rendered frames (default: 600).",
    )
    parser.add_argument(
        "--articulated-mode",
        choices=["auto", "keep", "strip"],
        default="auto",
        help="How to handle articulated objects in scene instances. 'auto' strips them if Bullet isn't available, 'keep' always keeps, 'strip' always strips.",
    )
    parser.add_argument(
        "--camera-height",
        dest="camera_height",
        default=1.5,
        type=float,
        help="Height of the agent's camera above the navmesh (default: 1.5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for navmesh sampling (default: None).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.fps < 1:
        raise ValueError("fps must be >= 1.")
    if args.speed <= 0:
        raise ValueError("speed must be positive.")
    if args.duration < 0:
        raise ValueError("duration must be >= 0 (0 means auto).")
    if args.waypoints < 0:
        raise ValueError("waypoints must be >= 0.")
    render_pan_video(args)
