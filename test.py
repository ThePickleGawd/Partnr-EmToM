#!/usr/bin/env python3
"""
Quick display sanity check for manual_obs popups.

Behavior:
- If `outputs/manual_obs/*.png` exists, load the newest file and show it.
- Otherwise, show a synthetic test pattern.
The goal is just to verify that matplotlib can open a window with your current backend/ DISPLAY.
"""

import glob
import os
import sys
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Avoid toolbar initialization issues on some Qt builds (_Stack errors).
matplotlib.rcParams["toolbar"] = "none"


def pick_image() -> Tuple[np.ndarray, str]:
    """
    Return an image array and a human-readable source string.
    Prefers saved manual_obs PNGs; falls back to a synthetic pattern.
    """
    candidates = glob.glob("outputs/manual_obs/*.png")
    if candidates:
        path = sorted(candidates)[-1]
        try:
            img = plt.imread(path)
            return img, f"Loaded saved manual_obs: {path}"
        except Exception as exc:  # pragma: no cover
            return (
                make_pattern(),
                f"Failed to load {path} ({exc!r}); showing synthetic pattern instead.",
            )
    return make_pattern(), "No saved manual_obs images found; showing synthetic pattern."


def make_pattern() -> np.ndarray:
    """Generate a simple RGB gradient pattern."""
    h, w = 240, 320
    y = np.linspace(0, 1, h)[:, None]
    x = np.linspace(0, 1, w)[None, :]
    r = np.broadcast_to(x, (h, w))
    g = np.broadcast_to(y, (h, w))
    b = 0.5 * np.ones((h, w))
    return np.stack([r, g, b], axis=-1)


def main() -> int:
    backend = matplotlib.get_backend()
    print(f"matplotlib backend: {backend}")
    if os.environ.get("DISPLAY") is None and backend.lower() not in {
        "agg",
        "module://matplotlib_inline.backend_inline",
    }:
        print("Warning: DISPLAY is not set; popup windows may not appear.")

    img, msg = pick_image()
    print(msg)

    plt.figure("Manual Obs Popup Test", figsize=(8, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()
    print("Showing image... close the window or press Ctrl+C to exit.")
    try:
        plt.show(block=True)
    except Exception as exc:  # pragma: no cover
        # Graceful fallback: save to file so user can open manually.
        fallback = "/tmp/manual_obs_test.png"
        plt.imsave(fallback, img)
        print(f"Popup failed ({exc}). Saved image to {fallback}.")
        print(
            "If you want a live window, try setting MPLBACKEND=TkAgg (or another GUI backend) "
            "and ensure DISPLAY is set with GUI libraries installed."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
