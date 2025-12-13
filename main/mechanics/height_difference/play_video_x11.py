#!/usr/bin/env python3
"""
Simple video player using matplotlib instead of OpenCV's Qt backend
Works better with X11 forwarding
"""
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2

def play_video_matplotlib(filename):
    """Play video using matplotlib which works better with X11"""
    cap = cv2.VideoCapture(filename)

    # Get first frame
    ret, frame = cap.read()
    if not ret:
        print(f"Cannot read video: {filename}")
        return

    # Convert BGR to RGB for matplotlib
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    im = ax.imshow(frame)
    plt.title('Press Q to close or wait for video to finish')

    def update_frame(frame_num):
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im.set_data(frame)
            return [im]
        else:
            # Video finished
            plt.close()
            return [im]

    # Get FPS
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create animation
    ani = animation.FuncAnimation(
        fig, update_frame, frames=total_frames,
        interval=1000/fps, blit=True, repeat=False
    )

    plt.show()
    cap.release()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python play_video_x11.py <video_file>")
        sys.exit(1)

    play_video_matplotlib(sys.argv[1])
