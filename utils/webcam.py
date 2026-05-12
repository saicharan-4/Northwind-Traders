"""
Webcam capture utility — grabs frames from the default camera,
runs the visual emotion analyser on each frame, and builds a
timeline of emotion probabilities over time.

Requires: opencv-python
"""

from __future__ import annotations
import time
import numpy as np
from PIL import Image
from typing import Optional, Callable


class WebcamCapture:
    """
    Captures frames from the webcam and optionally streams emotion
    predictions to build a timeline.
    """

    def __init__(self, camera_index: int = 0):
        """
        Args:
            camera_index: OpenCV camera index (0 = default webcam).
        """
        import cv2
        self.cv2    = cv2
        self.camera = cv2.VideoCapture(camera_index)
        if not self.camera.isOpened():
            raise RuntimeError(
                "Could not open webcam. Make sure a camera is connected "
                "and not in use by another application."
            )

    def capture_frame(self) -> Optional[Image.Image]:
        """
        Capture a single frame and return it as a PIL Image.
        Returns None if capture fails.
        """
        ret, frame = self.camera.read()
        if not ret:
            return None
        rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def capture_timeline(self,
                         analyser,
                         duration_seconds: float = 10.0,
                         fps: float = 1.0,
                         progress_callback: Optional[Callable] = None) -> list[dict]:
        """
        Capture frames for `duration_seconds` at `fps` frames-per-second,
        running the emotion analyser on each frame.

        Args:
            analyser:          FacialEmotionAnalyser instance.
            duration_seconds:  How long to record (seconds).
            fps:               Frames per second to sample.
            progress_callback: Optional fn(frame_idx, total_frames) for progress updates.

        Returns:
            List of timeline entries:
            [{"second": float, "top_emotion": str, **{emotion: float}}, ...]
        """
        interval   = 1.0 / fps
        total      = int(duration_seconds * fps)
        timeline   = []
        start_time = time.time()

        for i in range(total):
            frame = self.capture_frame()
            if frame is None:
                break

            elapsed = time.time() - start_time
            result  = analyser.predict(frame)
            entry   = {"second": round(elapsed, 2),
                       "top_emotion": result["top_emotion"]}
            entry.update(result["confidence"])
            timeline.append(entry)

            if progress_callback:
                progress_callback(i + 1, total)

            # throttle to requested fps
            sleep_time = interval - (time.time() - start_time - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

        return timeline

    def release(self):
        """Release the camera resource."""
        self.camera.release()

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass
