import typing

import chimerapy as cp
import cv2
import imutils
import numpy as np
from chimerapy_orchestrator import source_node

if typing.TYPE_CHECKING:
    from mss.base import MSSBase


@source_node(name="MMLAPIPE_ScreenCapture")
class ScreenCapture(cp.Node):
    """A generic video screen capture node using mss

    Parameters
    ----------
    name : str, optional (default: 'ScreenCaptureNode')
        The name of the node
    scale: float, optional (default: 0.5)
        The scale of the screen capture
    fps: int, optional (default: 30)
        The frame rate of the screen capture (unused, not guaranteed)
    frame_key: str, optional (default: 'frame')
        The key to use for the frame in the data chunk
    monitor: int, optional (default: 0)
        The monitor to capture
    """

    def __init__(
        self,
        scale: float = 0.5,
        fps: int = 30,
        frame_key: str = "frame",
        monitor: int = 0,
        name="ScreenCaptureNode",
    ):
        self.scale = scale
        self.fps = fps
        self.frame_key = frame_key
        self.capture = None
        self.monitor = monitor
        super().__init__(name=name)

    def setup(self):
        self.capture = None

    def _get_capture(self) -> "MSSBase":
        import mss

        if self.capture is None:
            self.capture = mss.mss()
        return self.capture

    def step(self) -> cp.DataChunk:
        img = self._get_capture().grab(self.capture.monitors[self.monitor])
        arr = np.array(img)
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        arr = imutils.resize(arr, width=int(arr.shape[1] * self.scale))

        data_chunk = cp.DataChunk()
        data_chunk.add(self.frame_key, arr)

        return data_chunk
