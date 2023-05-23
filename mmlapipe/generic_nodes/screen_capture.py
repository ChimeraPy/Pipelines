import chimerapy as cp
import cv2
import imutils
import numpy as np
from chimerapy_orchestrator import source_node


@source_node(name="MMLAPIPE_ScreenCapture")
class ScreenCapture(cp.Node):
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

    def _get_capture(self):
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


if __name__ == "__main__":
    node = ScreenCapture()
    node.setup()
    while True:
        node.step()
