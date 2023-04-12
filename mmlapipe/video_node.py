import pathlib
import time
from typing import Literal

import chimerapy as cp
import cv2
import imutils
from chimerapy_orchestrator import source_node


@source_node
class VideoNode(cp.Node):
    def __init__(
        self,
        name: str,
        src: pathlib.Path,
        show: bool = False,
        debug: Literal["step", "stream"] = None,
    ):
        self.src = src
        self.show = show

        super().__init__(name=name, debug=debug)

    def prep(self):
        self.cap = cv2.VideoCapture(str(self.src))

    def step(self) -> cp.DataChunk:
        time.sleep(1 / 30)

        # Read data
        ret, frame = self.cap.read()

        if self.show:
            cv2.imshow(f"{self.name}", imutils.resize(frame, width=400))
            cv2.waitKey(1)

        data_chunk = cp.DataChunk()
        data_chunk.add("color", frame, "image")

        return data_chunk

    def teardown(self):
        self.cap.release()
        if self.show:
            cv2.destroyAllWindows()
