from typing import Dict, List, Tuple

import chimerapy as cp
import cv2
import numpy as np
from chimerapy_orchestrator import step_node

from mmlapipe.mf_sort.data import MFSortFrame


@step_node(name="MMLAPIPE_BBoxPainter")
class BBoxPainter(cp.Node):
    """A node that paints bounding boxes on a frame."""

    def __init__(
        self,
        frames_key: str = "frame",
        name: str = "BBoxPainter",
        **kwargs,
    ) -> None:
        self.frames_key = frames_key
        super().__init__(name=name, **kwargs)

    @staticmethod
    def bbox_plot(
        img: np.ndarray,
        t: int,
        l: int,
        w: int,
        h: int,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> None:
        cv2.rectangle(img, (t, l), ((t + w), (l + h)), color, thickness)

    def step(self, data_chunks: Dict[str, cp.DataChunk]) -> cp.DataChunk:
        ret_chunk = cp.DataChunk()
        collected_frames = []
        for name, data_chunk in data_chunks.items():
            frames: List[MFSortFrame] = data_chunk.get(self.frames_key)["value"]
            for frame in frames:
                img = frame.arr
                for det in frame.detections:
                    for bbox in det.bboxes:
                        t, l, w, h = bbox.tlwh.astype(int)
                        self.bbox_plot(img, t, l, w, h, color=det.color)
                        if det.get_text() is not None:
                            cv2.putText(
                                img,
                                det.get_text(),
                                (t, l - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                det.color,
                                2,
                            )
                collected_frames.append(frame)

        for frame in collected_frames:
            cv2.imshow(frame.src_id, frame.arr)
            cv2.waitKey(1)

        return ret_chunk
