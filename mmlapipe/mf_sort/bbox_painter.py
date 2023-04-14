from typing import Dict, List

import chimerapy as cp
import cv2
import numpy as np
from chimerapy_orchestrator import step_node

from mmlapipe.mf_sort.data import BBoxes


@step_node(name="MMLAPIPE_BBoxPainter")
class BBoxPainter(cp.Node):
    """A node that paints bounding boxes on a frame."""

    def __init__(
        self,
        bboxes_key: str = "bboxes",
        frames_key: str = "frame",
        name: str = "BBoxPainter",
        **kwargs,
    ):
        self.bboxes_key = bboxes_key
        self.frames_key = frames_key
        super().__init__(name=name, **kwargs)

    @staticmethod
    def bbox_plot(img: np.ndarray, t, l, w, h, color=(0, 255, 0), thickness=2):
        cv2.rectangle(img, (t, l), ((t + w), (l + h)), color, thickness)

    def step(self, data_chunks: Dict[str, cp.DataChunk]) -> cp.DataChunk:
        ret_chunk = cp.DataChunk()
        for name, data_chunk in data_chunks.items():
            bboxes: List[BBoxes] = data_chunk.get(self.bboxes_key)["value"]
            for bbox in bboxes:
                img = bbox.array
                for det in bbox.detections:
                    t, l, w, h = det.tlwh.astype(int)
                    self.bbox_plot(img, t, l, w, h, color=bbox.color)
                    if bbox.text is not None:
                        cv2.putText(
                            img,
                            bbox.text,
                            (t, l - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            bbox.color,
                            2,
                        )

                cv2.imshow(bbox.src_id, img)
                cv2.waitKey(1)

        return ret_chunk
