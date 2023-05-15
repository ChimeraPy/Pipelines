from typing import Dict, List, Optional, Tuple

import chimerapy as cp
import cv2
import numpy as np
from chimerapy_orchestrator import step_node

from mmlapipe.mf_sort_tracking.data import MFSortFrame


@step_node(name="MMLAPIPE_BBoxPainter")
class BBoxPainter(cp.Node):
    """A node that paints bounding boxes on a frame."""

    def __init__(
        self,
        frames_key: str = "frame",
        draw_boxes: bool = True,
        show: bool = False,
        video_title_prefix: Optional[str] = None,
        name: str = "BBoxPainter",
        **kwargs,
    ) -> None:
        self.frames_key = frames_key
        self.draw_boxes = draw_boxes
        self.show = show
        self.video_title_prefix = video_title_prefix
        super().__init__(name=name, **kwargs)

    @staticmethod
    def bbox_plot(
        img: np.ndarray,
        t: int,
        l: int,  # noqa: E741
        w: int,
        h: int,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> None:
        cv2.rectangle(img, (t, l), ((t + w), (l + h)), color, thickness)

    def step(self, data_chunks: Dict[str, cp.DataChunk]) -> cp.DataChunk:
        ret_chunk = cp.DataChunk()
        collected_frames = []
        for name, data_chunk in data_chunks.items():  # noqa: B007
            frames: List[MFSortFrame] = data_chunk.get(self.frames_key)["value"]
            for frame in frames:
                img = frame.arr
                for det in frame.detections:
                    for bbox in det.bboxes:
                        t, l, w, h = bbox.tlwh.astype(int)  # noqa: E741

                        if self.draw_boxes:
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
            if self.show:
                cv2.imshow(frame.src_id, frame.arr)
                cv2.waitKey(1)

            if self.video_title_prefix is not None:
                self.save_video(
                    name=f"{self.video_title_prefix}_{frame.src_id}",
                    data=frame.arr,
                    fps=30,
                )

        ret_chunk.add(self.frames_key, collected_frames)

        return ret_chunk
