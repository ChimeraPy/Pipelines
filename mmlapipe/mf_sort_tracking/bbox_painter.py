from typing import Dict, List, Tuple, Optional

import chimerapy as cp
import cv2
import numpy as np
from chimerapy_orchestrator import step_node

from mmlapipe.mf_sort_tracking.data import MFSortFrame


@step_node(name="MMLAPIPE_BBoxPainter")
class BBoxPainter(cp.Node):
    """A node that paints bounding boxes on a frame.

    Parameters
    ----------
    frames_key: str, optional (default: "frame")
        The key to use for the frames in the data chunk
    name: str, optional (default: "BBoxPainter")
        The name of the node
    cover_region: Optional[Tuple[int, int, int, int]], optional (default: None)
        The region to cover with a box (t, l, w, h)
    video_title_prefix: Optional[str], optional (default: None)
        The prefix to use for the video title
    """

    def __init__(
        self,
        frames_key: str = "frame",
        name: str = "BBoxPainter",
        cover_region: Optional[Tuple[int, int, int, int]] = None,
        video_title_prefix: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.frames_key = frames_key
        self.cover_region = cover_region
        self.video_title_prefix = video_title_prefix
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

    def get_thickness(self, t, l, w, h):
        """Returns the thickness of the bounding box such that some region is blurred."""
        if self.cover_region is None:
            return 2
        else:
            bt, bl, bw, bh = self.cover_region
            if (bl < l) or (bt > t) or (bl + bw < l + w) or (bt + bh > t + h):
                return -1
            else:
                return 2

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
                        self.bbox_plot(img, t, l, w, h, color=det.color, thickness=self.get_thickness(t, l, w, h))

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

                if self.video_title_prefix is not None:
                    self.save_video(name=f"{self.video_title_prefix}_{frame.src_id}", data=img, fps=30)

                collected_frames.append(frame)

        for frame in collected_frames:
            cv2.imshow(frame.src_id, frame.arr)
            cv2.waitKey(1)

        return ret_chunk
