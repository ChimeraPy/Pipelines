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
        put_text: bool = True,
        show: bool = False,
        video_title_prefix: Optional[str] = None,
        name: str = "BBoxPainter",
        paint_classes: Optional[List[int]] = None,
        threshold_bbox_distance: Optional[int] = None,
        **kwargs,
    ) -> None:
        self.frames_key = frames_key
        self.draw_boxes = draw_boxes
        self.put_text = put_text
        self.show = show
        self.video_title_prefix = video_title_prefix
        self.paint_classes = paint_classes
        self.threshold_bbox_distance = threshold_bbox_distance
        self.tracker_distances = None
        super().__init__(name=name, **kwargs)

    def setup(self):
        if self.threshold_bbox_distance:
            self.tracker_distances = {}

    @staticmethod
    def _braw_boxes(
        img: np.ndarray,
        t: int,
        l: int,  # noqa: E741
        w: int,
        h: int,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> None:
        cv2.rectangle(img, (t, l), ((t + w), (l + h)), color, thickness)

    def _paint_classes(self, frame: MFSortFrame):
        for bbox in frame.all_boxes:
            if bbox.cls in self.paint_classes:
                t, l, w, h = bbox.tlwh.astype(int)  # noqa: E741
                self._braw_boxes(
                    frame.arr, t, l, w, h, color=(0, 255, 0), thickness=-1
                )

    @staticmethod
    def _put_text(img, t, l, text, color) -> None:  # noqa: E741
        if text is not None:
            cv2.putText(
                img,
                text,
                (t, l - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    def step(self, data_chunks: Dict[str, cp.DataChunk]) -> cp.DataChunk:
        ret_chunk = cp.DataChunk()
        collected_frames = []
        for name, data_chunk in data_chunks.items():  # noqa: B007
            frames: List[MFSortFrame] = data_chunk.get(self.frames_key)["value"]
            for frame in frames:
                img = frame.arr
                for det in frame.detections:
                    if self.threshold_bbox_distance:
                        if det.tracker_id not in self.tracker_distances:
                            self.tracker_distances[det.tracker_id] = []

                        if len(self.tracker_distances[det.tracker_id]) == 1000:
                            self.tracker_distances[det.tracker_id].pop(0)

                        self.tracker_distances[det.tracker_id].append(
                            det.bboxes[0].tlwh.astype(int)
                        )

                        if len(self.tracker_distances[det.tracker_id]) > 1:
                            first_box = self.tracker_distances[det.tracker_id][
                                0
                            ]
                            last_box = self.tracker_distances[det.tracker_id][
                                -1
                            ]
                            ft, fl, fw, fh = first_box
                            lt, ll, lw, lh = last_box
                            fx1, fy1, fx2, fy2 = ft, fl, ft + fw, fl + fh
                            lx1, ly1, lx2, ly2 = lt, ll, lt + lw, ll + lh
                            distance = np.sqrt(
                                (fx1 - lx1) ** 2 + (fy1 - ly1) ** 2
                            )
                            if distance > self.threshold_bbox_distance:
                                self._braw_boxes(
                                    img,
                                    lt,
                                    ll,
                                    lw,
                                    lh,
                                    color=det.color,
                                    thickness=2,
                                )

                    for bbox in det.bboxes:
                        t, l, w, h = bbox.tlwh.astype(int)  # noqa: E741

                        if self.draw_boxes:
                            self._braw_boxes(img, t, l, w, h, color=det.color)

                        if self.put_text:
                            self._put_text(img, t, l, det.get_text(), det.color)

                    if self.paint_classes is not None:
                        self._paint_classes(frame)

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
