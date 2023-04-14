from typing import Dict, List, Optional

import chimerapy as cp
import numpy as np
from chimerapy_orchestrator import step_node
from mf_sort import MF_SORT
from mf_sort.detection import Detection

from mmlapipe.mf_sort.data import BBoxes
from mmlapipe.utils import requires_packages


@step_node(name="MMLAPIPE_MF_SORTTracker")
@requires_packages("mf_sort")
class MF_SORTTracker(cp.Node):
    """A node that uses MF_SORT tracker to track objects in a video stream.

    Parameters
    ----------
    source_key: str, required
        The source key to use for tracking objects in the video stream
    max_age: int, optional (default: 30)
        The maximum age of a track
    min_hits: int, optional (default: 3)
        The minimum number of hits for a track
    iou_threshold: float, optional (default: 0.7)
        The IoU threshold for a track
    name: str, optional (default: "MF_SORTTracker")
        The name of the node
    target_class: int, optional (default: 0)
        The target class to track
    bboxes_key: str, optional (default: "bboxes")
        The key to use for the bboxes in the data chunk
    **kwargs
        Additional keyword arguments to pass to the Node constructor
    """
    def __init__(
        self,
        source_key: str,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.7,
        name="MF_SORTTracker",
        target_class: int = 0,
        bboxes_key: str = "bboxes",
        **kwargs,
    ):
        self.bboxes_key = bboxes_key
        self.tracker_kwargs = {
            "max_age": max_age,
            "min_hits": min_hits,
            "iou_threshold": iou_threshold,
        }
        self.target_class = target_class
        self.source_key = source_key

        self.tracker: Optional[MF_SORT] = None
        super().__init__(name=name, **kwargs)

    def prep(self):
        self.tracker = MF_SORT(**self.tracker_kwargs)
        self.COLORS = np.random.randint(
            0, 255, size=(200, 3), dtype="int"
        )  # UpTo 200 Tracked Objects

    def step(self, data_chunks: Dict[str, cp.DataChunk]) -> cp.DataChunk:
        ret_chunk = cp.DataChunk()
        tracked_bboxes = []

        for name, data_chunk in data_chunks.items():
            bboxes: List[BBoxes] = data_chunk.get(self.bboxes_key)["value"]
            for bbox in bboxes:
                if self.source_key == bbox.src_id:
                    filtered_detections = [
                        det for det in bbox.detections if det.cls == self.target_class
                    ]
                    results = self.tracker.step(filtered_detections)
                    detections_by_track_id = {}

                    for trk, trk_id in results:
                        if trk_id not in detections_by_track_id:
                            detections_by_track_id[trk_id] = []
                        det = Detection(
                            np.squeeze(trk.tlwh.copy()),
                            trk.confidence,
                            trk.cls,
                        )
                        detections_by_track_id[trk_id].append(det)

                    for trk_id, detections in detections_by_track_id.items():
                        tracked_bboxes.append(
                            BBoxes(
                                src_id=bbox.src_id,
                                frame_count=bbox.frame_count,
                                detections=detections,
                                array=bbox.array,
                                text=f"Track ID: {trk_id}",
                                color=tuple(
                                    int(i)
                                    for i in self.COLORS[trk_id % len(self.COLORS)]
                                ),
                            )
                        )

        ret_chunk.add(self.bboxes_key, tracked_bboxes)
        return ret_chunk
