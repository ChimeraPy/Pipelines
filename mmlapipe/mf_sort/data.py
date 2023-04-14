from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from mf_sort import Detection


@dataclass
class MFSortTrackedDetections:
    """Bounding Boxes for a frame from a video source."""

    tracker_id: Optional[int] = None
    color: Tuple[int, int, int] = (0, 255, 0)
    bboxes: List[Detection] = field(default_factory=list)

    def get_text(self):
        if self.tracker_id is not None:
            return f"Tracker: {self.tracker_id}"

    def __repr__(self):
        return f"<MFSortDetections {self.tracker_id}>"


@dataclass
class MFSortFrame:
    """A frame from a video source."""

    arr: np.ndarray
    frame_count: int
    src_id: str
    detections: List[MFSortTrackedDetections] = field(default_factory=list)

    def __repr__(self):
        return f"<Frame from {self.src_id} {self.frame_count}>"
