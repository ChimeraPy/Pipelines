from typing import List

import numpy as np
from mf_sort import Detection


class Identifiable:
    def same_origin(self, other: "Identifiable"):
        return other.identifier() == self.identifier()

    def identifier(self):
        raise NotImplementedError


class Frame(Identifiable):
    """A frame from a video source."""
    def __init__(self, arr: np.ndarray, frame_count: str, src_id: str):
        self.arr = arr
        self.frame_count = frame_count
        self.src_id = src_id

    def identifier(self):
        return {
            "src_id": self.src_id,
            "frame_count": self.frame_count,
        }

    def __repr__(self):
        return f"<Frame from {self.src_id} {self.frame_count}>"


class BBoxes(Identifiable):
    """A frame with bounding boxes."""
    def __init__(
        self,
        array: np.ndarray,
        detections: List[Detection],
        frame_count: str,
        src_id: str,
        color=None,
        text=None,
    ):
        self.array = array
        self.detections = detections
        self.frame_count = frame_count
        self.src_id = src_id
        self.color = color
        self.text = text

    def identifier(self):
        return {
            "src_id": self.src_id,
            "frame_count": self.frame_count,
        }

    def __repr__(self):
        return f"<BBoxes from {self.src_id} {self.frame_count}>"
