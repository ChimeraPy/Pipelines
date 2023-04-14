import threading
from collections import defaultdict
from queue import Queue
from typing import Dict, List

import chimerapy as cp
import cv2
import numpy as np

from mmlapipe.mf_sort.data import BBoxes, Frame


# FixMe: This is not not performant.
class ThreadedBBoxPainter(cp.Node):
    def __init__(
        self,
        bboxes_key: str = "bboxes",
        frames_key: str = "frame",
        multi_frame_delimiter: str = ":",
        name: str = "BBoxPainter",
        **kwargs,
    ):
        self.bboxes_key = bboxes_key
        self.frames_key = frames_key
        self.multi_frame_delimiter = multi_frame_delimiter
        self.frames_queue: Dict[str, Queue] = defaultdict(Queue)
        self.bboxes_queue: Dict[str, Queue] = defaultdict(Queue)
        self.thread: threading.Thread = None
        self.stopevent = threading.Event()
        super().__init__(name=name, **kwargs)

    def prep(self):
        self.thread = threading.Thread(target=self._paint, daemon=True)
        self.stopevent.clear()
        self.thread.start()

    @staticmethod
    def bbox_plot(img: np.ndarray, t: int, l: int, w: int, h: int):
        cv2.rectangle(img, (t, l), ((t + w), (l + h)), (0, 255, 0), 2)

    def _paint(self):
        while not self.stopevent.is_set():
            if not len(self.frames_queue) == len(self.bboxes_queue):
                continue
            # Wait for 100 frames and bboxes to be in the queue
            frame_queue_lengths = (q.qsize() for q in self.frames_queue.values())
            bboxes_queue_lengths = (q.qsize() for q in self.bboxes_queue.values())
            if not all(l > 100 for l in frame_queue_lengths):
                continue
            if not all(l > 100 for l in bboxes_queue_lengths):
                continue

            # Get the oldest frame and bboxes
            for window_name, queue in self.frames_queue.items():
                frame: Frame = queue.get()
                bboxes: BBoxes = self.bboxes_queue[window_name].get()
                img = frame.arr
                # Drop if doesn't match
                if frame.same_origin(bboxes):
                    for bbox in bboxes.boxes:
                        self.bbox_plot(img, *bbox.astype(int))
                    cv2.imshow(window_name, img)
                    cv2.waitKey(1)

    def step(self, data_chunks: Dict[str, cp.DataChunk]) -> cp.DataChunk:
        ret_chunk = cp.DataChunk()

        for name, data_chunk in data_chunks.items():
            if (frame := data_chunk.get(self.frames_key)) != {} and frame is not None:
                frame_obj: Frame = frame["value"]
                frames_queue = self.frames_queue[frame_obj.src_id]
                frames_queue.put(frame_obj)

            if (bboxes := data_chunk.get(self.bboxes_key)) != {} and bboxes is not None:
                bboxes_obj: List[BBoxes] = bboxes["value"]
                for bbox in bboxes_obj:
                    bboxes_queue = self.bboxes_queue[bbox.src_id]
                    bboxes_queue.put(bbox)

        return ret_chunk
