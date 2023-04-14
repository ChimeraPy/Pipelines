import threading
from queue import Queue
from typing import Dict, List

import chimerapy as cp
import cv2
import numpy as np
from chimerapy_orchestrator import step_node


@step_node(name="MMLAPIPE_BBoxPainter")
class BBoxPainter(cp.Node):
    def __init__(
        self,
        frames_srcs: List[str],
        bboxes_srcs: List[str],
        bboxes_key: str = "bboxes",
        frames_key: str = "frame",
        multi_frame_delimiter: str = ":",
        name: str = "BBoxPainter",
        **kwargs,
    ):
        self.frames_srcs = frames_srcs
        self.bboxes_key = bboxes_key
        self.bboxes_srcs = bboxes_srcs
        self.frames_key = frames_key
        self.multi_frame_delimiter = multi_frame_delimiter
        self.frames_queue: Dict[str, Queue] = {}
        self.bboxes_queue: Dict[str, Queue] = {}

        super().__init__(name=name, **kwargs)

    def prep(self):
        for name in self.frames_srcs:
            self.frames_queue[name] = Queue()
        for name in self.bboxes_srcs:
            self.bboxes_queue[name] = Queue()
        self._start_painting()

    def _start_painting(self):
        thread = threading.Thread(target=self._paint, daemon=True)
        thread.start()

    def _paint(self):
        # Wait for 100 frames and bboxes to be available
        while True:
            if any([q.qsize() < 100 for q in self.frames_queue.values()]) and any(
                [q.qsize() < 100 for q in self.bboxes_queue.values()]
            ):
                continue
            else:
                # peek at the first frame and bbox
                items = []
                for name in self.frames_queue:
                    items.append(
                        (self.frames_queue[name].get(), self.bboxes_queue[name].get())
                    )

                # paint
                for item in items:
                    metadata_frame = item[0][1]
                    metadata_bbox = item[1][1]
                    if metadata_frame["frame_count"] == metadata_bbox["frame_count"]:
                        frame = item[0][0]
                        bboxes = item[1][0]
                        for bbox in bboxes:
                            self.paint(frame, *bbox)
                        cv2.imshow(metadata_frame["source_name"], frame)

    @staticmethod
    def paint(img: np.ndarray, t: int, l: int, w: int, h: int):
        cv2.rectangle(img, (t, l), ((t + w), (l + h)), (0, 255, 0), 2)

    def step(self, data_chunks: Dict[str, cp.DataChunk]) -> cp.DataChunk:
        ret_chunk = cp.DataChunk()

        for name, data_chunk in data_chunks.items():
            if name in self.frames_srcs:
                self.enqueue(
                    data_chunk,
                    name,
                    "metadata",
                    self.frames_queue[name],
                )
            elif name in self.bboxes_srcs:
                self.enqueue(
                    data_chunk,
                    name,
                    "metadata",
                    self.bboxes_queue[name],
                )
            else:
                for source in self.frames_srcs:
                    if (
                        data_chunk.get(
                            key := f"{name}{self.multi_frame_delimiter}{source}{self.multi_frame_delimiter}{self.frames_key}"
                        )
                        != {}
                    ):
                        metadata_key = f"{name}{self.multi_frame_delimiter}{source}{self.multi_frame_delimiter}metadata"
                        self.enqueue(
                            data_chunk,
                            key,
                            metadata_key,
                            self.frames_queue[source],
                        )
                for source in self.bboxes_srcs:
                    if (
                        data_chunk.get(
                            key := f"{name}{self.multi_frame_delimiter}{source}{self.multi_frame_delimiter}{self.bboxes_key}"
                        )
                        != {}
                    ):
                        metadata_key = f"{name}{self.multi_frame_delimiter}{source}{self.multi_frame_delimiter}metadata"
                        self.enqueue(
                            data_chunk,
                            key,
                            metadata_key,
                            self.bboxes_queue[source],
                        )

        return ret_chunk

    @staticmethod
    def enqueue(data_chunk: cp.DataChunk, key: str, metadata_key: str, queue: Queue):
        data = data_chunk.get(key)
        if data != {}:
            print("data", data)
            metadata = data_chunk.get(metadata_key)
            queue.put((data["value"], metadata["value"]))

    @staticmethod
    def paint(img: np.ndarray, t: int, l: int, w: int, h: int):
        cv2.rectangle(img, (t, l), ((t + w), (l + h)), (0, 255, 0), 2)
