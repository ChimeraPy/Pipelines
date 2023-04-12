import logging
from typing import Dict

import chimerapy as cp
import cv2
import imutils
import numpy as np
import requests
from mf_sort.detector import Detector
from tqdm import tqdm

logging.disable(logging.WARNING)

from chimerapy_orchestrator import step_node


@step_node(name="MFSortDetector")
class MFSortDetector(cp.Node):
    def __init__(self, weights_location, name="MFSortDetector", img_size=640, **kwargs):
        self.weights_location = weights_location
        self.img_size = img_size
        self.model = None
        super().__init__(name=name, **kwargs)

    def prep(self):
        if self.weights_location.startswith("http"):
            weights_location = self.download(self.weights_location, "weights.pt")
        else:
            weights_location = self.weights_location

        self.model = Detector(
            weights=weights_location,
            imgsz=self.img_size,
        )

    @staticmethod
    def download(url, fname):
        resp = requests.get(url, stream=True)
        if resp.status_code != 200:
            raise Exception(f"Error downloading {url}")

        total = int(resp.headers.get('content-length', 0))
        with open(fname, 'wb') as file, tqdm(
                desc=fname,
                total=total,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for data in resp.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
        return file.name

    def step(self, data_chunks: Dict[str, cp.DataChunk]):
        frames = []
        for name, data_chunk in data_chunks.items():
            self.logger.debug(f"{self}: got from {name}, data={data_chunk}")

            frame = data_chunk.get("color")["value"]
            frame = imutils.resize(frame, width=640)
            detections = self.detect(0, frame)
            frame = self.paint(frame, detections)
            frames.append(frame)

        ret = cp.DataChunk()
        f = imutils.resize(frames[0], height=400, width=400)
        ret.add("tiled", f, "image")

        return ret

    def detect(self, frame_count, frame: np.ndarray):
        model_bboxes = self.model.predict([frame])[0]
        all_detections = []

        for det in model_bboxes:
            t, l, w, h = det.tlwh
            conf = det.confidence
            cls = det.cls
            all_detections.append([frame_count, cls, t, l, w, h, conf, -1, -1, -1])

        return all_detections

    @staticmethod
    def paint(frame, detections):
        for det in detections:
            trk_ID, x, y, w, h = det[1:6]
            conf = det[6]
            cls = det[1]
            frame = cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

        return frame

