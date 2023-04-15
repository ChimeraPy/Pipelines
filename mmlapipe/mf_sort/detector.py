import tempfile
from typing import Dict, List, Literal, Optional

import chimerapy as cp
import cv2
import numpy as np
import requests
from chimerapy_orchestrator import step_node
from mf_sort.detector import Detector
from tqdm import tqdm

from mmlapipe.mf_sort.data import MFSortFrame, MFSortTrackedDetections
from mmlapipe.utils import requires_packages


@step_node(name="MMLAPIPE_MFSortDetector")
@requires_packages("mf_sort", "requests")
class MFSortDetector(cp.Node):
    """A node that uses Yolo model from mf_sort package to detect objects in a video stream.

    Parameters
    ----------
    weights: str, required
        Path or url to the weights file
    imgsz: int, optional (default: 640)
        The size of the image to be used for detection
    device: Literal["cpu", "cuda"], optional (default: "cpu")
        The device to use for detection
    conf_thresh: float, optional (default: 0.4)
        The confidence threshold for detection
    iou_thresh: float, optional (default: 0.5)
        The IoU threshold for detection
    name: str, optional (default: "MFSortDetector")
        The name of the node
    frames_key: str, optional (default: "frame")
        The key to use for the frame in the data chunk
    **kwargs
        Additional keyword arguments to pass to the Node constructor
    """

    def __init__(
        self,
        weights,
        imgsz: int = 640,
        device: Literal["cpu", "cuda"] = "cpu",
        conf_thresh: float = 0.4,
        iou_thresh: float = 0.5,
        name: str = "MFSortDetector",
        frames_key: str = "frame",
        **kwargs,
    ) -> None:
        self.weights = weights
        self.detector_kwargs = {
            "weights": weights,
            "imgsz": imgsz,
            "device": device,
            "conf_thresh": conf_thresh,
            "iou_thresh": iou_thresh,
        }
        self.frames_key = frames_key
        self.detector: Optional[Detector] = None
        super().__init__(name=name, **kwargs)

    def prep(self) -> None:
        if self.detector_kwargs["weights"].startswith("http"):
            with tempfile.NamedTemporaryFile(suffix=".pt") as f:
                self.detector_kwargs["weights"] = self.download_weights(
                    self.detector_kwargs["weights"], f.name
                )
                self.detector = Detector(**self.detector_kwargs)
        else:
            self.detector_kwargs["device"] = "cuda"
            self.detector = Detector(**self.detector_kwargs)

    def step(self, data_chunks: Dict[str, cp.DataChunk]) -> cp.DataChunk:
        ret_chunk = cp.DataChunk()
        ret_frames = []
        for name, data_chunk in data_chunks.items():
            self.logger.debug(f"{self}: got from {name}, data={data_chunk}")
            frames: List[MFSortFrame] = data_chunk.get(self.frames_key)["value"]
            for frame in frames:
                img = frame.arr
                detections = self.detector.predict([img])[0]
                new_frame = MFSortFrame(
                    arr=img,
                    frame_count=frame.frame_count,
                    src_id=frame.src_id,
                    detections=[
                        MFSortTrackedDetections(tracker_id=None, bboxes=detections)
                    ],
                )

                if self.debug:
                    for detection in detections:
                        self.paint(img, *detection.tlwh.astype(int))
                    cv2.imshow(name, img)
                    cv2.waitKey(1)

                ret_frames.append(new_frame)

        ret_chunk.add(self.frames_key, ret_frames)

        return ret_chunk

    @staticmethod
    def paint(img: np.ndarray, t: int, l: int, w: int, h: int) -> None:
        cv2.rectangle(img, (t, l), ((t + w), (l + h)), (0, 255, 0), 2)

    @staticmethod
    def download_weights(url: str, fname: str, chunk_size=1024) -> str:
        resp = requests.get(url, stream=True)
        total = int(resp.headers.get("content-length", 0))

        with open(fname, "wb") as file, tqdm(
            desc="Downloading weights",
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in resp.iter_content(chunk_size=chunk_size):
                size = file.write(data)
                bar.update(size)

        return fname
