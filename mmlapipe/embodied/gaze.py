import tempfile
import typing
from typing import Dict, List, Literal, Optional

if typing.TYPE_CHECKING:
    from l2cs import Pipeline

import torch
import chimerapy as cp
import cv2
import numpy as np
import requests
from chimerapy_orchestrator import step_node
import torch.backends.cudnn as cudnn
from tqdm import tqdm

cudnn.enabled = True

@step_node(name="MMLAPIPE_GazeL2CSNet")
class GazeL2CSNet(cp.Node):
    """A node that uses L2CS-Net model from l2cs package to predict 3D gaze from a video stream.

    Parameters
    ----------
    weights: str, required
        Path or url to the weights file
    imgsz: int, optional (default: 640)
        The size of the image to be used for detection
    device: Literal["cpu", "cuda"], optional (default: "cpu")
        The device to use for detection
    name: str, optional (default: "GazeL2CSNet")
        The name of the node
    frames_key: str, optional (default: "frame")
        The key to use for the frame in the data chunk
    **kwargs
        Additional keyword arguments to pass to the Node constructor
    """

    def __init__(
        self,
        weights,
        device: Literal["cpu", "cuda"] = "cpu",
        arch: str = 'ResNet50',
        name: str = "GazeL2CSNet",
        frames_key: str = "frame",
        **kwargs,
    ) -> None:

        self.weights = weights
        self.model_params= {
            "arch": arch,
            "weights": weights,
            "device": torch.device(device),
        }
        self.frames_key = frames_key
        self.model: Optional['Pipeline'] = None
        self.debug = True
        super().__init__(name=name, **kwargs)

    def setup(self) -> None:
        from l2cs import Pipeline

        if self.model_params["weights"].startswith("http"):
            with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
                self.model_params["weights"] = self.download_weights(
                    self.model_params["weights"], f.name
                )
                self.model = Pipeline(**self.model_params)
        else:
            self.model = Pipeline(**self.model_params)

    def step(self, data_chunks: Dict[str, cp.DataChunk]) -> cp.DataChunk:
        from l2cs import render
        ret_chunk = cp.DataChunk()
        ret_frames = []

        for name, data_chunk in data_chunks.items():
            self.logger.debug(f"{self}: got from {name}, data={data_chunk}")
            frame: np.ndarray = data_chunk.get(self.frames_key)["value"]
            results = self.model.step(frame)

            if self.debug:
                vis = render(frame, results)
                cv2.imshow(name, vis)
                cv2.waitKey(1)

        return ret_chunk

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
