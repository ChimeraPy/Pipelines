import tempfile
import typing
from typing import Dict, Literal, Optional

if typing.TYPE_CHECKING:
    from l2cs import Pipeline

import chimerapy as cp
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from chimerapy_orchestrator import step_node

from mmlapipe.utils import download_file

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
    show: bool, optional (default: False)
        Whether to show the output
    **kwargs
        Additional keyword arguments to pass to the Node constructor
    """

    def __init__(
        self,
        weights,
        device: Literal["cpu", "cuda"] = "cpu",
        arch: str = "ResNet50",
        name: str = "GazeL2CSNet",
        frames_key: str = "frame",
        show: bool = False,
        **kwargs,
    ) -> None:

        self.weights = weights
        self.model_params = {
            "arch": arch,
            "weights": weights,
            "device": torch.device(device),
        }
        self.frames_key = frames_key
        self.model: Optional["Pipeline"] = None
        self.show = show
        self.render = None
        super().__init__(name=name, **kwargs)

    def setup(self) -> None:
        from l2cs import Pipeline, render

        if self.model_params["weights"].startswith("http"):
            with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
                self.model_params["weights"] = self.download_weights(
                    self.model_params["weights"], f.name
                )
                self.model = Pipeline(**self.model_params)
        else:
            self.model = Pipeline(**self.model_params)
        self.render = render

    def step(self, data_chunks: Dict[str, cp.DataChunk]) -> cp.DataChunk:
        frame: np.ndarray = data_chunks["camera"].get(self.frames_key)["value"]
        results = self.model.step(frame)
        vis = self.render(frame, results)
        if self.show:
            cv2.imshow("gaze", vis)
            cv2.waitKey(1)

        ret_chunk = cp.DataChunk()
        ret_chunk.add("frame", vis, "image")
        ret_chunk.add("results", results)

        return ret_chunk

    @staticmethod
    def download_weights(url: str, fname: str, chunk_size=1024) -> str:
        return download_file(
            url, fname, chunk_size=chunk_size, desc="Downloading weights"
        )
