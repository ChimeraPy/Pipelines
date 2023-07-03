from typing import Dict, List, Literal, Optional

import chimerapy as cp
import cv2
import math
import numpy as np
import imutils
from chimerapy_orchestrator import step_node
from mmlapipe.pose.data import YOLOFrame
import os


# task: pose, seg
# scale: n, s, m, l, x
@step_node
class MultiPoseNode(cp.Node):

    """A node to apply YOLOv8 models on video src.

    Parameters:
    ----------
    name: str, optional (default: 'SaveNode')
        The name of the node.

    per_row_display: int, optional (default: 1)
        The number of videos to display per row.

    task: str, optional (default: '-pose')
        The type of task to perform. (dash to satisfy the model name)
        "" - detection
        "-pose" - pose estimation
        "-seg" - segmentation
        "-cls" - classification (not fully supported yet)

    scale: str, optional (default: 'n')
        The scale of the model. Choice: n, s, m, l, x

    device: Literal["cpu", "cuda"], optional (default: "cpu")
        The device to use for running model.

    save: bool, optional (default: False)
        Whether to save the detection/estimation results.

    save_format: str, optional (default: 'txt')
        The format in which to save the results.

    frames_key: str, optional (default: 'frame')
        The key to access the frames in the video.
    """

    def __init__(
        self,
        name: str,
        task: str = "pose",
        scale: str = "n",
        device: Literal["cpu", "cuda"] = "cpu",
        frames_key: str = "frame",
    ):
        self.task = task
        self.scale = scale
        self.device = device if device == "cpu" else 0
        self.frames_key = frames_key
        super().__init__(name=name)

    def setup(self):
        # import ultralytics YOLO
        from ultralytics import YOLO

        # load model according to params
        if self.task:
            self.model = YOLO(f"yolov8{self.scale}-{self.task}.pt")
        else:
            self.model = YOLO(f"yolov8{self.scale}.pt")

    def step(self, data_chunks: Dict[str, cp.DataChunk]):
        # Aggregate all inputs
        ret_chunk = cp.DataChunk()
        ret_frames = []
        for name, data_chunk in data_chunks.items():  # noqa: B007
            frames: List[YOLOFrame] = data_chunk.get(self.frames_key)["value"]
            for frame in frames:
                img = frame.arr
                # disable verbose output
                result = self.model(img, device=self.device, verbose=False)[0]
                # pass down both the rendered result image and the numerial results
                new_frame = YOLOFrame(
                    arr=result.plot(),
                    frame_count=frame.frame_count,
                    src_id=frame.src_id,
                    result=result,
                )
                ret_frames.append(new_frame)
        ret_chunk.add(self.frames_key, ret_frames)

        return ret_chunk