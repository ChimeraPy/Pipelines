from typing import Dict, List, Literal, Optional

import chimerapy as cp
import cv2
import imutils
from chimerapy_orchestrator import step_node
import os
os.environ['YOLO_VERBOSE'] = 'False'

# task: pose, seg, cls
# scale: n, s, m, l, x
@step_node
class PoseNode(cp.Node):

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
        "-cls" - classification

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
        per_row_display=1,
        task: str = "-pose",
        scale: str = 'n', 
        device: Literal["cpu", "cuda"] = "cpu",
        save: bool = False,
        frames_key: str = "frame",
        
    ):

        self.per_row_display = per_row_display
        self.task = task
        self.scale = scale
        self.device = device if device == "cpu" else 0
        self.save = save
        self.frames_key = frames_key

        super().__init__(name=name)

    def setup(self):
        # import ultralytics YOLO
        from ultralytics import YOLO
        
        # load model according to params
        self.model = YOLO("yolov8"+self.scale+self.task+".pt")

    def step(self, data_chunks: Dict[str, cp.DataChunk]):
        # Aggregate all inputs
        imgs = []
        for name, data_chunk in data_chunks.items():  # noqa: B007
            imgs.append(data_chunk.get(self.frames_key)["value"])

        # Apply the model
        results = self.model(imgs,device=self.device)

        # Get the rendered image
        renders = [x.plot() for x in results]
        ret_chunk = cp.DataChunk()

       # Add rendered frames to return chunk
        im_list_2d = []
        for i in range(len(renders), 0, -1 * self.per_row_display):
            im_list_2d.append(
                [
                    imutils.resize(renders[i - (j + 1)], width=400)
                    for j in range(self.per_row_display)
                ]
            )

        im_tiles = self.concat_tiles(im_list_2d)
        ret_chunk.add("tiled", im_tiles, "image")

        # Add model results to return chunk if save is True
        if self.save:
            ret_chunk.add("results", results, "other")

        return ret_chunk

    def concat_tiles(self, im_list_2d):
        return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])
