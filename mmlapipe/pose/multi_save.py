import os
from typing import Any, Dict, Optional, List
import pandas as pd

import chimerapy as cp
import numpy as np
from chimerapy_orchestrator import sink_node
from mmlapipe.pose.data import YOLOFrame

"""
    Helper function to save results as pandas df
    Code adapted from method tojson() from YOLOv8 repo results.py
"""


def to_dataframe(results, normalize=False):
    lst = []
    data = results.boxes.data.cpu().tolist()
    h, w = results.orig_shape if normalize else (1, 1)
    for i, row in enumerate(data):
        box = {"x1": row[0] / w, "y1": row[1] / h, "x2": row[2] / w, "y2": row[3] / h}
        conf = row[4]
        id = int(row[5])
        name = results.names[id]
        result = {"name": name, "class": id, "confidence": conf, "box": box}
        if results.masks:
            xy = results.masks.xy[i]  # numpy array
            result["segments"] = {
                "x": (xy[:, 0] / w).tolist(),
                "y": (xy[:, 1] / h).tolist(),
            }
        if results.keypoints is not None:
            x, y, visible = (
                results.keypoints[i].data[0].cpu().unbind(dim=1)
            )  # torch Tensor
            result["keypoints"] = {
                "x": (x / w).tolist(),
                "y": (y / h).tolist(),
                "visible": visible.tolist(),
            }
        lst.append(result)

    return pd.DataFrame(lst)


@sink_node
class MultiSaveNode(cp.Node):
    """A node to save results from Yolov8 models

    Parameters
    ----------
    name : str, optional (default: 'MultiSaveNode')
        The name of the node
    source_key: str
        The name of the source video that needs its results saved
    frames_key: str, optional (default: 'frame')
        The key to access the frames in the video.
    filename: str, optional (default: 'yolo_results')
        The name of the file that results will be saved to
    format: str, optional (default: 'df')
        The format that results will be saved as. Two option available
        video (mp4, param: vid) and table (csv, param: df)
    fps: int, optional (default: 30)
        Video fps that can be manually set
    """

    def __init__(
        self,
        source_key: str,
        frames_key: str = "frame",
        name: str = "SaveNode",
        filename: str = "yolo_results",
        file_format: Optional[List[str]] = ["df", "video"],
        fps: int = 30,
    ) -> None:
        self.source_key = source_key
        self.frames_key = frames_key
        self.format = file_format
        self.filename = filename
        self.fps = fps
        super().__init__(name=name)

    def step(self, data_chunks: Dict[str, cp.DataChunk]) -> None:
        for name, data_chunk in data_chunks.items():  # noqa: B007
            frames: List[YOLOFrame] = data_chunk.get(self.frames_key)["value"]
            for frame in frames:
                if self.source_key == frame.src_id:
                    if "df" in self.format:
                        results = frame.result
                        if results:
                            dfs = [to_dataframe(result) for result in results]
                            df = pd.concat(dfs, ignore_index=True)
                            self.save_tabular(self.filename + "-" + frame.src_id, df)

                    if "vid" in self.format:
                        img = frame.arr
                        if img.size > 0:
                            self.save_video(
                                self.filename + "-" + frame.src_id, img, self.fps
                            )

                    # print(f"{self.format[0]} saved to {self.filename}", flush=True)
