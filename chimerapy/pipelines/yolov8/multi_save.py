from typing import Dict, List

import numpy as np
import pandas as pd

import chimerapy.engine as cpe
from chimerapy.orchestrator import sink_node

from .data import YOLOFrame


def to_dataframe(results, frame_cnt, normalize=False):
    """
    Helper function to save results as pandas df
    Code adapted from method tojson() from YOLOv8 repo results.py
    """
    lst = []
    data = results.boxes.data.cpu().tolist()
    h, w = results.orig_shape if normalize else (1, 1)
    for i, row in enumerate(data):
        box = np.array([row[0] / w, row[1] / h, row[2] / w, row[3] / h])
        conf = row[4]
        id = int(row[5])
        name = results.names[id]
        result = {
            "frame_count": frame_cnt,
            "name": name,
            "class": id,
            "confidence": conf,
            "box": box,
        }
        if results.masks:
            xy = results.masks.xy[i]
            result["segments"] = np.array(
                [
                    (xy[:, 0] / w).tolist(),
                    (xy[:, 1] / h).tolist(),
                ]
            )
        if results.keypoints is not None:
            x, y, visible = (
                results.keypoints[i].data[0].cpu().unbind(dim=1)
            )  # torch Tensor
            result["keypoints"] = np.array(
                [
                    (x / w).tolist(),
                    (y / h).tolist(),
                    visible.tolist(),
                ]
            )
        lst.append(result)

    return pd.DataFrame(lst)


@sink_node(name="CPPipelines_YoloMultiSaveNode")
class MultiSaveNode(cpe.Node):
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
    file_format: str, optional (default: 'df')
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
        file_format: str = "df",
        fps: int = 30,
    ) -> None:
        self.source_key = source_key
        self.frames_key = frames_key
        self.format = file_format
        self.filename = filename
        self.fps = fps
        super().__init__(name=name)

    def step(self, data_chunks: Dict[str, cpe.DataChunk]) -> None:
        for _, data_chunk in data_chunks.items():
            frames: List[YOLOFrame] = data_chunk.get(self.frames_key)["value"]
            for frame in frames:
                if self.source_key == frame.src_id:
                    if "df" == self.format:
                        results = frame.result
                        frame_cnt = frame.frame_count
                        if results:
                            dfs = [
                                to_dataframe(result, frame_cnt)
                                for result in results
                            ]
                            df = pd.concat(dfs, ignore_index=True)
                            self.save_tabular(
                                self.filename + "-" + frame.src_id, df
                            )

                    elif "vid" == self.format:
                        img = frame.arr
                        if img.size > 0:
                            self.save_video(
                                self.filename + "-" + frame.src_id,
                                img,
                                self.fps,
                            )
