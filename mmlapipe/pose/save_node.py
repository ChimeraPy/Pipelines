import os
from typing import Any, Dict, Optional, List
import pandas as pd

import chimerapy as cp
import numpy as np
from chimerapy_orchestrator import sink_node

"""
    Helper function to save results as pandas df
    Code adapted from method tojson() from YOLOv8 repo results.py
"""
def to_dataframe(results, normalize=False):
    lst = []
    data = results.boxes.data.cpu().tolist()
    h, w = results.orig_shape if normalize else (1, 1)
    for i, row in enumerate(data):
        box = {'x1': row[0] / w, 'y1': row[1] / h, 'x2': row[2] / w, 'y2': row[3] / h}
        conf = row[4]
        id = int(row[5])
        name = results.names[id]
        result = {'name': name, 'class': id, 'confidence': conf, 'box': box}
        if results.masks:
            xy = results.masks.xy[i]  # numpy array
            result['segments'] = {'x': (xy[:, 0] / w).tolist(), 'y': (xy[:, 1] / h).tolist()}
        if results.keypoints is not None:
            x, y, visible = results.keypoints[i].data[0].cpu().unbind(dim=1)  # torch Tensor
            result['keypoints'] = {'x': (x / w).tolist(), 'y': (y / h).tolist(), 'visible': visible.tolist()}
        lst.append(result)

    return pd.DataFrame(lst)



@sink_node
class SaveNode(cp.Node):
    """A node to save results from Yolov8 models

    Parameters
    ----------
    name : str, optional (default: 'SaveNode')
        The name of the node
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
        name: str = "SaveNode",
        filename: str = "yolo_results",
        file_format: Optional[List[str]] = ["df","video"],
        fps: int = 30

    ) -> None:
        self.format = file_format
        self.filename = filename
        self.fps = fps
        super().__init__(name=name)

    def step(self, data_chunks: Dict[str, cp.DataChunk]) -> None:
        for name, item in data_chunks.items():
            if item: 
                if "df" in self.format:
                    results = item.get("results")
                    if results:
                        dfs = [to_dataframe(result) for result in results['value']]
                        df = pd.concat(dfs, ignore_index=True)
                        self.save_tabular(self.filename, df)
                        
                if "vid" in self.format:
                    frames = item.get("tiled")["value"]
                    if frames.size > 0:
                        self.save_video(self.filename, frames, self.fps)
                        
                # print(f"{self.format[0]} saved to {self.filename}", flush=True)
    
            else:
                print("results is None, not in data chunks", flush=True)


