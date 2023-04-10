from typing import Dict

import chimerapy as cp
import cv2
from chimerapy_orchestrator import sink_node


@sink_node(name="ShowWindowMMLAPIPE")
class ShowWindow(cp.Node):
    def __init__(self, name="ShowWindowMMLAPIPE", **kwargs):
        super(ShowWindow, self).__init__(name=name, **kwargs)

    def step(self, data_chunks: Dict[str, cp.DataChunk]):
        for name, data_chunk in data_chunks.items():
            self.logger.debug(f"{self}: got from {name}, data={data_chunk}")

            cv2.imshow(self.name, data_chunk.get("tiled")["value"])
            cv2.waitKey(1)
