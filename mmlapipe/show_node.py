from typing import Dict

import chimerapy as cp
import cv2

class ShowWindow(cp.Node):
    def step(self, data_chunks: Dict[str, cp.DataChunk]):

        for name, data_chunk in data_chunks.items():
            self.logger.debug(f"{self}: got from {name}, data={data_chunk}")

            cv2.imshow(name, data_chunk.get("frame")["value"])
            cv2.waitKey(1)
