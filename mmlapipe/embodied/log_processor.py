from typing import Dict
import typing
import datetime

if typing.TYPE_CHECKING:
    from elp import LogProcessor

import pandas as pd
import chimerapy as cp
import cv2
import numpy as np
from chimerapy_orchestrator import step_node

CROP = {'top': 179, 'left': 580, 'bottom': 73, 'right': 356}

@step_node(name="MMLAPIPE_GEMSTEPLogProcessor")
class GEMSTEPLogProcessor(cp.Node):
    """A node that processes logs from elp package to obtain game state information.

    Parameters
    ----------
    name: str, optional (default: "GEMSTEPLogProcessor")
        The name of the node
    **kwargs
        Additional keyword arguments to pass to the Node constructor
    """

    def __init__(
        self,
        name: str = "GEMSTEPLogProcessor",
        data_key: str = "data",
        frame_key: str = 'frame',
        **kwargs,
    ) -> None:

        self.data_key = data_key
        self.frame_key = frame_key
        self.result = None
        self.debug = True
        super().__init__(name=name, **kwargs)

    def setup(self) -> None:
        from elp import LogProcessor
        self.log_processor = LogProcessor(
            corrections={'OFFSET': (-180,-260), 'AFFINE': (2.25,2)}
        )

    def step(self, data_chunks: Dict[str, cp.DataChunk]) -> cp.DataChunk:

        # Extract the data
        data = data_chunks['logs'].get(self.data_key)['value']
        frame = data_chunks['screen-capture'].get(self.frame_key)['value']

        # # Crop the frame to only get the play area
        frame = frame[CROP['top']:-CROP['bottom'], CROP['left']:-CROP['right']]

        # Process it
        if isinstance(data, pd.Series):
            self.result = self.log_processor.step(data, timestamp=data['timestamp'])
        elif isinstance(data, pd.DataFrame):
            if len(data) >= 1:
                self.result = self.log_processor.step(data, timestamp=data.iloc[-1]['timestamp'])

        # Render it
        if self.result:
            frame = self.result.render(frame)
            
        if self.debug:
            delta = datetime.datetime.now() - self.start_time
            cv2.putText(
                frame,
                str(delta),
                (0,55), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0,0,255), 
                2, 
                cv2.LINE_AA
            )
            cv2.imshow('test', frame)
            cv2.waitKey(1)

        # Package it
        ret_data_chunk = cp.DataChunk()
        ret_data_chunk.add(self.data_key, self.result) 
        ret_data_chunk.add(self.frame_key, frame, 'image')

        return ret_data_chunk
