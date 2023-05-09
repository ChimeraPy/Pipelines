
import tempfile
import typing
from typing import Dict, List, Literal, Optional

if typing.TYPE_CHECKING:
    from elp import LogProcessor

import chimerapy as cp
import cv2
import numpy as np
from chimerapy_orchestrator import step_node

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
        **kwargs,
    ) -> None:

        self.data_key = data_key
        self.debug = True
        super().__init__(name=name, **kwargs)

    def setup(self) -> None:
        ...

    def step(self, data_chunks: Dict[str, cp.DataChunk]) -> cp.DataChunk:
        ...
