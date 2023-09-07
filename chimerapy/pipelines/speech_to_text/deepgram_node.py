from typing import Optional

from deepgram import Deepgram

import chimerapy.engine as cpe
from chimerapy.orchestrator import step_node


@step_node(name="CPPipelines_DeepgramNode")
class DeepgramNode(cpe.Node):
    def __init__(self, api_key: str, name: str = "DeepgramNode"):
        super().__init__(name=name)
        self.api_key = api_key
        self.deepgram_client: Optional[Deepgram] = None

    def setup(self) -> None:
        self.deepgram_client = Deepgram(self.api_key)
