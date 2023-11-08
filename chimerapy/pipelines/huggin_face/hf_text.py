import chimerapy.engine as cpe
from chimerapy.orchestrator import source_node
import time

@source_node(name="CPPipelines_HFText")
class HFText(cpe.Node):
    """A text node that reads in textual input from a text file"""

    def __init__(self,
                  name: str = "text",
                  data_key:str = "data",
                  source: str = "./test.txt"

                  ) -> None:
        self.data_key = data_key
        self.source = source
        super().__init__(name=name)

    def setup(self):
        self.file = open(self.source, 'r')

    def step(self) -> cpe.DataChunk:
        if self.file:
            line = self.file.readline()
            # simulate input
            # at rate of 1 line per second
            time.sleep(1)
            if line:
                ret_chunk = cpe.DataChunk()

                ret_chunk.add(
                    self.data_key,
                    line
                )
                return ret_chunk
        return None

