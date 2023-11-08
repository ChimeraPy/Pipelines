from typing import Dict, List, Literal, Optional

import chimerapy.engine as cpe
from chimerapy.orchestrator import step_node
from PIL import Image 


from .data import YOLOFrame


@step_node(name="CPPipelines_HFVQANode")
class HFVQANode(cpe.Node):

    """A node to apply VQA models.

    Parameters:
    ----------
    name: str, optional (default: 'HFNode')
        The name of the node.

    model_name: str, required
        The model name of the model from Hugging Face to be applied.

    task: str, optional (default: "")
        Specify the test to perform when model task is not defined.

    device: Literal["cpu", "cuda"], optional (default: "cpu")
        The device to use for running model.

    frames_key: str, optional (default: 'frame')
        The key to access the frames in the video.
    """

    def __init__(
        self,
        name: str,
        model_name: str = "",
        task: str = "",
        device: Literal["cpu", "cuda"] = "cpu",
        data_key: str = "data",
    ):
        self.model_name = model_name
        self.device = 0 if device == "cuda" else "cpu"
        self.task = task
        self.data_key = data_key
    
        super().__init__(name=name)


    def setup(self):
        from transformers import pipeline
        
        try:
            if self.task != "":
                if self.model_name != "":
                    self.model = pipeline(task = self.task, model = self.model_name, device = self.device)
                else:
                    self.model = pipeline(task = self.task, device = self.device)
            else:
                self.model = pipeline(model = self.model_name, device = self.device)

            print(f"Successfully imported model: {self.model_name}")
        except AttributeError:
            print(f"Failed to import model: {self.model_name}. Model not found in transformers library.")

    def step(self, data_chunks: Dict[str, cpe.DataChunk]) -> cpe.DataChunk:

        ret_chunk = cpe.DataChunk()

        for _, data_chunk in data_chunks.items():
            [question, frame] = data_chunk.get(self.data_key)["value"]
        
            img = Image.fromarray(frame.arr)
            result = self.model(question=question, image = img)
            # print model output to command line
            print(result)

        return ret_chunk

