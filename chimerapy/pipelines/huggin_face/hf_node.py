from typing import Dict, List, Literal, Optional

import chimerapy.engine as cpe
from chimerapy.orchestrator import step_node
from PIL import Image 


from .data import YOLOFrame


@step_node(name="CPPipelines_HFNode")
class HFNode(cpe.Node):

    """A node to apply Hugging Face models on video src.

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
        model_name: str,
        task: str = "",
        device: Literal["cpu", "cuda"] = "cpu",
        frames_key: str = "frame",
    ):
        self.model_name = model_name
        # self.weights = weights
        self.task = task
        self.frames_key = frames_key
    
        super().__init__(name=name)


    def setup(self):
        from transformers import pipeline
        try:
            if self.task != "":
                self.model = pipeline(task = self.task, model = self.model_name, device_map="auto")
            else:
                self.model = pipeline(model = self.model_name)

            print(f"Successfully imported model: {self.model_name}")
        except AttributeError:
            print(f"Failed to import model: {self.model_name}. Model not found in transformers library.")

    def step(self, data_chunks: Dict[str, cpe.DataChunk]) -> cpe.DataChunk:
        # Aggregate all inputs
        ret_chunk = cpe.DataChunk()

        for _, data_chunk in data_chunks.items():
            frame = data_chunk.get(self.frames_key)["value"]
        
            img = Image.fromarray(frame.arr)
            result = self.model(img)
            print(result)

            new_frame = YOLOFrame(
                arr=frame.arr,
                frame_count=frame.frame_count,
                src_id=frame.src_id,
                result=result
            )

            ret_chunk.add(self.frames_key, new_frame)

        return ret_chunk
