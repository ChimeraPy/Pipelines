import chimerapy.engine as cpe
from chimerapy.orchestrator import source_node
from chimerapy.pipelines.generic_nodes.video_nodes import Video
from .data import YOLOFrame

@source_node(name="CPPipelines_HFVQA")
class HFVQA(Video):
    """The Node that mimic's data input for VQA model"""

    def __init__(self,
                  video_src: str,
                  name: str = "text",
                  data_key:str = "data",
                  frame_key = "frame",
                  text_src = "what is in the image",
                  download_video = False,
                  **kwargs
                  ) -> None:
        
        self.frame_key = frame_key
        self.data_key = data_key
        self.text_src = text_src
        super().__init__(name=name,video_src=video_src, frame_key=frame_key, loop=True, download_video=download_video, include_meta=True, **kwargs)

    def step(self) -> cpe.DataChunk:
        data_chunk = super().step()
        ret_chunk = cpe.DataChunk()
        frame_arr = data_chunk.get(self.frame_key)["value"]
        src_id = data_chunk.get("metadata")["value"]["source_name"]
        frame_count = data_chunk.get("metadata")["value"]["frame_count"]
        ret_chunk.add(
            self.data_key,
            # [question, frame],
            # for now hardcoded the questions
            [self.text_src,
            YOLOFrame(frame_arr, src_id=src_id, frame_count=frame_count)]
        )

        return ret_chunk

