from typing import TYPE_CHECKING, Callable, Dict, List, Literal, Optional, Type

import chimerapy as cp
import cv2

if TYPE_CHECKING:
    import torch
    from ultralytics import YOLO
    from ultralytics.yolo.data.augment import LetterBox
    from ultralytics.yolo.utils.plotting import Annotator

    from .data import MFSortFrame

from chimerapy_orchestrator import step_node


@step_node(name="MMLAPIPE_Anonymizer")
class Anonymizer(cp.Node):
    """A node that blocks/blurs people the video stream using a YoloV8 instance segmentation masks.

    Parameters
    ----------
    model_name : str, optional, default="yolov8m-seg"
        The name of the YOLO model to use
    frames_key : str, optional, default="frame"
        The key of the frames in the data chunk, by default "frame"
    device : Literal["cpu", "cuda"], optional, default="cpu"
        The device to use
    show : bool, optional, default=False
        Whether to show the video stream
    alpha : float, optional, default=0.5
        The alpha value for the overlay
    name : str, optional, default="Anonymizer"
        The name of the node
    """

    def __init__(
        self,
        model_name: str = "yolov8m-seg",
        frames_key: str = "frame",
        device: Literal["cpu", "cuda"] = "cpu",
        show: bool = False,
        alpha: float = 0.5,
        name: str = "Anonymizer",
    ):
        self.model_name = model_name
        self.frames_key = frames_key
        self.device = device
        self.model: Optional[YOLO] = None
        self.show: bool = show
        self.alpha: float = alpha

        self.Annotator: Optional[Type[Annotator]] = None
        self.LetterBox: Optional[Type[LetterBox]] = None
        self.colors: Optional[Callable] = None
        self.torch: Optional[torch] = None

        super().__init__(name=name)

    def setup(self) -> None:
        import torch
        from ultralytics import YOLO
        from ultralytics.yolo.data.augment import LetterBox
        from ultralytics.yolo.utils.plotting import Annotator, colors

        self.model = YOLO(self.model_name)
        self.model.to(self.device)

        self.Annotator = Annotator
        self.LetterBox = LetterBox
        self.colors = colors
        self.torch = torch

    def step(self, data_chunks: Dict[str, cp.DataChunk]) -> cp.DataChunk:
        collected_frames: List[MFSortFrame] = []

        for _, data_chunk in data_chunks.items():
            frames: List[MFSortFrame] = data_chunk.get(self.frames_key)["value"]
            for frame in frames:
                collected_frames.append(frame)

        arrays = [frame.arr for frame in collected_frames]
        results = self.model.predict(
            arrays, classes=0, verbose=False
        )  # predict only people

        for frame, result in zip(collected_frames, results):
            annotator = self.Annotator(frame.arr, line_width=0, pil=False)
            pred_masks = result.masks
            pred_boxes = result.boxes

            if pred_masks is not None:

                img = self.LetterBox(pred_masks.shape[1:])(
                    image=annotator.result()
                )

                img_gpu = (
                    self.torch.as_tensor(
                        img,
                        dtype=self.torch.float16,
                        device=pred_masks.data.device,
                    )
                    .permute(2, 0, 1)
                    .flip(0)
                    .contiguous()
                    / 255
                )

                idx = pred_boxes.cls if pred_boxes else range(len(pred_masks))
                annotator.masks(
                    pred_masks.data,
                    colors=[self.colors(x, True) for x in idx],
                    im_gpu=img_gpu,
                    alpha=self.alpha,
                )

            if self.show:
                cv2.imshow(frame.src_id, annotator.result())
                cv2.waitKey(1)

        ret_chunk = cp.DataChunk()
        ret_chunk.add(self.frames_key, collected_frames)

        return ret_chunk
