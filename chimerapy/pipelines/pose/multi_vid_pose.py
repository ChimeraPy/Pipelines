from typing import Dict, List, Literal, Optional

import chimerapy.engine as cpe
from chimerapy.orchestrator import step_node

from .data import YOLOFrame

COCO_ORIGINAL_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


@step_node(name="CPPipelines_YoloMultiPoseNode")
class MultiPoseNode(cpe.Node):

    """A node to apply YOLOv8 models on video src.

    Parameters:
    ----------
    name: str, optional (default: 'SaveNode')
        The name of the node.

    per_row_display: int, optional (default: 1)
        The number of videos to display per row.

    task: str, optional (default: 'pose')
        The type of task to perform. (dash to satisfy the model name)
        "" - detection
        "pose" - pose estimation
        "seg" - segmentation
        "cls" - classification (not fully supported yet)

    scale: str, optional (default: 'n')
        The scale of the model. Choice: n, s, m, l, x

    device: Literal["cpu", "cuda"], optional (default: "cpu")
        The device to use for running model.

    save: bool, optional (default: False)
        Whether to save the detection/estimation results.

    save_format: str, optional (default: 'txt')
        The format in which to save the results.

    frames_key: str, optional (default: 'frame')
        The key to access the frames in the video.

    classes: list, optional (default: ['person'])
        The classes to be detected by the model.
    """

    def __init__(
        self,
        name: str,
        task: str = "pose",
        scale: str = "n",
        device: Literal["cpu", "cuda"] = "cpu",
        frames_key: str = "frame",
        classes: Optional[List[str]] = None,
    ):

        if classes is None:
            classes = ["person"]  # default to person

        self.task = task
        self.scale = scale
        self.device = device if device == "cpu" else 0
        self.frames_key = frames_key
        # adapted from yolo_node.py
        self.classes_idx = []
        if classes:
            for obj_class in classes:
                class_index = COCO_ORIGINAL_NAMES.index(obj_class)
                self.classes_idx.append(class_index)

        super().__init__(name=name)

    def setup(self):
        # import ultralytics YOLO
        from ultralytics import YOLO

        # load model according to params
        if self.task:
            self.model = YOLO(f"yolov8{self.scale}-{self.task}.pt")
        else:
            self.model = YOLO(f"yolov8{self.scale}.pt")

    def step(self, data_chunks: Dict[str, cpe.DataChunk]) -> cpe.DataChunk:
        # Aggregate all inputs
        ret_chunk = cpe.DataChunk()
        ret_frames = []
        for _, data_chunk in data_chunks.items():  # noqa: B007
            frames: List[YOLOFrame] = data_chunk.get(self.frames_key)["value"]
            for frame in frames:
                img = frame.arr
                # disable verbose output and select interested classes
                result = self.model(
                    img,
                    device=self.device,
                    verbose=False,
                    classes=self.classes_idx,
                )[0]
                # pass down both the rendered result image and the numerial results
                new_frame = YOLOFrame(
                    arr=result.plot(),
                    frame_count=frame.frame_count,
                    src_id=frame.src_id,
                    result=result,
                )
                ret_frames.append(new_frame)

        ret_chunk.add(self.frames_key, ret_frames)

        return ret_chunk
