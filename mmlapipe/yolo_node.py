from typing import Dict, List, Literal, Optional

import chimerapy as cp
import cv2
import imutils
import numpy as np
from chimerapy_orchestrator.utils import step_node
# Reference: https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
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


@step_node
class YOLONode(cp.Node):
    def __init__(
        self,
        name: str,
        classes: Optional[List[str]] = None,
        per_row_display=2,
        debug: Literal["step", "stream"] = None,
    ):
        # Obtain the index of the object in the original list of classes
        self.interested_classes_idx = []
        if classes:
            for obj_class in classes:
                class_index = COCO_ORIGINAL_NAMES.index(obj_class)
                self.interested_classes_idx.append(class_index)

        self.per_row_display = per_row_display

        super().__init__(name=name, debug=debug)

    def prep(self):
        # Create the YOLOv5 model
        import torch

        self.model = torch.hub.load("ultralytics/yolov5", "yolov5s", trust_repo=True)

        # Select only the interested classes
        if self.interested_classes_idx:
            self.model.classes = self.interested_classes_idx

    def step(self, data_chunks: Dict[str, cp.DataChunk]):
        # Aggreate all inputs
        imgs = []
        for name, data_chunk in data_chunks.items():
            imgs.append(data_chunk.get("color")["value"])

        # Apply the model
        results = self.model(imgs)

        # Get the rendered image
        renders = results.render()
        data_chunk = cp.DataChunk()

        for i, name in enumerate(data_chunks):
            data_chunk.add(f"xyx-{i}", results.pandas().xyxy[i])
            data_chunk.add(f"render-{i}", renders[i], "image")

        im_list_2d = []
        for i in range(len(renders), 0, -1 * self.per_row_display):
            im_list_2d.append(
                [
                    imutils.resize(renders[i - (j + 1)], width=400)
                    for j in range(self.per_row_display)
                ]
            )

        im_tiles = self.concat_tiles(im_list_2d)
        data_chunk.add("tiled", im_tiles, "image")
        return data_chunk

    def concat_tiles(self, im_list_2d):
        return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])
