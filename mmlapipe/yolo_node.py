from typing import Dict, List, Literal, Optional

import chimerapy as cp
import numpy as np
import cv2
import imutils

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

class YOLONode(cp.Node):

    def __init__(self, name:str, classes:Optional[List[str]]=None, debug:Literal['step', 'stream']=None):
        # Obtain the index of the object in the original list of classes
        self.interested_classes_idx = []
        if classes:
            for obj_class in classes: 
                class_index = COCO_ORIGINAL_NAMES.index(obj_class)
                self.interested_classes_idx.append(class_index)
        
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
            imgs.append(data_chunk.get('color')['value'])

        # Apply the model
        results = self.model(imgs)

        # Get the rendered image
        renders = results.render()
        data_chunk = cp.DataChunk()

        for i, name in enumerate(data_chunks):

            data_chunk.add(f'xyx-{i}', results.pandas().xyxy[i])
            data_chunk.add(f'render-{i}', renders[i], 'image')

            cv2.imshow(f'{name}-{i}', imutils.resize(renders[i], width=1000))
            cv2.waitKey(1)
        
        return data_chunk

