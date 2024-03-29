# Import Built-in
import copy
import os
import pathlib

import cv2
import pytest

import chimerapy.engine as cpe

# Internal Imports
from chimerapy.pipelines.yolo_node import YOLONode

# Test Imports

# Constants
CWD = pathlib.Path(os.path.abspath(__file__)).parent
GIT_ROOT = pathlib.Path(os.path.abspath(__file__)).parent.parent
DATA_DIR = GIT_ROOT / "data"
TEST_VIDEO_FOLDER = DATA_DIR / "TestData"
assert TEST_VIDEO_FOLDER.exists()


@pytest.fixture
def color_cap():
    return cv2.VideoCapture(str(TEST_VIDEO_FOLDER / "test1.mp4"))


def test_yolo_main(color_cap):

    yolo = YOLONode(name="yolo", debug="step", classes=["person"])

    yolo.logger = yolo.get_logger()
    yolo.setup()
    for i in range(50):  # noqa: B007

        # Simulate input feed frame
        ret, frame = color_cap.read()
        data_chunk = cpe.DataChunk()
        data_chunk.add("frame", frame, "image")

        yolo.step({"test": data_chunk})

    yolo.teardown()
    yolo.shutdown()


def test_yolo_main_multiple_inputs(color_cap):

    yolo = YOLONode(name="yolo", debug="step", classes=["person"])

    yolo.logger = yolo.get_logger()
    yolo.setup()
    for i in range(50):  # noqa: B007

        # Simulate input feed frame
        ret, frame = color_cap.read()
        data_chunk = cpe.DataChunk()
        data_chunk.add("frame", frame, "image")

        inputs = {
            "test": data_chunk,
            "test2": copy.copy(data_chunk),
            "test3": copy.copy(data_chunk),
        }

        yolo.step(inputs)

    yolo.teardown()
    yolo.shutdown()
