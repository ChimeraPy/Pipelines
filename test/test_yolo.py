# Import Built-in
import pathlib
import os
import copy
import time

import cv2
import pytest
import chimerapy as cp

# Internal Imports
import mmlapipe

# Test Imports
from .conftest import DATA_DIR

# Constants
TEST_VIDEO_FOLDER = DATA_DIR/'TestData'
assert TEST_VIDEO_FOLDER.exists()

@pytest.fixture
def color_cap():
    return cv2.VideoCapture(str(TEST_VIDEO_FOLDER/'test1.mp4'))

def test_yolo_main(color_cap):

    yolo = mmlapipe.YOLONode(name="yolo", debug="step", classes=['person'])

    yolo.logger = yolo.get_logger()
    yolo.setup()
    for i in range(50):

        # Simulate input feed frame
        ret, frame = color_cap.read()
        data_chunk = cp.DataChunk()
        data_chunk.add('color', frame, 'image')

        yolo.step({'test': data_chunk})

    yolo.teardown()
    yolo.shutdown()

def test_yolo_main_multiple_inputs(color_cap):

    yolo = mmlapipe.YOLONode(name="yolo", debug="step", classes=['person'])

    yolo.logger = yolo.get_logger()
    yolo.setup()
    for i in range(50):

        # Simulate input feed frame
        ret, frame = color_cap.read()
        data_chunk = cp.DataChunk()
        data_chunk.add('color', frame, 'image')

        inputs = {'test': data_chunk, 'test2': copy.copy(data_chunk), 'test3': copy.copy(data_chunk)}

        yolo.step(inputs)

    yolo.teardown()
    yolo.shutdown()
