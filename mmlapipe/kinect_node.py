from typing import Literal
import pathlib
import time
import imutils

import chimerapy as cp
import cv2

class KinectNode(cp.Node):
    
    def __init__(self, name: str, kinect_data_folder: pathlib.Path, debug:Literal['step', 'stream']= None):
        self.kinect_data_folder = kinect_data_folder
        
        super().__init__(name=name, debug=debug)

    def prep(self):
        self.color_cap = cv2.VideoCapture(str(self.kinect_data_folder/'ColorStream.mp4'))
        self.depth_cap = cv2.VideoCapture(str(self.kinect_data_folder/'DepthStream.mp4'))

    def step(self) -> cp.DataChunk:

        time.sleep(1/30)

        # Read data
        ret, frame = self.color_cap.read()
        ret, depth = self.depth_cap.read()

        cv2.imshow(f'{self.name}-color', imutils.resize(frame, width=400))
        cv2.imshow(f'{self.name}-depth', imutils.resize(depth, width=400))
        cv2.waitKey(1)

        data_chunk = cp.DataChunk()
        data_chunk.add('color', frame, 'image')
        data_chunk.add('depth', depth, 'image')

        return data_chunk

    def teardown(self):
        self.color_cap.release()
        self.depth_cap.release()
        cv2.destroyAllWindows()
