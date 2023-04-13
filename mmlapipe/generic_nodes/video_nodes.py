import time
from typing import Any, Dict, Optional, Tuple, Union

import chimerapy as cp
import cv2
import imutils
import numpy as np
from chimerapy_orchestrator import sink_node, source_node


@source_node(name="MMLAPIPE_Video")
class Video(cp.Node):
    """A generic video capture node.

    This can be used to capture a local webcam or a video from the local file system

    Parameters
    ----------
    name : str, optional (default: 'VideoNode')
        The name of the node
    width: int, optional (default: 640)
        The width of the video
    height: int, optional (default: 480)
        The height of the video
    video_src: Union[str, int], optional (default: 0)
        The video source. This can be a local file path or a webcam index
    frame_rate: int, optional (default: 30)
        The frame rate of the video, in frames per second
    frame_key: str, optional (default: 'frame')
        The key to use for the frame in the data chunk
    include_meta: bool, optional (default: False)
        Whether to include the metadata in the data chunk
    **kwargs
        Additional keyword arguments to pass to the Node constructor

    Notes
    -----
        The frame_rate is not guaranteed to be exact. It is only a best effort.

    ToDo: Research and Implement a proper frame rate limiter
    """

    def __init__(
        self,
        video_src: Union[str, int] = 0,
        name: str = "VideoNode",
        width: Optional[int] = 640,
        height: Optional[int] = 480,
        frame_rate: int = 30,
        frame_key: str = "frame",
        include_meta: bool = False,
        **kwargs,
    ) -> None:
        self.video_src = video_src
        self.width = width
        self.height = height
        self.frame_rate = frame_rate
        self.include_meta = include_meta
        self.frame_key = frame_key
        self.cp: Optional[cv2.VideoCapture] = None
        super().__init__(name=name, **kwargs)

    def prep(self) -> None:
        self.cp = cv2.VideoCapture(self.video_src)

    def step(self) -> cp.DataChunk:
        data_chunk = cp.DataChunk()
        ret, frame = self.cp.read()

        if not ret:
            self.logger.error("Could not read frame from video source")
            h = self.height or 480
            w = self.width or 640
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(
                frame,
                "Read Error",
                (h // 2, w // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        if self.width or self.height:
            frame = imutils.resize(frame, width=self.width, height=self.height)

        if self.debug:
            cv2.imshow(
                f"{self.name}_{self.id[0:6]}", frame
            )  # Will the window name be unique?
            cv2.waitKey(1)

        data_chunk.add(self.frame_key, frame, "image")

        if self.include_meta:
            data_chunk.add(
                "metadata",
                {
                    "width": self.width,
                    "height": self.height,
                    "source_id": self.id,
                    "source_name": self.name,
                    "frame_rate": self.frame_rate,
                    "belongs_to_video_src": bool(ret),
                },
            )
        time.sleep(1 / self.frame_rate)
        return data_chunk

    def teardown(self) -> None:
        self.cp.release()
        if self.debug:
            cv2.destroyAllWindows()


@sink_node(name="MMLAPIPE_ShowWindows")
class ShowWindows(cp.Node):
    """A node to show the video/images in a window.

    Parameters
    ----------
    name : str, optional (default: 'ShowWindow')
        The name of the node
    frames_key: str, optional (default: 'frame')
        The key to use for the frame in the data chunk
    window_xy: Tuple[int, int], optional (default: None)
        The x, y coordinates of the first window. Use this to position the window(s)
    items_per_row: int, optional (default: 2)
        The number of rows of windows to show. This is used to position the windows
    **kwargs
        Additional keyword arguments to pass to the Node constructor
    """

    def __init__(
        self,
        name: str = "ShowWindow",
        frames_key: str = "frame",
        items_per_row: int = 2,
        window_xy: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> None:
        self.frames_key = frames_key
        self.window_xy = np.array(window_xy, dtype=int) if window_xy else None
        self.items_per_row = items_per_row
        super().__init__(name=name, **kwargs)

    def step(self, data_chunks: Dict[str, cp.DataChunk]) -> None:
        max_f_height = 0
        prev_position = None
        for idx, (name, data_chunk) in enumerate(data_chunks.items()):
            frame = data_chunk.get(self.frames_key)["value"]
            maybe_metadata = data_chunk.get("metadata")
            window_id = self._get_window_id(
                name, maybe_metadata["value"] if maybe_metadata else None
            )

            cv2.imshow(window_id, frame)
            if self.window_xy is not None:
                if prev_position is None:
                    x, y = self.window_xy
                else:
                    if idx % self.items_per_row == 0:
                        x, y = self.window_xy
                        y += prev_position[1] + max_f_height
                    else:
                        x, y = (
                            prev_position[0] + prev_position[2],
                            prev_position[1] - 76,
                        )
                        if prev_position[-1] > max_f_height:
                            max_f_height = prev_position[-1]

                prev_position = cv2.getWindowImageRect(window_id)
                cv2.moveWindow(window_id, x, y)
            cv2.waitKey(1)

    def _get_window_id(self, src_name: str, metadata: Optional[Dict[str, Any]]) -> str:
        """Get the window id for the window to be shown."""
        window_id = src_name
        if metadata:
            src_id = metadata.get("source_id", "")
            if src_id:
                window_id = f"{src_name}_{src_id[0:6]}"

        return window_id

    def teardown(self) -> None:
        cv2.destroyAllWindows()
