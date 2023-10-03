from typing import List, Optional

import asyncio
import cv2
import logging

from g3pylib import connect_to_glasses

import chimerapy.engine as cpe
from chimerapy.orchestrator import source_node


@source_node(name="CPPipelines_G3")
class G3(cpe.Node):
    """A node that connects to a Tobii Glasses 3 instance to perform streaming and recording operations.

    Sample `gaze_data` in datachunk:
    ```JSON
    {
        "gaze2d": [0.468, 0.483],
        "gaze3d": [37.543, -18.034, 821.265],
        "eyeleft": {
            "gazeorigin": [28.799, -7.165, -23.945],
            "gazedirection": [0.0463, -0.0337, 0.998],
            "pupildiameter": 2.633
        },
        "eyeright": {
            "gazeorigin": [-28.367, -5.353, -21.426],
            "gazedirection": [0.0436, 0.00611, 0.999],
            "pupildiameter": 2.782
        }
    }
    ```

    Parameters
    ----------
    hostname: str, required
        The G3 device's serial number (e.g. TG03B-080200004381), used for connections.
    name: str, optional (default: "")
        The name of the node. If not provided, the `hostname` will be used.
    show_gaze: bool, optional (default: False)
        Whether to render the gaze circle on video frames
    frames_key: str, optional (default: "frame")
        The key to use for the video frame in the data chunk
    **kwargs
        Additional keyword arguments to pass to the Node constructor
    """

    def __init__(
        self,
        hostname: str,
        name: str = "",
        show_gaze: bool = False,
        frame_key: str = "frame",
        **kwargs,
    ):
        self.hostname = hostname
        self.show_gaze = show_gaze
        self.frame_key = frame_key

        if not name:
            name = hostname

        super().__init__(name=name, **kwargs)

    async def setup(self) -> None:
        self.g3 = await connect_to_glasses.with_hostname(
            self.hostname, using_zeroconf=True
        )

        streams = await self.g3.stream_rtsp(scene_camera=True, gaze=True)
        self.scene_stream = streams.scene_camera.decode()
        self.gaze_stream = streams.gaze.decode()

    async def step(self) -> cpe.DataChunk:
        ret_chunk = cpe.DataChunk()

        frame_data, frame_timestamp = await self.scene_stream.get()
        gaze_data, gaze_timestamp = await self.gaze_stream.get()

        # Match frame and gaze timestamps
        while gaze_timestamp is None or frame_timestamp is None:
            if frame_timestamp is None:
                frame_data, frame_timestamp = await self.scene_stream.get()
            if gaze_timestamp is None:
                gaze_data, gaze_timestamp = await self.gaze_stream.get()
        while gaze_timestamp < frame_timestamp:
            gaze_data, gaze_timestamp = await self.gaze_stream.get()
            while gaze_timestamp is None:
                gaze_data, gaze_timestamp = await self.gaze_stream.get()

        # logging.info(f"Frame timestamp: {frame_timestamp}")
        # logging.info(f"Gaze timestamp: {gaze_timestamp}")
        frame_data = frame_data.to_ndarray(format="bgr24")

        if self.show_gaze and "gaze2d" in gaze_data:
            gaze2d = gaze_data["gaze2d"]
            logging.info(f"Gaze2d: {gaze2d[0]:9.4f},{gaze2d[1]:9.4f}")

            # Convert rational (x,y) to pixel location (x,y)
            h, w = frame_data.shape[:2]
            fix = (int(gaze2d[0] * w), int(gaze2d[1] * h))

            # Draw gaze
            frame_data = cv2.circle(frame_data, fix, 10, (0, 0, 255), 3)

        ret_chunk.add(self.frame_key, frame_data, "image")
        ret_chunk.add("gaze_data", gaze_data)
        # ret_chunk.add("frame_timestamp", frame_timestamp)
        # ret_chunk.add("gaze_timestamp", gaze_timestamp)

        return ret_chunk

    @cpe.register  # .with_config(style="blocking")
    async def calibrate(self) -> bool:
        return await self.g3.calibrate.run()

    # TODO: check recorder state before making start, stop, cancel calls
    # TODO: add visual cue to indicate recorder state, e.g. UI components or greyed out buttons
    @cpe.register
    async def start_recording(self) -> bool:
        return await self.g3.recorder.start()

    @cpe.register
    async def cancel_recording(self) -> None:
        await self.g3.recorder.cancel()

    @cpe.register
    async def stop_recording(self) -> bool:
        # TODO: check if recording has started
        return await self.g3.recorder.stop()

    @cpe.register
    async def take_snapshot(self) -> bool:
        return await self.g3.recorder.snapshot()

    @cpe.register.with_config(params={"download_dir": "str"})
    async def download_latest_recording(self, download_dir: str) -> Optional[str]:
        if not await self.g3.recordings:
            return None

        return await self.g3.recordings[0].download_files(download_dir)

    @cpe.register.with_config(params={"download_dir": "str"})
    async def download_all_recordings(self, download_dir: str) -> Optional[List[str]]:
        if not await self.g3.recordings:
            return None

        return await asyncio.gather(
            recording.download_files(download_dir) for recording in self.g3.recordings
        )

    def teardown(self) -> None:
        self.g3.close()
        self.scene_stream = None
        self.gaze_stream = None
