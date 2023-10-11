from typing import List, Optional

import asyncio
import base64
import cv2
import logging
import numpy as np

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

        (
            self.scene_queue,
            self.unsub_to_scene,
        ) = await self.g3.rudimentary.subscribe_to_scene()
        (
            self.gaze_queue,
            self.unsub_to_gaze,
        ) = await self.g3.rudimentary.subscribe_to_gaze()

        await self.g3.rudimentary.start_streams()
        self.is_recording = False

    async def step(self) -> cpe.DataChunk:
        # start on-device recording if node enters RECORDING state
        if self.state.fsm == "RECORDING" and not self.is_recording:
            await self.g3.recorder.start()
            self.is_recording = True
        elif self.state.fsm == "STOPPED" and self.is_recording:
            await self.g3.recorder.stop()
            self.is_recording = False

        ret_chunk = cpe.DataChunk()

        frame_timestamp, frame_data_b64 = await self.scene_queue.get()
        gaze_timestamp, gaze_data = await self.gaze_queue.get()

        # print("frame_data, typeof ", str(type(frame_data)), frame_data)
        # print("frame_timestamp, typeof ", str(type(frame_timestamp)), frame_timestamp)
        # print("gaze_data, typeof ", str(type(gaze_data)), gaze_data)
        # print("gaze_timestamp, typeof ", str(type(gaze_timestamp)), gaze_timestamp)

        frame_nparr = np.fromstring(base64.b64decode(frame_data_b64), dtype=np.uint8)
        frame_data = cv2.imdecode(frame_nparr, cv2.IMREAD_COLOR)

        # self.save_video("stream", frame_data, 25)

        if self.show_gaze and "gaze2d" in gaze_data:
            gaze2d = gaze_data["gaze2d"]
            logging.info(f"Gaze2d: {gaze2d[0]:9.4f},{gaze2d[1]:9.4f}")

            # Convert rational (x,y) to pixel location (x,y)
            h, w = frame_data.shape[:2]
            fix = (int(gaze2d[0] * w), int(gaze2d[1] * h))

            # Draw gaze
            frame_data = cv2.circle(frame_data, fix, 10, (0, 0, 255), 3)

        # TODO: figure out why CV2 window isn't showing
        ret_chunk.add(self.frame_key, frame_data, "image")
        # print("---------------------ADDED FRAME DATA------------------------")
        ret_chunk.add("gaze_data", gaze_data)
        # print("---------------------ADDED GAZE DATA------------------------")
        #     # ret_chunk.add("frame_timestamp", frame_timestamp)
        #     # ret_chunk.add("gaze_timestamp", gaze_timestamp)

        return ret_chunk

    @cpe.register  # .with_config(style="blocking")
    async def calibrate(self) -> bool:
        # TODO: test if calibration works separately for rudimentary and regular
        await self.g3.rudimentary.calibrate()
        return await self.g3.calibrate.run()

    # TODO: check recorder state before making start, stop, cancel calls
    # TODO: add visual cue to indicate recorder state, e.g. UI components or greyed out buttons
    # @cpe.register
    # async def start_recording(self) -> bool:
    #     return await self.g3.recorder.start()

    @cpe.register
    async def cancel_recording(self) -> None:
        await self.g3.recorder.cancel()

    # @cpe.register
    # async def stop_recording(self) -> bool:
    #     # TODO: check if recording has started
    #     return await self.g3.recorder.stop()

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

    async def teardown(self) -> None:
        # await self.streams.__aexit__()
        await self.g3.recorder.stop()
        await self.g3.rudimentary.stop_streams()
        await self.unsub_to_scene
        await self.unsub_to_gaze

        await self.g3.close()
