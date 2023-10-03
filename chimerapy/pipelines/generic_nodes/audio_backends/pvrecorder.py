import struct
from queue import Queue
from threading import Event, Thread
from typing import Any, Dict, Optional

from pvrecorder import PvRecorder

from chimerapy.pipelines.generic_nodes.audio_backends.abc import (
    AudioBackend,
    AudioFormat,
    ChunkSize,
    SampleRate,
)


class PVRecorderBackend(AudioBackend):
    """Audio backend using PVRecorder.

    Parameters
    ----------
    chunk_queue : Queue
        The queue to put the audio chunks into
    input_device_id : int, optional (default: -1)
        The input device ID to use
    audio_format : AudioFormat, optional (default: AudioFormat.INT16), ignored for this backend
        The audio format to use
    sample_rate : SampleRate, optional (default: SampleRate.RATE_44100), ignored for this backend
        The sample rate to use
    chunk_size : ChunkSize, optional (default: ChunkSize.CHUNK_1024)
        The chunk size to use
    """

    OPTION_MAPPERS = {
        ChunkSize.CHUNK_512: 512,
        ChunkSize.CHUNK_1024: 1024,
        ChunkSize.CHUNK_2048: 2048,
        ChunkSize.CHUNK_4096: 4096,
    }

    BACKEND_TYPE = "blocking"

    def __init__(
        self,
        chunk_queue: Queue = None,
        input_device_id: int = 0,
        audio_format: AudioFormat = AudioFormat.INT16,
        sample_rate: SampleRate = SampleRate.RATE_44100,
        chunk_size: ChunkSize = ChunkSize.CHUNK_1024,
    ):
        super().__init__(
            chunk_queue, input_device_id, audio_format, sample_rate, chunk_size
        )
        self.stream: Optional[PvRecorder] = None
        self.recorder_thread: Optional[Thread] = None
        self.stop_event: Optional[Event] = Event()

    def setup(self) -> None:
        self.stream = PvRecorder(
            frame_length=self.OPTION_MAPPERS[self.chunk_size],
            device_index=self.input_device_id,
            buffered_frames_count=100,
        )

    def start_streaming(self) -> None:
        self.stream.start()

    def stop_streaming(self) -> None:
        if self.stream:
            self.stream.stop()

    def teardown(self) -> None:
        if self.stream:
            self.stream.delete()

    def audio_save_info(self) -> Dict[str, Any]:
        return (
            {
                "sampwidth": 2,
                "framerate": self.stream.sample_rate,
                "nframes": self.stream.frame_length,
                "function": "save_audio_v2",
            }
            if self.stream
            else {}
        )

    def read(self) -> bytes:
        frame = self.stream.read()
        data_bytes = struct.pack("h" * len(frame), *frame)
        return data_bytes
