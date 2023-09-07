from queue import Queue
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pyaudio

from chimerapy.pipelines.generic_nodes.audio_backends.abc import (
    AudioBackend,
    AudioFormat,
    ChunkSize,
    SampleRate,
)


class PyAudioBackend(AudioBackend):
    """Audio backend using PyAudio.

    Parameters
    ----------
    chunk_queue : Queue
        The queue to put the audio chunks into
    input_device_id : int, optional (default: 0)
        The input device ID to use
    audio_format : AudioFormat, optional (default: AudioFormat.INT16)
        The audio format to use
    sample_rate : SampleRate, optional (default: SampleRate.RATE_44100)
        The sample rate to use
    chunk_size : ChunkSize, optional (default: ChunkSize.CHUNK_1024)
        The chunk size to use
    """

    OPTION_MAPPERS = {
        AudioFormat.INT16: pyaudio.paInt16,
        AudioFormat.INT32: pyaudio.paInt32,
        SampleRate.RATE_44100: 44100,
        SampleRate.RATE_48000: 48000,
        SampleRate.RATE_96000: 96000,
        ChunkSize.CHUNK_1024: 1024,
        ChunkSize.CHUNK_2048: 2048,
        ChunkSize.CHUNK_4096: 4096,
    }

    NUMPYFORMATS = {
        AudioFormat.INT16: np.int16,
        AudioFormat.INT32: np.int32,
    }

    BACKEND_TYPE = "nonblocking"

    def __init__(
        self,
        chunk_queue: Queue,
        input_device_id: int = 0,
        audio_format: AudioFormat = AudioFormat.INT16,
        sample_rate: SampleRate = SampleRate.RATE_44100,
        chunk_size: ChunkSize = ChunkSize.CHUNK_1024,
    ):
        super().__init__(
            chunk_queue, input_device_id, audio_format, sample_rate, chunk_size
        )
        self.audio: Optional[pyaudio.PyAudio] = None
        self.stream: Optional[pyaudio.Stream] = None

    def setup(self) -> None:
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=self.OPTION_MAPPERS[self.audio_format],
            channels=1,
            rate=self.OPTION_MAPPERS[self.sample_rate],
            input=True,
            frames_per_buffer=self.OPTION_MAPPERS[self.chunk_size],
            stream_callback=self._callback,
            input_device_index=self.input_device_id,
            start=False,
        )

    def start_streaming(self) -> None:
        if self.stream:
            self.stream.start_stream()

    def stop_streaming(self) -> None:
        if self.stream:
            self.stream.stop_stream()

    def teardown(self) -> None:
        if self.stream:
            self.stream.close()
            self.audio.terminate()

    def _callback(
        self, in_data, frame_count, time_info, status
    ) -> Tuple[Optional[bytes], int]:
        self.queue.put(
            np.frombuffer(in_data, dtype=self.NUMPY_FORMATS[self.audio_format])
        )
        return None, pyaudio.paContinue

    def audio_save_info(self) -> Dict[str, Any]:
        return {
            "format": self.OPTION_MAPPERS[self.audio_format],
            "rate": self.OPTION_MAPPERS[self.sample_rate],
            "function": "save_audio",
        }

    def read(self):
        raise NotImplementedError(
            "PyAudioBackend is nonblocking, use queue instead."
        )
