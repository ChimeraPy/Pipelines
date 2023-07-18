from abc import ABC, abstractmethod
from enum import Enum
from multiprocessing import Queue
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pyaudio

import chimerapy.engine as cpe
from chimerapy.orchestrator import source_node


class Backends(str, Enum):
    """An enum for audio backends."""

    PYAUDIO = "pyaudio"


class AudioFormat(str, Enum):
    """An enum for audio formats"""

    INT16 = "INT16"
    INT32 = "INT32"


class SampleRate(str, Enum):
    """An enum for sample rates"""

    RATE_44100 = "RATE_44100"
    RATE_48000 = "RATE_48000"
    RATE_96000 = "RATE_96000"


class ChunkSize(str, Enum):
    """An enum for chunk sizes"""

    CHUNK_1024 = "CHUNK_1024"
    CHUNK_2048 = "CHUNK_2048"
    CHUNK_4096 = "CHUNK_4096"
    CHUNK_8192 = "CHUNK_8192"
    CHUNK_16384 = "CHUNK_16384"


class AudioBackend(ABC):
    """An abstract base class for audio backends."""

    OPTION_MAPPERS = {}

    def __init__(
        self,
        chunk_queue: Queue,
        input_device_id: int = 0,
        audio_format: AudioFormat = AudioFormat.INT16,
        sample_rate: SampleRate = SampleRate.RATE_44100,
        chunk_size: ChunkSize = ChunkSize.CHUNK_1024,
    ):
        self.queue = chunk_queue
        self.audio_format = audio_format
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.input_device_id = input_device_id

    @abstractmethod
    def setup(self):
        return NotImplemented

    @abstractmethod
    def start_streaming(self):
        return NotImplemented

    @abstractmethod
    def stop_streaming(self):
        return NotImplemented

    @abstractmethod
    def teardown(self):
        return NotImplemented

    @abstractmethod
    def save_kwargs(self):
        return NotImplemented


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

    NUMPY_FORMATS = {
        AudioFormat.INT16: np.int16,
        AudioFormat.INT32: np.int32,
    }

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

    def save_kwargs(self) -> Dict[str, Any]:
        return {
            "format": self.OPTION_MAPPERS[self.audio_format],
            "rate": self.OPTION_MAPPERS[self.sample_rate],
        }


@source_node(name="CPPipelines_AudioNode")
class AudioNode(cpe.Node):
    """A generic audio node, which can be used to stream audio (mono, one channel) from a device.

    Parameters
    ----------
    input_device_id : int, optional (default: 0)
        The input device ID to use
    backend : Backends, optional (default: Backends.PYAUDIO)
        The backend to use
    audio_format : AudioFormat, optional (default: AudioFormat.INT16)
        The audio format to use
    sample_rate : SampleRate, optional (default: SampleRate.RATE_44100)
        The sample rate to use
    chunk_size : ChunkSize, optional (default: ChunkSize.CHUNK_1024)
        The chunk size to use
    save_name : str, optional (default: "test")
        The name of the file to save the audio to
    name : str, optional (default: "AudioNode")
        The name of the node
    """

    backends = {Backends.PYAUDIO: PyAudioBackend}

    def __init__(
        self,
        input_device_id: int = 0,
        backend: Backends = Backends.PYAUDIO,
        audio_format: AudioFormat = AudioFormat.INT16,
        sample_rate: SampleRate = SampleRate.RATE_44100,
        chunk_size: ChunkSize = ChunkSize.CHUNK_1024,
        save_name: Optional[str] = None,
        chunk_key: str = "audio",
        name="AudioNode",
    ):
        if backend not in self.backends:
            raise ValueError(f"Invalid backend: {backend}")

        self.options = {
            "input_device_id": input_device_id,
            "backend": backend,
            "audio_format": audio_format,
            "sample_rate": sample_rate,
            "chunk_size": chunk_size,
        }
        self.queue: Optional[Queue] = None
        self.backend: Optional[AudioBackend] = None
        self.started = False
        self.save_name = save_name
        self.chunk_key = chunk_key
        super().__init__(name=name)

    def setup(self) -> None:
        self.queue = Queue()
        Backend = self.backends[self.options.pop("backend")]
        self.backend = Backend(
            self.queue,
            audio_format=self.options["audio_format"],
            sample_rate=self.options["sample_rate"],
            chunk_size=self.options["chunk_size"],
        )
        self.backend.setup()

    def step(self) -> cpe.DataChunk:
        if not self.started:
            self.backend.start_streaming()
            self.started = True
        audio_data = self.queue.get()

        if self.save_name is not None:
            self.save_audio(
                name=self.save_name,
                data=audio_data,
                channels=1,
                **self.backend.save_kwargs(),
            )

        ret_chunk = cpe.DataChunk()
        ret_chunk.add(self.chunk_key, audio_data)

        return ret_chunk

    def teardown(self) -> None:
        self.backend.stop_streaming()
        self.backend.teardown()
