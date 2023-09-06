from abc import ABC, abstractmethod
from enum import Enum
from queue import Queue


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
