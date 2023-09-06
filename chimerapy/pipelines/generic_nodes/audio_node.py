from enum import Enum
from multiprocessing import Queue
from typing import Optional

import chimerapy.engine as cpe
from chimerapy.orchestrator import source_node
from chimerapy.pipelines.generic_nodes import audio_backends
from chimerapy.pipelines.generic_nodes.audio_backends.abc import (
    AudioBackend,
    AudioFormat,
    ChunkSize,
    SampleRate,
)


class Backends(str, Enum):
    """An enum for audio backends."""

    PYAUDIO = "pyaudio"
    PVRECORDER = "pvrecorder"


def _check_backend(backend: Backends) -> None:
    audio_backends.get_backend(backend)


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
        _check_backend(backend)

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
        Backend = audio_backends.get_backend(self.options.pop("backend"))
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
