from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from chimerapy.pipelines.generic_nodes.audio_backends.abc import (
        AudioBackend,
    )


def get_backend(backend: str) -> Type["AudioBackend"]:
    """Get the audio backend class from a string.

    Parameters
    ----------
    backend : str
        The backend to get

    Returns
    -------
    Type[AudioBackend]
        The audio backend class
    """
    if backend == "pyaudio":
        from chimerapy.pipelines.generic_nodes.audio_backends.pyaudio import (
            PyAudioBackend,
        )

        return PyAudioBackend

    elif backend == "pvrecorder":
        from chimerapy.pipelines.generic_nodes.audio_backends.pvrecorder import (
            PVRecorderBackend,
        )

        return PVRecorderBackend

    else:
        raise ValueError(f"Invalid backend: {backend}")
