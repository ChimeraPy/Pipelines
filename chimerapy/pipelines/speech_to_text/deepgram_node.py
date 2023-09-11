import json
from typing import Any, Dict, Optional

from deepgram import Deepgram

import chimerapy.engine as cpe
from chimerapy.orchestrator import step_node


@step_node(name="CPPipelines_DeepgramNode")
class DeepgramNode(cpe.Node):
    """A node which transcribes live audio using Deepgram.

    Parameters
    ----------
    api_key : str
        The Deepgram API key
    name : str, optional (default: "DeepgramNode")
        The name of the node
    chunk_key : str, optional (default: "audio_chunk")
        The key of the audio chunk in the data chunk
    deepgram_options : Dict[str, Any], optional (default: None)
        Options to pass to the Deepgram client(deepgram.transcription.live)
    """

    def __init__(
        self,
        api_key: str,
        name: str = "DeepgramNode",
        chunk_key: str = "audio_chunk",
        deepgram_options: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name=name)
        self.api_key = api_key
        self.deepgram_client: Optional[Deepgram] = None
        self.transcribers = {}
        self.chunk_key = chunk_key
        self.deepgram_options = deepgram_options or {}

    async def setup(self) -> None:
        """Setup the Deepgram client."""
        self.deepgram_client = Deepgram(self.api_key)

    async def step(
        self, data_chunks: Dict[str, cpe.DataChunk]
    ) -> cpe.DataChunk:
        """Transcribe the audio chunks."""
        for name, data_chunk in data_chunks.items():
            await self._create_transcription(name)

            transcriber = self.transcribers[name]
            audio_chunk = data_chunk.get(self.chunk_key)["value"]
            transcriber.send(audio_chunk)

    async def _create_transcription(self, name) -> None:
        """Create a transcription for the given name."""
        if name not in self.transcribers:
            try:
                self.transcribers[
                    name
                ] = await self.deepgram_client.transcription.live(
                    self.deepgram_options
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to create transcription for {name}: {e}"
                )
                return

            transcriber = self.transcribers[name]
            transcriber.registerHandler(
                transcriber.event.CLOSE,
                lambda c: print(f"Connection closed with code {c}."),
            )
            transcriber.registerHandler(
                transcriber.event.ERROR, lambda e: print(f"Error: {e}")
            )
            transcriber.registerHandler(
                transcriber.event.TRANSCRIPT_RECEIVED,
                lambda t: self._save_transcript(name, t),
            )
            self.logger.info(f"Created transcription for {name}")

    def _save_transcript(self, name, response) -> None:
        """Save the transcript to a csv file."""
        transcript_data = {
            "transcript": response["channel"]["alternatives"][0]["transcript"],
            "conf": response["channel"]["alternatives"][0]["confidence"],
            "start": response["start"],
            "end": response["start"] + response["duration"],
            "deepgram_json": json.dumps(response, indent=0),
        }
        self.save_tabular(name, transcript_data)

    async def teardown(self) -> None:
        """Finish all transcriptions."""
        for transcriber in self.transcribers.values():
            await transcriber.finish()
