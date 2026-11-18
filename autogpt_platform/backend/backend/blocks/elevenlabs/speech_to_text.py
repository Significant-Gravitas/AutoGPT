"""Transcribe speech in an audio or video file, labelling who said what."""

import mimetypes
from io import BytesIO
from pathlib import Path
from typing import Optional

from aiohttp import ClientTimeout
from pydantic import BaseModel

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.blocks.elevenlabs._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    ElevenLabsCredentials,
    ElevenLabsCredentialsInput,
)
from backend.data.execution import ExecutionContext
from backend.data.model import CredentialsField, NodeExecutionStats, SchemaField
from backend.util.file import MediaFileType, get_exec_file_path, store_media_file
from backend.util.request import Requests

SCRIBE_URL = "https://api.elevenlabs.io/v1/speech-to-text"
SCRIBE_MODEL = "scribe_v3"
# Scribe's batch price, $0.20 per audio hour (elevenlabs.io/pricing/api, 18 Nov 2026).
USD_PER_AUDIO_SECOND = 0.20 / 3600
SCRIBE_TIMEOUT = ClientTimeout(total=15 * 60)


class SpeechSegment(BaseModel):
    speaker: str
    start: float
    end: float
    text: str


class ScribeWord(BaseModel):
    text: str
    type: str = "word"
    start: Optional[float] = None
    end: Optional[float] = None
    speaker_id: Optional[str] = None


class ScribeTranscript(BaseModel):
    text: str
    words: list[ScribeWord] = []
    audio_duration_secs: Optional[float] = None


class TranscribeSpeechBlock(Block):
    """Transcribe speech with ElevenLabs Scribe and split it by speaker."""

    class Input(BlockSchemaInput):
        credentials: ElevenLabsCredentialsInput = CredentialsField(
            description="ElevenLabs API key"
        )
        audio: MediaFileType = SchemaField(
            description="Audio or video file to transcribe (URL, data URI or workspace file)"
        )
        language: str = SchemaField(
            description="Spoken language as an ISO 639 code, such as 'en' or 'nl'; leave empty to detect it",
            default="",
            advanced=True,
        )
        max_speakers: Optional[int] = SchemaField(
            description="Most speakers in the recording, up to 32; leave empty to let the model decide",
            default=None,
            ge=1,
            le=32,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        transcript: str = SchemaField(description="Everything that was said")
        segments: list[SpeechSegment] = SchemaField(
            description="Each stretch of speech by one speaker, with start and end in seconds"
        )
        speakers: list[str] = SchemaField(
            description="The speakers heard, in order of first appearance"
        )

    def __init__(self):
        super().__init__(
            id="3936f95c-30aa-4800-91d6-653dff50edb3",
            description=(
                "Transcribes speech in an audio or video file and labels who said "
                "what. Returns the full transcript, each speaker's segments with "
                "timestamps, and the speakers heard"
            ),
            categories={BlockCategory.AI, BlockCategory.MULTIMEDIA},
            input_schema=TranscribeSpeechBlock.Input,
            output_schema=TranscribeSpeechBlock.Output,
            test_input={
                "audio": "data:audio/mpeg;base64,SUQz",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("transcript", "Hi there. Hello."),
                (
                    "segments",
                    [
                        SpeechSegment(
                            speaker="speaker_0", start=0.0, end=0.9, text="Hi there."
                        ),
                        SpeechSegment(
                            speaker="speaker_1", start=1.2, end=1.6, text="Hello."
                        ),
                    ],
                ),
                ("speakers", ["speaker_0", "speaker_1"]),
            ],
            test_mock={
                "read_audio": lambda *args, **kwargs: ("clip.mp3", b"ID3"),
                "transcribe": lambda *args, **kwargs: ScribeTranscript(
                    text="Hi there. Hello.",
                    audio_duration_secs=1.6,
                    words=[
                        ScribeWord(
                            text="Hi", start=0.0, end=0.3, speaker_id="speaker_0"
                        ),
                        ScribeWord(
                            text=" ",
                            type="spacing",
                            start=0.3,
                            end=0.4,
                            speaker_id="speaker_0",
                        ),
                        ScribeWord(
                            text="there.", start=0.4, end=0.9, speaker_id="speaker_0"
                        ),
                        ScribeWord(
                            text=" ",
                            type="spacing",
                            start=0.9,
                            end=1.2,
                            speaker_id="speaker_0",
                        ),
                        ScribeWord(
                            text="Hello.", start=1.2, end=1.6, speaker_id="speaker_1"
                        ),
                    ],
                ),
            },
            effect=BlockEffect.READ,
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: ElevenLabsCredentials,
        execution_context: ExecutionContext,
        **kwargs,
    ) -> BlockOutput:
        file_name, content = await self.read_audio(input_data.audio, execution_context)
        result = await self.transcribe(
            credentials,
            file_name,
            content,
            input_data.language,
            input_data.max_speakers,
        )
        self.merge_stats(
            NodeExecutionStats(
                provider_cost=audio_seconds(result) * USD_PER_AUDIO_SECOND,
                provider_cost_type="cost_usd",
            )
        )
        segments = group_by_speaker(result.words)
        yield "transcript", result.text
        yield "segments", segments
        yield "speakers", list(dict.fromkeys(s.speaker for s in segments))

    @staticmethod
    async def read_audio(
        audio: MediaFileType, execution_context: ExecutionContext
    ) -> tuple[str, bytes]:
        local_path = await store_media_file(
            file=audio,
            execution_context=execution_context,
            return_format="for_local_processing",
        )
        if not execution_context.graph_exec_id:
            raise ValueError("execution_context.graph_exec_id is required")
        path = Path(get_exec_file_path(execution_context.graph_exec_id, local_path))
        return path.name, path.read_bytes()

    @staticmethod
    async def transcribe(
        credentials: ElevenLabsCredentials,
        file_name: str,
        content: bytes,
        language: str,
        max_speakers: Optional[int],
    ) -> ScribeTranscript:
        form = {"model_id": SCRIBE_MODEL, "diarize": "true"}
        if language:
            form["language_code"] = language
        if max_speakers:
            form["num_speakers"] = str(max_speakers)
        content_type = mimetypes.guess_type(file_name)[0] or "application/octet-stream"
        # One attempt: a retry would re-send the BytesIO from where the first read left it.
        response = await Requests(retry_max_attempts=1).post(
            SCRIBE_URL,
            headers={"xi-api-key": credentials.api_key.get_secret_value()},
            data=form,
            files=[("file", (file_name, BytesIO(content), content_type))],
            timeout=SCRIBE_TIMEOUT,
        )
        return ScribeTranscript.model_validate(response.json())


def audio_seconds(result: ScribeTranscript) -> float:
    """The billable length: Scribe's reported duration, else where the last word ends."""
    if result.audio_duration_secs is not None:
        return result.audio_duration_secs
    return max((w.end for w in result.words if w.end is not None), default=0.0)


def group_by_speaker(words: list[ScribeWord]) -> list[SpeechSegment]:
    """Merge consecutive words of one speaker into a segment; spacing never opens one."""
    segments: list[SpeechSegment] = []
    for word in words:
        if word.type == "spacing" or word.start is None or word.end is None:
            if segments:
                segments[-1].text += word.text
            continue
        speaker = word.speaker_id or "unknown"
        if segments and segments[-1].speaker == speaker:
            segments[-1].end = word.end
            segments[-1].text += word.text
            continue
        segments.append(
            SpeechSegment(
                speaker=speaker, start=word.start, end=word.end, text=word.text
            )
        )
    return [s.model_copy(update={"text": s.text.strip()}) for s in segments]
