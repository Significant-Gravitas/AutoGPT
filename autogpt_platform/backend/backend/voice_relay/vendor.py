"""The interview's voice vendor, behind one small interface.

A call talks to a :class:`VendorSession`: it sends the person's audio, the
server's notes and tool results, and reads :data:`VendorEvent` objects back.
:class:`GeminiLive` is the production adapter; ``fake_vendor`` is the test
double. Nothing outside this module touches the vendor SDK's own types, but
``vendor_failure``, which reads its errors for why a session failed.
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from google.genai import Client, types
from pydantic import BaseModel, ConfigDict, Field

from backend.voice_relay.config import IN_RATE

logger = logging.getLogger(__name__)

CLOSE_TIMEOUT_S = 5.0


class _Event(BaseModel):
    model_config = ConfigDict(frozen=True)


class Usage(_Event):
    """A usage report: the tokens the vendor billed since its last one."""

    prompt_tokens: int = 0
    response_tokens: int = 0
    thoughts_tokens: int = 0
    prompt_by_modality: dict[str, int] = Field(default_factory=dict)
    response_by_modality: dict[str, int] = Field(default_factory=dict)


class UserText(_Event):
    """A caption fragment of what the vendor heard the person say."""

    text: str


class AgentText(_Event):
    """A caption fragment of what Otto is saying."""

    text: str


class AgentAudio(_Event):
    """Otto's speech: 24 kHz mono PCM16."""

    pcm: bytes


class Interrupted(_Event):
    """The person spoke over Otto; the rest of what Otto was saying is dropped."""


class TurnComplete(_Event):
    """Otto has finished a turn."""


class FunctionCall(_Event):
    call_id: str | None
    name: str
    args: dict[str, Any]


class ToolCall(_Event):
    """Otto called tools; each needs a result before it carries on."""

    calls: tuple[FunctionCall, ...]


class GoingAway(_Event):
    """The vendor will close the session soon."""

    time_left: str | None = None


VendorEvent = (
    Usage
    | UserText
    | AgentText
    | AgentAudio
    | Interrupted
    | TurnComplete
    | ToolCall
    | GoingAway
)


class ToolResult(_Event):
    call_id: str | None
    name: str
    response: dict[str, Any]


@runtime_checkable
class VendorSession(Protocol):
    """One live voice session, held for one call (or until it drops)."""

    async def send_audio(self, pcm: bytes) -> None: ...

    async def end_audio_stream(self) -> None:
        """Tell the vendor the person's audio has paused, closing their turn."""
        ...

    async def send_note(self, text: str, *, turn_complete: bool) -> None:
        """Send the server's own text into the conversation."""
        ...

    async def send_tool_results(self, results: Sequence[ToolResult]) -> None: ...

    def events(self) -> AsyncIterator[VendorEvent]:
        """The session's events until it drops, which raises."""
        ...

    async def close(self) -> None: ...


@runtime_checkable
class VendorConnector(Protocol):
    async def connect(self, *, opening: str) -> VendorSession:
        """Open a session and send *opening* as the conversation's first turn."""
        ...


class GeminiLive:
    """Gemini Live sessions for the voice interview."""

    def __init__(
        self,
        client: Client,
        *,
        model: str,
        voice: str,
        system_prompt: str,
        tool_name: str,
        tool_description: str,
        tool_parameters: Mapping[str, object],
    ) -> None:
        self._client = client
        self._model = model
        self._config = types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            system_instruction=system_prompt,
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                )
            ),
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection()
            ),
            tools=[
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=tool_name,
                            description=tool_description,
                            parameters_json_schema=dict(tool_parameters),
                        )
                    ]
                )
            ],
        )

    async def connect(self, *, opening: str) -> "GeminiLiveSession":
        manager = self._client.aio.live.connect(model=self._model, config=self._config)
        live = await manager.__aenter__()
        session = GeminiLiveSession(manager, live)
        try:
            await session.send_note(opening, turn_complete=True)
        except BaseException:
            await session.close()
            raise
        return session


class GeminiLiveSession:
    def __init__(self, manager: Any, live: Any) -> None:
        self._manager = manager
        self._live = live

    async def send_audio(self, pcm: bytes) -> None:
        await self._live.send_realtime_input(
            audio=types.Blob(data=pcm, mime_type=f"audio/pcm;rate={IN_RATE}")
        )

    async def end_audio_stream(self) -> None:
        await self._live.send_realtime_input(audio_stream_end=True)

    async def send_note(self, text: str, *, turn_complete: bool) -> None:
        await self._live.send_client_content(
            turns=types.Content(role="user", parts=[types.Part(text=text)]),
            turn_complete=turn_complete,
        )

    async def send_tool_results(self, results: Sequence[ToolResult]) -> None:
        await self._live.send_tool_response(
            function_responses=[
                types.FunctionResponse(
                    id=result.call_id, name=result.name, response=dict(result.response)
                )
                for result in results
            ]
        )

    async def events(self) -> AsyncIterator[VendorEvent]:
        # The SDK's receive() stops after each turn; the session goes on.
        while True:
            received = False
            async for message in self._live.receive():
                received = True
                for event in events_of(message):
                    yield event
            if not received:
                raise ConnectionError("the vendor's stream ended")

    async def close(self) -> None:
        try:
            await asyncio.wait_for(
                self._manager.__aexit__(None, None, None), timeout=CLOSE_TIMEOUT_S
            )
        except Exception:
            logger.debug("Gemini Live session did not close cleanly", exc_info=True)


def events_of(message: types.LiveServerMessage) -> list[VendorEvent]:
    """A Live message as vendor events, in the order the call handles them."""
    events: list[VendorEvent] = []
    usage = message.usage_metadata
    if usage is not None:
        events.append(
            Usage(
                prompt_tokens=usage.prompt_token_count or 0,
                response_tokens=usage.response_token_count or 0,
                thoughts_tokens=usage.thoughts_token_count or 0,
                prompt_by_modality=_by_modality(usage.prompt_tokens_details),
                response_by_modality=_by_modality(usage.response_tokens_details),
            )
        )
    if message.go_away is not None:
        time_left = message.go_away.time_left
        events.append(
            GoingAway(time_left=None if time_left is None else str(time_left))
        )
    if message.tool_call is not None and message.tool_call.function_calls:
        events.append(
            ToolCall(
                calls=tuple(
                    FunctionCall(
                        call_id=call.id,
                        name=call.name or "",
                        args=dict(call.args or {}),
                    )
                    for call in message.tool_call.function_calls
                )
            )
        )
    content = message.server_content
    if content is None:
        return events
    if content.input_transcription is not None and content.input_transcription.text:
        events.append(UserText(text=content.input_transcription.text))
    if content.output_transcription is not None and content.output_transcription.text:
        events.append(AgentText(text=content.output_transcription.text))
    if content.interrupted:
        events.append(Interrupted())
    if content.model_turn is not None:
        for part in content.model_turn.parts or []:
            if part.inline_data is not None and part.inline_data.data:
                events.append(AgentAudio(pcm=part.inline_data.data))
    if content.turn_complete:
        events.append(TurnComplete())
    return events


def _by_modality(details: list[types.ModalityTokenCount] | None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for detail in details or []:
        modality = detail.modality
        name = str(getattr(modality, "value", modality)).lower()
        counts[name] = counts.get(name, 0) + (detail.token_count or 0)
    return counts
