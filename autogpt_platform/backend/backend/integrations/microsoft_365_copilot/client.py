import asyncio
import codecs
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

GRAPH_BASE_URL = "https://graph.microsoft.com/beta"


class Microsoft365CopilotError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None):
        super().__init__(message)
        self.status = status


class Microsoft365CopilotDeclined(Microsoft365CopilotError):
    """Copilot refused the request under its Responsible AI policy."""

    def __init__(self) -> None:
        super().__init__(
            "Microsoft 365 Copilot declined this request under its "
            "Responsible AI policy; rephrase and try again"
        )


def build_chat_request(
    message: str,
    *,
    timezone: str,
    additional_context: list[str] | None = None,
    web_enabled: bool | None = None,
    file_uris: list[str] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "message": {"text": message},
        "locationHint": {"timeZone": timezone},
    }
    if additional_context:
        body["additionalContext"] = [
            {"text": context} for context in additional_context if context
        ]

    contextual_resources: dict[str, Any] = {}
    if web_enabled is not None:
        contextual_resources["webContext"] = {"isWebEnabled": web_enabled}
    if file_uris:
        contextual_resources["files"] = [{"uri": uri} for uri in file_uris]
    if contextual_resources:
        body["contextualResources"] = contextual_resources
    return body


async def _iter_sse_json(chunks: AsyncIterator[bytes]) -> AsyncIterator[dict[str, Any]]:
    buffer = ""
    data_lines: list[str] = []
    decoder = codecs.getincrementaldecoder("utf-8")()

    def parse_event() -> dict[str, Any] | None:
        payload = "\n".join(data_lines)
        if not payload or payload == "[DONE]":
            return None
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as error:
            raise Microsoft365CopilotError(
                "Microsoft 365 Copilot returned an invalid stream response"
            ) from error
        return parsed if isinstance(parsed, dict) else None

    async for chunk in chunks:
        buffer += decoder.decode(chunk)
        while "\n" in buffer:
            raw_line, buffer = buffer.split("\n", 1)
            line = raw_line.removesuffix("\r")
            if not line:
                if data_lines:
                    event = parse_event()
                    data_lines = []
                    if event is not None:
                        yield event
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())

    buffer += decoder.decode(b"", final=True)
    trailing_line = buffer.removesuffix("\r")
    if trailing_line.startswith("data:"):
        data_lines.append(trailing_line[5:].lstrip())
    if data_lines:
        event = parse_event()
        if event is not None:
            yield event


async def iter_copilot_text_deltas(
    chunks: AsyncIterator[bytes],
) -> AsyncIterator[str]:
    response_message_id: str | None = None
    emitted_text = ""

    async for conversation in _iter_sse_json(chunks):
        # Responsible AI refusals arrive as ordinary 200 snapshots with the
        # conversation flagged; without this they would end as an empty or
        # partial "successful" answer.
        if conversation.get("state") == "disengagedForRai":
            raise Microsoft365CopilotDeclined()
        messages = conversation.get("messages")
        if not isinstance(messages, list) or not messages:
            continue

        selected: dict[str, Any] | None = None
        if response_message_id:
            selected = next(
                (
                    message
                    for message in messages
                    if isinstance(message, dict)
                    and message.get("id") == response_message_id
                ),
                None,
            )
        if selected is None and response_message_id is None:
            candidate = messages[-1]
            if isinstance(candidate, dict):
                selected = candidate
                candidate_id = candidate.get("id")
                if isinstance(candidate_id, str):
                    response_message_id = candidate_id
        if selected is None:
            continue

        text = selected.get("text")
        if not isinstance(text, str):
            continue
        if not text.startswith(emitted_text):
            # Graph occasionally rewrites already-streamed prose (citation
            # markers, entity tags). A streamed delta cannot un-emit text, so
            # resync to the new snapshot and keep streaming from there
            # instead of dropping every later delta.
            logger.warning(
                "Microsoft 365 Copilot rewrote streamed text; resyncing "
                "(%d emitted, %d in snapshot)",
                len(emitted_text),
                len(text),
            )
            emitted_text = text
            continue
        delta = text[len(emitted_text) :]
        if delta:
            emitted_text = text
            yield delta


class Microsoft365CopilotClient:
    def __init__(
        self,
        access_token: str,
        *,
        base_url: str = GRAPH_BASE_URL,
        timeout_seconds: float = 120.0,
    ):
        self._access_token = access_token
        self._base_url = base_url.rstrip("/")
        # A grounded Copilot answer may legitimately stream for several minutes.
        # Limit connection setup and periods with no bytes, not the total lifetime
        # of the SSE response.
        self._timeout = aiohttp.ClientTimeout(
            total=None,
            connect=30.0,
            sock_connect=30.0,
            sock_read=timeout_seconds,
        )
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "Microsoft365CopilotClient":
        self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self

    async def __aexit__(self, *_args: object) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._access_token}",
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }

    def _client(self) -> aiohttp.ClientSession:
        if self._session is None:
            raise RuntimeError(
                "Microsoft365CopilotClient must be used as a context manager"
            )
        return self._session

    async def create_conversation(self) -> str:
        url = f"{self._base_url}/copilot/conversations"
        try:
            async with self._client().post(
                url, headers=self._headers, json={}
            ) as response:
                await self._raise_for_error(response, "create a Copilot conversation")
                payload = await response.json()
        except Microsoft365CopilotError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as error:
            raise Microsoft365CopilotError(
                "Microsoft 365 Copilot could not create a Copilot conversation"
            ) from error
        conversation_id = payload.get("id") if isinstance(payload, dict) else None
        if not isinstance(conversation_id, str) or not conversation_id:
            raise Microsoft365CopilotError(
                "Microsoft 365 Copilot returned a conversation without an ID"
            )
        return conversation_id

    async def stream_chat(
        self,
        conversation_id: str,
        message: str,
        *,
        timezone: str,
        additional_context: list[str] | None = None,
        web_enabled: bool | None = None,
        file_uris: list[str] | None = None,
    ) -> AsyncIterator[str]:
        url = f"{self._base_url}/copilot/conversations/{conversation_id}/chatOverStream"
        request = build_chat_request(
            message,
            timezone=timezone,
            additional_context=additional_context,
            web_enabled=web_enabled,
            file_uris=file_uris,
        )
        try:
            async with self._client().post(
                url, headers=self._headers, json=request
            ) as response:
                await self._raise_for_error(response, "continue a Copilot conversation")
                async for delta in iter_copilot_text_deltas(
                    response.content.iter_any()
                ):
                    yield delta
        except Microsoft365CopilotError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as error:
            raise Microsoft365CopilotError(
                "Microsoft 365 Copilot could not continue a Copilot conversation"
            ) from error

    async def _raise_for_error(
        self, response: aiohttp.ClientResponse, action: str
    ) -> None:
        if 200 <= response.status < 300:
            return
        error_code = None
        try:
            payload = await response.json()
            error = payload.get("error") if isinstance(payload, dict) else None
            if isinstance(error, dict) and isinstance(error.get("code"), str):
                error_code = error["code"]
        except (aiohttp.ContentTypeError, json.JSONDecodeError):
            pass
        suffix = f" ({error_code})" if error_code else ""
        raise Microsoft365CopilotError(
            f"Microsoft 365 Copilot could not {action}{suffix}",
            status=response.status,
        )
