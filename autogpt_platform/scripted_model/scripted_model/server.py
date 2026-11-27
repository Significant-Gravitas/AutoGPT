"""The HTTP server: routes, streamed responses and the journal."""

import json
import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

from . import chat_completions, embeddings, messages, responses
from .fake_stripe.fake import FakeStripe
from .fake_stripe.fake import serves as stripe_serves
from .holds import Holds
from .journal import Journal, Request, fingerprint
from .planner import Conversation, Plan, Reply, plan
from .scripts import ScriptError
from .streaming import Event, sse

logger = logging.getLogger("scripted_model")

JOURNAL_PATH = "/__scripted__/journal"
RELEASE_PATH = "/__scripted__/release"
# A held reply nobody releases finishes after this, failing its journal check,
# so a journey that died mid-way doesn't leave a platform turn open for good.
HOLD_TIMEOUT_S = 120.0


@dataclass(frozen=True)
class Dialect:
    name: str
    conversation: Callable[[dict[str, Any]], Conversation]
    events: Callable[[Reply, str], Iterator[Event]]
    whole: Callable[[Reply, str], dict[str, Any]]
    error: Callable[[str], dict[str, Any]]


ANTHROPIC = Dialect(
    "anthropic",
    messages.conversation,
    messages.events,
    messages.message,
    messages.error,
)
OPENAI = Dialect(
    "openai",
    chat_completions.conversation,
    chat_completions.events,
    chat_completions.completion,
    chat_completions.error,
)


class ScriptedModelServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        scripts_dir: Path,
        stripe: FakeStripe | None = None,
    ) -> None:
        super().__init__(address, Handler)
        self.scripts_dir = scripts_dir
        self.journal = Journal()
        self.holds = Holds()
        self.stripe = stripe or FakeStripe.from_env()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @property
    def model_server(self) -> ScriptedModelServer:
        return cast(ScriptedModelServer, self.server)

    def do_HEAD(self) -> None:
        # The CLI checks it can reach its base URL before a session starts.
        if urlparse(self.path).path.endswith("/api/hello"):
            self._send_json(HTTPStatus.OK, {}, body=False)
        else:
            self._unexpected()

    def do_GET(self) -> None:
        url = urlparse(self.path)
        if stripe_serves(url.path):
            self._serve_stripe()
        elif url.path == "/healthz":
            self._send_json(HTTPStatus.OK, {"ok": True})
        elif url.path == JOURNAL_PATH:
            query = {key: values[-1] for key, values in parse_qs(url.query).items()}
            entries = self.model_server.journal.find(
                script=query.get("script"),
                nonce=query.get("nonce"),
                credential=query.get("credential"),
            )
            self._send_json(HTTPStatus.OK, {"entries": entries})
        elif url.path.endswith("/api/hello"):
            self._send_json(HTTPStatus.OK, {})
        else:
            self._unexpected()

    def do_POST(self) -> None:
        url = urlparse(self.path)
        path = url.path
        if stripe_serves(path):
            self._serve_stripe()
        elif path == RELEASE_PATH:
            self._read_body()
            self._release(parse_qs(url.query).get("nonce", [""])[-1])
        elif path.endswith("/messages/count_tokens"):
            body = self._read_json()
            tokens = max(1, len(json.dumps(body)) // 4) if body is not None else 1
            self._send_json(HTTPStatus.OK, {"input_tokens": tokens})
        elif path.endswith("/messages"):
            self._answer(ANTHROPIC)
        elif path.endswith("/chat/completions"):
            self._answer(OPENAI)
        elif path.endswith("/embeddings"):
            self._embedding_response()
        elif path.endswith("/responses"):
            self._structured_response()
        else:
            self._unexpected()

    def do_DELETE(self) -> None:
        if stripe_serves(urlparse(self.path).path):
            self._serve_stripe()
        else:
            self._unexpected()

    def _serve_stripe(self) -> None:
        """The fake Stripe's API, checkout pages and journey controls."""
        url = urlparse(self.path)
        body = self._read_body() if self.command in ("POST", "DELETE") else b""
        headers = {name.lower(): value for name, value in self.headers.items()}
        response = self.model_server.stripe.handle(
            self.command, url.path, url.query, body, headers
        )
        self.send_response(response.status)
        self.send_header("Content-Type", response.content_type)
        if response.location:
            self.send_header("Location", response.location)
        self.send_header("Content-Length", str(len(response.body)))
        self.end_headers()
        self.wfile.write(response.body)

    def _embedding_response(self) -> None:
        body = self._read_json()
        if not isinstance(body, dict):
            logger.error(
                "unreadable openai-embeddings request: body is not a JSON object"
            )
            self._send_json(
                HTTPStatus.BAD_REQUEST, embeddings.error("body is not a JSON object")
            )
            return
        try:
            payload = embeddings.response(body)
        except embeddings.EmbeddingError as error:
            logger.error("unreadable openai-embeddings request: %s", error)
            self._send_json(HTTPStatus.BAD_REQUEST, embeddings.error(str(error)))
            return

        model = str(body["model"])
        entry = self.model_server.journal.record_side_call(
            Request("openai-embeddings", self.path, self._credential()), model, ""
        )
        logger.info("openai-embeddings side call (%s)", model)
        self._send_json(HTTPStatus.OK, payload)
        self.model_server.journal.finish(entry, completed=True)

    def _structured_response(self) -> None:
        body = self._read_json()
        if not isinstance(body, dict):
            logger.error(
                "unreadable openai-responses request: body is not a JSON object"
            )
            self._send_json(
                HTTPStatus.BAD_REQUEST, responses.error("body is not a JSON object")
            )
            return
        try:
            payload, output_text = responses.response(body)
        except responses.ResponseError as error:
            logger.error("unreadable openai-responses request: %s", error)
            self._send_json(HTTPStatus.BAD_REQUEST, responses.error(str(error)))
            return

        model = str(body["model"])
        entry = self.model_server.journal.record_side_call(
            Request("openai-responses", self.path, self._credential()),
            model,
            output_text,
        )
        logger.info("openai-responses side call (%s)", model)
        self._send_json(HTTPStatus.OK, payload)
        self.model_server.journal.finish(entry, completed=True)

    def _answer(self, dialect: Dialect) -> None:
        body = self._read_json()
        if not isinstance(body, dict):
            self._send_json(
                HTTPStatus.BAD_REQUEST, dialect.error("body is not a JSON object")
            )
            return
        try:
            conversation = dialect.conversation(body)
        except (AttributeError, KeyError, TypeError) as error:
            # A body this dialect can't read: answer it rather than leave the
            # client waiting on a handler that died.
            logger.error("unreadable %s request: %r", dialect.name, error)
            self._send_json(
                HTTPStatus.BAD_REQUEST, dialect.error(f"malformed request: {error!r}")
            )
            return
        try:
            planned = plan(conversation, self.model_server.scripts_dir)
        except ScriptError as error:
            self._refuse(dialect, conversation, str(error))
            return
        entry = self.model_server.journal.record_plan(
            self._request(dialect), conversation, planned
        )
        _log_plan(dialect.name, conversation, planned)
        if conversation.stream:
            self._stream(dialect.events(planned.reply, conversation.model), entry)
        else:
            self._send_json(
                HTTPStatus.OK, dialect.whole(planned.reply, conversation.model)
            )
            self.model_server.journal.finish(entry, completed=True)

    def _refuse(
        self, dialect: Dialect, conversation: Conversation, reason: str
    ) -> None:
        """A request the script can't answer fails the turn with the reason,
        rather than leaving the chat waiting on a reply that never comes."""
        logger.error("refused %s request: %s", dialect.name, reason)
        self.model_server.journal.record_error(
            self._request(dialect), conversation, reason
        )
        self._send_json(HTTPStatus.BAD_REQUEST, dialect.error(reason))

    def _request(self, dialect: Dialect) -> Request:
        return Request(dialect.name, self.path, self._credential())

    def _credential(self) -> str | None:
        """The API key the request presented: Anthropic's ``x-api-key``, or a
        bearer token (OpenAI clients, the CLI's auth token)."""
        key = self.headers.get("x-api-key") or ""
        authorization = self.headers.get("Authorization") or ""
        if not key and authorization.lower().startswith("bearer "):
            key = authorization[len("bearer ") :].strip()
        return fingerprint(key) if key else None

    def _stream(self, events: Iterator[Event], entry: dict[str, Any]) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            for event in events:
                if event.hold:
                    self._hold(entry)
                if event.delay_ms:
                    time.sleep(event.delay_ms / 1000)
                self._write_chunk(sse(event))
            self._write_chunk(b"")
        except ConnectionError:
            logger.warning(
                "client went away mid-stream (journal entry %s)", entry["id"]
            )
            self.close_connection = True
            self.model_server.journal.finish(entry, completed=False)
            return
        self.model_server.journal.finish(entry, completed=True)

    def _hold(self, entry: dict[str, Any]) -> None:
        nonce = entry["nonce"]
        logger.info("holding run %s's reply until the journey releases it", nonce)
        released = self.model_server.holds.wait(nonce, HOLD_TIMEOUT_S)
        if not released:
            logger.error("run %s was never released; finishing its reply", nonce)
        detail = "released" if released else f"not released in {HOLD_TIMEOUT_S:g}s"
        self.model_server.journal.add_check(entry, "held reply", released, detail)

    def _release(self, nonce: str) -> None:
        if not nonce:
            message = "scripted-model: a release names its run: ?nonce=<nonce>"
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": {"message": message}})
            return
        self.model_server.holds.release(nonce)
        logger.info("released run %s", nonce)
        self._send_json(HTTPStatus.OK, {"released": nonce})

    def _write_chunk(self, data: bytes) -> None:
        self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def _read_json(self) -> Any:
        try:
            return json.loads(self._read_body() or b"null")
        except json.JSONDecodeError:
            return None

    def _read_body(self) -> bytes:
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            return self._read_chunked()
        return self.rfile.read(int(self.headers.get("Content-Length") or 0))

    def _read_chunked(self) -> bytes:
        parts = []
        while size := int(self.rfile.readline().split(b";")[0].strip() or b"0", 16):
            parts.append(self.rfile.read(size))
            self.rfile.readline()
        self.rfile.readline()
        return b"".join(parts)

    def _unexpected(self) -> None:
        if self.command in ("POST", "PUT", "PATCH", "DELETE"):
            # Unread, the body would be taken for the next request's start.
            self._read_body()
        logger.warning("unexpected request: %s %s", self.command, self.path)
        self.model_server.journal.record_unexpected(self.command, self.path)
        self._send_json(
            HTTPStatus.NOT_FOUND,
            {
                "error": {
                    "message": f"scripted-model: nothing serves {self.command} {self.path}"
                }
            },
        )

    def _send_json(self, status: HTTPStatus, payload: Any, body: bool = True) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if body:
            self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        logger.debug(format, *args)


def _log_plan(dialect: str, conversation: Conversation, planned: Plan) -> None:
    script = planned.marker.script if planned.marker else "-"
    if planned.kind == "side_call":
        logger.info(
            "%s side call (%s) for script %s", dialect, conversation.model, script
        )
        return
    calls = ", ".join(name for name, _ in planned.reply.tool_calls) or "none"
    logger.info(
        "%s turn %s/%s of %s (%s): tool calls: %s",
        dialect,
        (planned.turn_index or 0) + 1,
        planned.turn_count,
        script,
        conversation.model,
        calls,
    )
    for check in planned.checks:
        level = logging.INFO if check.ok else logging.ERROR
        logger.log(level, "expected %s: %s", check.what, check.detail)
