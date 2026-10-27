"""Keep Google user data out of the LLM traces sent to Langfuse.

Google's Limited Use policy bars staff from reading what the Gmail, Drive,
Docs, Sheets and Calendar blocks return, so neither their output nor the
arguments AutoPilot passed them may reach a trace. Every traced payload goes
through :func:`redact_google_data`: the Langfuse SDK applies it as its
``mask`` (baseline turns and every call through the Langfuse-wrapped OpenAI
client), and LangSmith's Claude Agent SDK tracing, which exports SDK turns to
Langfuse over OTLP, applies it as the client's ``hide_inputs`` and
``hide_outputs``.

Data counts as Google's by the block that produced it, never by what it says:

* a call whose ``block_id`` (``run_block``), capability ``id``
  (``run_capability``) or block ``review_id`` (``resume_capability``,
  ``continue_run_block``) names a block on the ``google`` provider;
* a ``block_output`` response whose ``provider`` is ``google``;
* a tool output this process built from such a response, remembered by hash,
  which covers outputs too large to hand the model whole (it then gets a
  truncated preview that no longer parses);
* the late result of an approved held call (``gate.held``), which reaches the
  model as a ``<held_call_result>`` user row rather than as a tool result,
  when the output it wraps is any of the above or the row was remembered.

Within a conversation, the result answering a Google call is redacted whatever
it says, and so is the input of a call whose result was Google data, so the
history a later turn sends carries neither. Tool names, call ids, block ids,
timings and token counts stay readable.

The approval gate's model calls (the action supervisor and the content judge)
quote a call's arguments or result inside a prompt of their own, where none of
the above can find it, so a Google call's gate runs inside
:func:`tracing_google_data`, which redacts every text those calls trace.
"""

import contextlib
import functools
import hashlib
import json
import re
import threading
from collections import OrderedDict
from collections.abc import Iterator
from contextvars import ContextVar
from typing import Any

import langsmith
from langfuse import Langfuse
from pydantic import BaseModel

from backend.blocks import get_blocks
from backend.copilot.capabilities.block_meta import get_block_provider
from backend.copilot.constants import COPILOT_NODE_PREFIX, parse_node_id_from_exec_id
from backend.integrations.providers import ProviderName
from backend.util.settings import Settings

REDACTED = "[redacted: Google data]"
_IMAGE_REDACTED = "[redacted: image data]"

# Set on a ``tool-outputs/`` workspace file that saved a Google block's output,
# so reading it back is recognised as Google data too.
GOOGLE_DATA_METADATA_KEY = "google_data"

# The arguments that carry what a call sends to the block; ``resume_capability``
# re-runs it with ``input_overrides`` merged in.
_PAYLOAD_KEYS = ("input_data", "input", "input_overrides")
_REMEMBERED_LIMIT = 4096

# How ``gate.held`` wraps a held call's late result.
_HELD_RESULT_OPEN = re.compile(r"<held_call_result\b[^>]*>\n")
_HELD_RESULT_CLOSE = "\n</held_call_result>"
# Each pair hashes a slice of up to the whole text; a turn opens with a few
# held results at most.
_HELD_RESULT_PAIR_LIMIT = 100

_remembered: OrderedDict[str, None] = OrderedDict()
_remembered_lock = threading.Lock()

_redacting_every_text: ContextVar[bool] = ContextVar(
    "google_data_redacting_every_text", default=False
)


def redact_google_data(data: Any) -> Any:
    """Return *data* with Google block calls and results replaced by
    :data:`REDACTED` and image payloads omitted. Everything else is returned
    unchanged, unless :func:`tracing_google_data` is active, when every text
    is replaced."""
    if _redacting_every_text.get():
        return _redact_every_text(data)
    return _redact(data, _google_call_ids(data))


@contextlib.contextmanager
def tracing_google_data(active: bool = True) -> Iterator[None]:
    """While active, redact every text traced in this context, for model
    calls handed a Google call's arguments or result as plain text."""
    token = _redacting_every_text.set(_redacting_every_text.get() or active)
    try:
        yield
    finally:
        _redacting_every_text.reset(token)


def mask_langfuse_data(*, data: Any, **kwargs: Any) -> Any:
    """The Langfuse SDK's ``mask`` hook."""
    return redact_google_data(data)


def install_trace_redaction() -> None:
    """Apply the redaction to both tracers.

    Langfuse first: its client keeps the settings of the first one built in
    the process, and building it sets the OpenTelemetry tracer provider the
    LangSmith client then exports through, as it would have lazily.
    """
    install_langfuse_mask()
    langsmith.configure(client=langsmith.Client(hide_inputs=_hide, hide_outputs=_hide))


def install_langfuse_mask() -> None:
    """Build the process's Langfuse client with the mask.

    Has to run before anything calls ``langfuse.get_client()``, which hands
    out copies of the first client built: the SDK keeps that one's settings
    and drops a mask passed later. Raises when the client it gets back does
    not mask, so the caller can leave tracing off rather than trace unmasked.
    """
    # The configured credentials, which ``.env`` puts in ``Settings`` but not
    # in the environment the SDK would otherwise read them from. ``host``
    # still yields to ``LANGFUSE_BASE_URL``, as it would by default.
    secrets = Settings().secrets
    client = Langfuse(
        public_key=secrets.langfuse_public_key or None,
        secret_key=secrets.langfuse_secret_key or None,
        host=secrets.langfuse_host or None,
        mask=mask_langfuse_data,
    )
    # ``_resources`` is None when the client is disabled (no keys): it sends
    # nothing, so there is nothing to mask.
    resources = client._resources
    if resources is not None and resources.mask is not mask_langfuse_data:
        raise RuntimeError(
            "A Langfuse client was built before the Google data mask; "
            "its traces would not be redacted"
        )


def remember_google_output(output: str) -> None:
    """Record a tool output built from a Google block's result, so it is
    redacted wherever it is traced even when it no longer parses."""
    with _remembered_lock:
        _remembered[_digest(output)] = None
        while len(_remembered) > _REMEMBERED_LIMIT:
            _remembered.popitem(last=False)


def remember_cut_google_output(whole: str, cut: str) -> None:
    """Remember *cut*, a later cut of the tool output *whole*, when *whole*
    was Google data, so the text the model finally reads is redacted too."""
    if cut != whole and _is_google_text(whole):
        remember_google_output(cut)


def is_google_output(text: str) -> bool:
    """Whether the tool output *text* is Google data: a Google
    ``block_output``, or an output remembered as built from one."""
    digest = _digest(text)
    with _remembered_lock:
        if digest in _remembered:
            _remembered.move_to_end(digest)
            return True
    if not text.startswith("{"):
        return False
    try:
        parsed = json.loads(text)
    except ValueError:
        return False
    return (
        isinstance(parsed, dict)
        and parsed.get("type") == "block_output"
        and parsed.get("provider") == ProviderName.GOOGLE.value
    )


def is_google_call(arguments: dict[str, Any] | None) -> bool:
    """Whether a tool call's arguments name a block on the ``google``
    provider, as ``run_block``'s ``block_id``, ``run_capability``'s ``id``,
    or the ``review_id`` ``resume_capability`` and ``continue_run_block``
    take, which embeds the block id."""
    if not arguments:
        return False
    return (
        is_google_block_ref(arguments.get("block_id"))
        or is_google_block_ref(arguments.get("id"))
        or is_google_block_ref(_reviewed_block_id(arguments.get("review_id")))
    )


def is_google_block_ref(ref: object) -> bool:
    """Whether *ref* names a block on the ``google`` provider, by id,
    ``block:<id>`` capability id or class name."""
    return isinstance(ref, str) and ref.strip().lower() in _google_block_refs()


def _reviewed_block_id(review_id: object) -> str | None:
    """The block a block review's id names: ``copilot-node-{block_id}:{hex}``."""
    if not isinstance(review_id, str) or not review_id.startswith(COPILOT_NODE_PREFIX):
        return None
    return parse_node_id_from_exec_id(review_id).removeprefix(COPILOT_NODE_PREFIX)


def _hide(data: dict) -> dict:
    return redact_google_data(data)


def _google_call_ids(data: Any) -> set[str]:
    """Ids of the tool calls in *data* that ran a Google block: by the block
    the call names, or by the Google data its result carries."""
    ids: set[str] = set()
    for node in _dicts(data):
        call_id = node.get("id")
        if isinstance(call_id, str) and is_google_call(_call_arguments(node)):
            ids.add(call_id)
        answered = _answered_call_id(node)
        if answered is not None and _holds_google_output(node.get("content")):
            ids.add(answered)
    return ids


def _redact(node: Any, google_ids: set[str]) -> Any:
    node = _plain(node)
    if isinstance(node, str):
        return REDACTED if _is_google_text(node) else node
    if isinstance(node, list):
        return [_redact(item, google_ids) for item in node]
    if not isinstance(node, dict):
        return node
    node = _without_image_data(node)
    redacted = {key: _redact(value, google_ids) for key, value in node.items()}
    if _answered_call_id(node) in google_ids and "content" in node:
        redacted["content"] = REDACTED
    call_id = node.get("id")
    if isinstance(call_id, str) and call_id in google_ids:
        _redact_call_arguments(redacted)
    if is_google_call(node):
        redacted = _without_payload(redacted)
    return redacted


def _without_image_data(node: dict[str, Any]) -> dict[str, Any]:
    """Wire image blocks lose provenance on replay, so never trace their pixels."""
    if node.get("type") == "image":
        if "data" in node:
            node = {**node, "data": _IMAGE_REDACTED}
        source = node.get("source")
        if isinstance(source, dict) and source.get("type") == "base64":
            return {**node, "source": {**source, "data": _IMAGE_REDACTED}}
    if node.get("type") == "image_url":
        image_url = node.get("image_url")
        if isinstance(image_url, dict):
            return {**node, "image_url": {**image_url, "url": _IMAGE_REDACTED}}
    if node.get("type") == "input_image" and "image_url" in node:
        return {**node, "image_url": _IMAGE_REDACTED}
    return node


def _redact_every_text(node: Any) -> Any:
    node = _plain(node)
    if isinstance(node, str):
        return REDACTED
    if isinstance(node, list):
        return [_redact_every_text(item) for item in node]
    if isinstance(node, dict):
        return {key: _redact_every_text(value) for key, value in node.items()}
    return node


def _redact_call_arguments(call: dict[str, Any]) -> None:
    """Redact the payload of a call already known to be Google's, in either
    the Anthropic (``input``) or the OpenAI (``function.arguments``) shape."""
    arguments = call.get("input")
    if isinstance(arguments, dict):
        call["input"] = _without_payload(arguments)
        return
    function = call.get("function")
    if not isinstance(function, dict):
        return
    parsed = _call_arguments(call)
    if parsed is not None:
        call["function"] = {
            **function,
            "arguments": json.dumps(_without_payload(parsed)),
        }


def _without_payload(arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        key: REDACTED if key in _PAYLOAD_KEYS else value
        for key, value in arguments.items()
    }


def _call_arguments(node: dict[str, Any]) -> dict[str, Any] | None:
    """The arguments of a tool call node, or None when *node* is not one."""
    if node.get("type") == "tool_use":
        arguments = node.get("input")
        return arguments if isinstance(arguments, dict) else None
    function = node.get("function")
    if not isinstance(function, dict):
        return None
    raw = function.get("arguments")
    if not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _answered_call_id(node: dict[str, Any]) -> str | None:
    """The call a tool result answers: Anthropic ``tool_use_id`` or OpenAI
    (and LangSmith history) ``tool_call_id``."""
    answered = node.get("tool_use_id") or node.get("tool_call_id")
    return answered if isinstance(answered, str) else None


def _holds_google_output(content: Any) -> bool:
    if isinstance(content, str):
        return _is_google_text(content)
    if isinstance(content, list):
        return any(_holds_google_output(item) for item in content)
    if isinstance(content, dict):
        return any(_holds_google_output(value) for value in content.values())
    return False


def _is_google_text(text: str) -> bool:
    return is_google_output(text) or _wraps_google_held_result(text)


def _wraps_google_held_result(text: str) -> bool:
    """Whether *text* carries a held call's late result that is Google data.

    The row may be folded into a longer message (the SDK's turn opener), and
    the output it wraps may itself quote the closing tag, so every closing
    tag after an opening one is tried. A text with more pairs than that
    could take is redacted rather than scanned.
    """
    if "<held_call_result" not in text:
        return False
    closes = [
        match.start() for match in re.finditer(re.escape(_HELD_RESULT_CLOSE), text)
    ]
    pairs = 0
    for opening in _HELD_RESULT_OPEN.finditer(text):
        for close in closes:
            if close < opening.end():
                continue
            pairs += 1
            if pairs > _HELD_RESULT_PAIR_LIMIT:
                return True
            row = text[opening.start() : close + len(_HELD_RESULT_CLOSE)]
            inner = text[opening.end() : close]
            if is_google_output(inner) or is_google_output(row):
                return True
    return False


def _dicts(node: Any) -> Iterator[dict[str, Any]]:
    node = _plain(node)
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _dicts(value)
    elif isinstance(node, list):
        for item in node:
            yield from _dicts(item)


def _plain(node: Any) -> Any:
    """SDK message objects (the OpenAI client's tool calls) as plain data."""
    return node.model_dump(mode="json") if isinstance(node, BaseModel) else node


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


@functools.cache
def _google_block_refs() -> frozenset[str]:
    refs: set[str] = set()
    for block_id, block_class in get_blocks().items():
        if get_block_provider(block_class()) != ProviderName.GOOGLE.value:
            continue
        refs.update(
            (
                block_id.lower(),
                f"block:{block_id}".lower(),
                block_class.__name__.lower(),
            )
        )
    return frozenset(refs)
