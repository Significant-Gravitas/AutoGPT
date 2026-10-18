"""Tool names and call ids that both sides accept.

OpenAI-style upstreams take function names and call ids of at most 64
characters from ``[a-zA-Z0-9_-]``; Anthropic takes tool-use ids from the same
alphabet. Anything already valid passes unchanged in both directions. Anything
else gets a deterministic stand-in -- a cleaned prefix plus a hash -- so the
mapping needs no state: the same name or id always maps to the same value,
in this request and the next one.
"""

import hashlib
import re
from collections.abc import Iterable
from uuid import uuid4

_VALID = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_UNSAFE = re.compile(r"[^a-zA-Z0-9_-]+")


def upstream_tool_name(name: str) -> str:
    """The name the upstream sees for the CLI tool ``name``."""
    if _VALID.fullmatch(name):
        return name
    return _stand_in(name, prefix="")


def name_map(names: Iterable[str]) -> dict[str, str]:
    """Upstream name -> CLI name, for every name this request uses."""
    return {upstream_tool_name(name): name for name in names}


def upstream_call_id(call_id: str) -> str:
    """The id the upstream sees for the CLI's tool-use id."""
    if _VALID.fullmatch(call_id):
        return call_id
    return _stand_in(call_id, prefix="call_")


def cli_tool_id(upstream_id: str | None) -> str:
    """The tool-use id the CLI sees for an upstream call id.

    An upstream that sends none gets a fresh one; the CLI echoes it back, and
    no upstream needs its own id back to match a result to its call once the
    whole history is resent.
    """
    if not upstream_id:
        return f"toolu_{uuid4().hex[:24]}"
    if _VALID.fullmatch(upstream_id):
        return upstream_id
    return _stand_in(upstream_id, prefix="toolu_")


def _stand_in(value: str, *, prefix: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    cleaned = _UNSAFE.sub("_", value).strip("_")
    room = 64 - len(prefix) - len(digest) - 1
    return f"{prefix}{cleaned[:room]}_{digest}" if cleaned else f"{prefix}{digest}"
