import ast

import pytest

from backend.util.architecture_test import (
    BACKEND_ROOT,
    _is_test_file,
    _iter_backend_py_files,
)

# ---------------------------------------------------------------------------
# Rule: Redis pub/sub is read only through ShardedSubscription
# ---------------------------------------------------------------------------
#
# Motivation: a redis-py pubsub left to itself goes deaf without a word. After
# a dropped connection it reconnects and resubscribes on its own, so what was
# published in between is lost and nobody re-reads it; a shard that stops
# answering without closing the socket is never noticed, because nothing
# reads a reply to a PING. `ShardedSubscription` confirms its SSUBSCRIBE,
# turns both into `EventStreamLostError`, and so is the only place a pubsub
# may be opened or read.

_WRAPPER = "data/sharded_pubsub.py"
_PUBSUB_CLASSES = frozenset({"PubSub", "ClusterPubSub"})
_PUBSUB_OPENERS = frozenset({"connect_sharded_pubsub", "connect_sharded_pubsub_async"})
_SUBSCRIBE_COMMANDS = frozenset({"SUBSCRIBE", "SSUBSCRIBE", "PSUBSCRIBE"})
_PUBSUB_READS = frozenset({"listen", "get_message", "get_sharded_message", "run"})


def test_redis_pubsub_is_read_only_through_sharded_subscription():
    offenders: list[str] = []
    for py in _iter_backend_py_files():
        rel = py.relative_to(BACKEND_ROOT)
        if _is_test_file(rel) or rel.as_posix() == _WRAPPER:
            continue
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        offenders += [
            f"  backend/{rel.as_posix()}:{line} {what}"
            for line, what in _raw_pubsub_uses(tree)
        ]
    assert not offenders, (
        "Read Redis pub/sub only through `ShardedSubscription` "
        f"(`backend/{_WRAPPER}`): a raw redis-py pubsub reconnects behind your "
        "back after a drop and never notices a shard that went silent, so its "
        "listener goes deaf with no error. Use "
        "`AsyncRedisEventBus.listen_with_resync` (or `listen_events` and handle "
        "`EventStreamLostError`), or `ShardedSubscription` for several channels "
        "on one connection:\n" + "\n".join(sorted(offenders))
    )


@pytest.mark.parametrize(
    "source, flagged",
    [
        ("pubsub = client.pubsub()", True),
        ("ps = get_redis().pubsub(ignore_subscribe_messages=True)", True),
        ("from redis.asyncio.client import PubSub", True),
        ("from redis.client import PubSub as P", True),
        ("from redis.cluster import ClusterPubSub", True),
        ("client = await redis.connect_sharded_pubsub_async(channel)", True),
        ("from backend.data.redis_client import connect_sharded_pubsub", True),
        ('await cluster.execute_command("SSUBSCRIBE", channel)', True),
        ('await conn.execute_command("subscribe", channel)', True),
        ("async for message in self._pubsub.listen():\n    pass", True),
        ("message = await pubsub.get_message(timeout=1.0)", True),
        ("ps = redis.asyncio.client.PubSub(pool)", True),
        ('ps = getattr(client, "pubsub")()', True),
        # Not pub/sub: the event bus API and other APIs' message reads.
        ("async for e in event_bus.listen(user_id=u, graph_id=g):\n    pass", False),
        ("await client.get_message(message_id)", False),
        ('await cluster.execute_command("SPUBLISH", channel, message)', False),
        ("stream = await ShardedSubscription.open(channel)", False),
    ],
)
def test_raw_pubsub_scan_reads_each_form(source: str, flagged: bool):
    assert bool(_raw_pubsub_uses(ast.parse(source))) is flagged


def _raw_pubsub_uses(tree: ast.Module) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            ("redis", "backend.data.redis_client")
        ):
            found += [
                (node.lineno, f"imports `{alias.name}`")
                for alias in node.names
                if alias.name in _PUBSUB_CLASSES | _PUBSUB_OPENERS
            ]
        if not isinstance(node, ast.Call):
            continue
        name = _called_name(node.func)
        if name == "pubsub" or _getattr_of(node) == "pubsub":
            found.append((node.lineno, "opens a redis-py pubsub"))
        elif name in _PUBSUB_CLASSES:
            found.append((node.lineno, f"constructs a `{name}`"))
        elif name in _PUBSUB_OPENERS:
            found.append((node.lineno, f"calls `{name}`"))
        elif name == "execute_command" and _first_string(node) in _SUBSCRIBE_COMMANDS:
            found.append((node.lineno, "sends a raw subscribe command"))
        elif name in _PUBSUB_READS and _receiver_mentions_pubsub(node.func):
            found.append((node.lineno, f"reads a pubsub with `{name}()`"))
    return found


def _called_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _getattr_of(call: ast.Call) -> str | None:
    """The attribute name a ``getattr(obj, "name")`` call reads."""
    if _called_name(call.func) == "getattr" and len(call.args) >= 2:
        name = call.args[1]
        if isinstance(name, ast.Constant) and isinstance(name.value, str):
            return name.value
    return None


def _first_string(call: ast.Call) -> str | None:
    if call.args and isinstance(call.args[0], ast.Constant):
        value = call.args[0].value
        return value.upper() if isinstance(value, str) else None
    return None


def _receiver_mentions_pubsub(func: ast.expr) -> bool:
    return isinstance(func, ast.Attribute) and "pubsub" in ast.unparse(
        func.value
    ).lower().replace("_", "")
