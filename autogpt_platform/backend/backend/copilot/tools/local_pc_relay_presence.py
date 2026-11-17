"""Atomic Redis presence operations for Local PC relay connections."""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from .local_pc_relay_protocol import (
    RelayPresence,
    as_text,
    owner_presence_index_key,
    presence_key,
)

logger = logging.getLogger(__name__)
PRESENCE_TTL_SECONDS = 45
MAX_DISCOVERY_PRESENCES = 64


def _presence_connection_kind(presence: RelayPresence) -> str | None:
    value = presence.hello.get("connection_kind")
    return value if isinstance(value, str) and value else None


def _presence_index_member(presence: RelayPresence) -> str:
    return json.dumps(
        [presence.session_id, presence.connection_id], separators=(",", ":")
    )


def _parse_presence_index_member(member: str) -> tuple[str, str]:
    value = json.loads(member)
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise ValueError("invalid Local PC presence index member")
    return value[0], value[1]


def _owner_index_keys(presence: RelayPresence) -> tuple[str, ...]:
    keys = [
        owner_presence_index_key(presence.user_id, None),
        owner_presence_index_key(presence.user_id, presence.client_id),
    ]
    connection_kind = _presence_connection_kind(presence)
    if connection_kind is not None:
        keys.extend(
            (
                owner_presence_index_key(presence.user_id, None, connection_kind),
                owner_presence_index_key(
                    presence.user_id, presence.client_id, connection_kind
                ),
            )
        )
    return tuple(keys)


async def _index_presence(redis: Any, presence: RelayPresence) -> None:
    member = _presence_index_member(presence)
    now = time.time()
    for key in _owner_index_keys(presence):
        pipeline = redis.pipeline(transaction=True)
        pipeline.zadd(key, {member: presence.expires_at})
        pipeline.zremrangebyscore(key, "-inf", now)
        pipeline.expire(key, PRESENCE_TTL_SECONDS)
        await pipeline.execute()


async def _remove_from_owner_indexes(redis: Any, presence: RelayPresence) -> None:
    member = _presence_index_member(presence)
    for key in _owner_index_keys(presence):
        await redis.zrem(key, member)


async def read_presence(redis: Any, session_id: str) -> RelayPresence | None:
    raw = await redis.hgetall(presence_key(session_id))
    if not raw:
        return None
    fields = {as_text(key): as_text(value) for key, value in raw.items()}
    try:
        presence = _parse_presence(fields)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        logger.warning("Ignoring malformed Local PC relay presence for %s", session_id)
        return None
    if presence.session_id != session_id or presence.expires_at <= time.time():
        return None
    return presence


async def register_presence(
    redis: Any,
    session_id: str,
    *,
    hello: dict[str, Any],
    user_id: str,
    client_id: str,
    connection_id: str | None = None,
) -> tuple[RelayPresence, RelayPresence | None]:
    previous = await read_presence(redis, session_id)
    presence = RelayPresence(
        session_id=session_id,
        connection_id=connection_id or str(uuid.uuid4()),
        user_id=user_id,
        client_id=client_id,
        hello=hello,
        expires_at=time.time() + PRESENCE_TTL_SECONDS,
    )
    key = presence_key(session_id)
    pipeline = redis.pipeline(transaction=True)
    pipeline.hset(
        key,
        mapping={
            "session_id": presence.session_id,
            "connection_id": presence.connection_id,
            "user_id": presence.user_id,
            "client_id": presence.client_id,
            "hello": json.dumps(presence.hello, separators=(",", ":")),
            "expires_at": str(presence.expires_at),
        },
    )
    pipeline.expire(key, PRESENCE_TTL_SECONDS)
    await pipeline.execute()
    await _index_presence(redis, presence)
    if previous is not None and previous.connection_id != presence.connection_id:
        await _remove_from_owner_indexes(redis, previous)
    return presence, previous


async def refresh_presence(redis: Any, presence: RelayPresence) -> bool:
    expires_at = time.time() + PRESENCE_TTL_SECONDS
    result = await redis.eval(
        """
        if redis.call('HGET', KEYS[1], 'connection_id') ~= ARGV[1] then
            return 0
        end
        redis.call('HSET', KEYS[1], 'expires_at', ARGV[2])
        redis.call('EXPIRE', KEYS[1], ARGV[3])
        return 1
        """,
        1,
        presence_key(presence.session_id),
        presence.connection_id,
        str(expires_at),
        str(PRESENCE_TTL_SECONDS),
    )
    if int(result or 0) != 1:
        await _remove_from_owner_indexes(redis, presence)
        return False
    presence.expires_at = expires_at
    await _index_presence(redis, presence)
    return True


async def clear_presence(redis: Any, presence: RelayPresence) -> None:
    await redis.eval(
        """
        if redis.call('HGET', KEYS[1], 'connection_id') == ARGV[1] then
            return redis.call('DEL', KEYS[1])
        end
        return 0
        """,
        1,
        presence_key(presence.session_id),
        presence.connection_id,
    )
    await _remove_from_owner_indexes(redis, presence)


async def owner_presences(
    redis: Any,
    user_id: str,
    client_id: str | None,
    *,
    connection_kind: str | None = None,
    limit: int | None = None,
) -> AsyncIterator[RelayPresence]:
    index_key = owner_presence_index_key(user_id, client_id, connection_kind)
    stop = -1 if limit is None else max(0, limit - 1)
    members = await redis.zrevrange(index_key, 0, stop)
    stale_members: list[str] = []
    for raw_member in members:
        member = as_text(raw_member)
        try:
            session_id, connection_id = _parse_presence_index_member(member)
        except (TypeError, ValueError, json.JSONDecodeError):
            stale_members.append(member)
            continue
        presence = await read_presence(redis, session_id)
        if (
            presence is None
            or presence.connection_id != connection_id
            or presence.user_id != user_id
            or (client_id is not None and presence.client_id != client_id)
            or (
                connection_kind is not None
                and _presence_connection_kind(presence) != connection_kind
            )
        ):
            stale_members.append(member)
            continue
        yield presence
    if stale_members:
        await redis.zrem(index_key, *stale_members)


def _parse_presence(fields: dict[str, str]) -> RelayPresence:
    return RelayPresence(
        session_id=fields["session_id"],
        connection_id=fields["connection_id"],
        user_id=fields["user_id"],
        client_id=fields["client_id"],
        hello=json.loads(fields["hello"]),
        expires_at=float(fields["expires_at"]),
    )
