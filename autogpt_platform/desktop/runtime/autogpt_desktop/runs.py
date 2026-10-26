"""Agent runs and AutoPilot turns: what is in flight, and what becomes of it
when a process dies or the app closes.

An executor takes a lock in Valkey for each run it works on, valid for five
minutes, renewed while the run goes on, and written to disk. `Cache.active`
combines those locks with AutoPilot's running session metadata; the
supervisor reports the count to the shell, which asks before it quits with
work in flight.

A lock that runs out is never taken again: its run goes on without one. A
computer that sleeps for longer than the five minutes does that to every
run it sleeps through. So an agent run whose lock has gone is asked about
in the database, and counts for as long as it is RUNNING there and the
process that was working on it lives (`Watch`). A stop of the runs ends
every run the database has as RUNNING (`running`): besides those, the ones
nobody is working on at that moment, which would otherwise start again.
AutoPilot remains active until its metadata records completion. If metadata
also expires, previously observed work remains active until its host exits.

A new executor that finds the lock of a dead one takes the run for someone
else's, drops its message, and the run stays "running" for ever. On a desktop
there is one executor, so a lock that exists before it starts is always a
dead one's: `Cache.clear_stale_locks` removes them before every start of the
process that hosts an executor, at boot and when the supervisor restarts it.
The run's message is still in the queue, and the platform then picks the run
up where it left off. An interrupted run resumes.

A run the user chose to stop as the app closed does not: its executor is
asked to cancel it (`Queue.cancel`), and whatever has not ended when the
services are down is ended in the database (`end_stopped`), where the
platform skips the message of a run that has ended. The choice is on disk
from the moment it is made (`record_stopped`), so a runtime that dies before
it got that far leaves it for the next start; a runtime that lives applies
what it was told, whether or not the disk took it.

An AutoPilot turn has no such status. One that did not end when asked has
its message taken off the queue (`Queue.drop_turns`), and its chat, which
the platform still takes for busy, is handed to the next start to close:
ENDED_TURNS_FILE, which the process hosting the copilot executor reads
(servicehost.py), because closing a chat is upstream's own code's to do.
A chat is listed with its original turn identity only once its message is
gone. That identity guards retries against closing a later turn. A missing
lock is not proof that the broker acknowledged the message.

`reconcile` runs once per start, before any service does: it applies such a
record, and fails runs that have been "running" for more than a day rather
than resume them out of the blue. `sweep_orphans` runs once per start, a
minute after the app is up: a run from before this start that nobody holds
and no message stands for (the broker's data was lost, or the message used
up its deliveries) would otherwise stay "running" for ever.

Cleanup is bounded and failures retain durable cancellation intent. Startup
refuses to launch executors while a saved stop cannot be reconciled. Every
name below is upstream's (tests/test_upstream_names.py and
tests/backend_contract.py check them).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable, Iterable
from contextlib import closing, suppress
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("autogpt_desktop")

# Service name (apps.SERVICES) -> the lock keys its executor takes:
# backend/executor/manager.py and copilot/executor/utils.get_session_lock_key.
EXECUTOR = "executor"
LOCK_PATTERNS = {
    EXECUTOR: "exec_lock:*",
    "copilot-executor": "copilot:session:*:lock",
}

GRAPH_EXECUTION_TABLE = "AgentGraphExecution"
GRAPH_EXECUTION_COLUMNS = (
    "id",
    "executionStatus",
    "createdAt",
    "updatedAt",
    "isDeleted",
)
NODE_EXECUTION_TABLE = "AgentNodeExecution"
NODE_EXECUTION_COLUMNS = ("executionStatus", "endedTime", "agentGraphExecutionId")
EXECUTION_STATUSES = ("QUEUED", "RUNNING", "TERMINATED", "FAILED")

# backend/executor/utils.py and backend/copilot/executor/utils.py: the queue
# an executor takes its work from, the exchange a request to cancel is
# published to (to every executor: the routing key is empty), and the one
# field of that request, which is also where a turn's own message names its
# session.
GRAPH_RUN_QUEUE = "graph_execution_queue_v2"
GRAPH_CANCEL_EXCHANGE = "graph_execution_cancel"
GRAPH_CANCEL_FIELD = "graph_exec_id"
COPILOT_TURN_QUEUE = "copilot_execution_queue_v2"
COPILOT_CANCEL_EXCHANGE = "copilot_cancel"
COPILOT_SESSION_FIELD = "session_id"
SESSION_META_PREFIX = "chat:task:meta:"

# What the user chose to stop as the app closed: {"graph_runs": [ids],
# "copilot_turns": [session ids], "copilot_turn_ids": {session: turn}}.
# There from the choice until all of it has ended.
STOPPED_RUNS_FILE = "stopped-runs.json"
# Beside it: a list of session_id/turn_id records for the next copilot
# executor's host to close. Legacy session-only entries require recovery.
ENDED_TURNS_FILE = "ended-turns.json"
RESUME_WINDOW_HOURS = 24

# After the app is up, and then until it has been done once. Not at boot: by
# now every message that survived has been delivered and its run locked.
ORPHAN_SWEEP_AFTER_SECONDS = 60
ORPHAN_SWEEP_EVERY_SECONDS = 60
ORPHAN_SWEEP_TRIES = 30

# How often a run that goes on without its lock is asked about again.
UNLOCKED_RUNS_EVERY_SECONDS = 10
# The broker hands the message of a killed executor back a moment after the
# executor is gone, and may be a moment late answering at all.
DROP_TRIES = 5
DROP_RETRY_SECONDS = 0.5

UNFINISHED_AND_OLD = f"""\
SELECT id FROM platform."{GRAPH_EXECUTION_TABLE}"
WHERE "executionStatus" IN ('RUNNING', 'QUEUED') AND NOT "isDeleted"
AND "createdAt" < (now() AT TIME ZONE 'utc') - interval '{RESUME_WINDOW_HOURS} hours'
"""
# "createdAt" is UTC without a zone, and so is what it is compared with.
UNFINISHED_BEFORE = f"""\
SELECT id FROM platform."{GRAPH_EXECUTION_TABLE}"
WHERE "executionStatus" IN ('RUNNING', 'QUEUED') AND NOT "isDeleted"
AND "createdAt" < %(before)s
"""
# Only an executor marks a run RUNNING: these are, or were, being worked on.
RUNNING = f"""\
SELECT id FROM platform."{GRAPH_EXECUTION_TABLE}"
WHERE "executionStatus" = 'RUNNING' AND NOT "isDeleted"
"""
RUNNING_AMONG = f"{RUNNING}AND id = ANY(%(ids)s)\n"
# The platform's own shape for ending a run from outside the executor
# (backend/data/diagnostics.py, and the node cascade in data/execution.py).
END_NODES = f"""\
UPDATE platform."{NODE_EXECUTION_TABLE}"
SET "executionStatus" = 'FAILED', "endedTime" = now() AT TIME ZONE 'utc'
WHERE "executionStatus" = 'RUNNING' AND "agentGraphExecutionId" = ANY(%(ids)s)
"""
END_RUNS = f"""\
UPDATE platform."{GRAPH_EXECUTION_TABLE}"
SET "executionStatus" = %(status)s, "updatedAt" = now() AT TIME ZONE 'utc'
WHERE id = ANY(%(ids)s) AND "executionStatus" IN ('RUNNING', 'QUEUED') AND NOT "isDeleted"
"""


@dataclass(frozen=True)
class Activity:
    """What is in flight: the ids of agent runs, and the session ids of
    AutoPilot turns (to their user: AutoPilot replying)."""

    graph_runs: tuple[str, ...] = ()
    copilot_turns: tuple[str, ...] = ()
    copilot_turn_ids: tuple[tuple[str, str], ...] = ()

    def __bool__(self) -> bool:
        return bool(self.graph_runs or self.copilot_turns)

    @property
    def counts(self) -> tuple[int, int]:
        return len(self.graph_runs), len(self.copilot_turns)

    def describe(self) -> str:
        runs, turns = self.counts
        parts = [
            f"{runs} agent run{'' if runs == 1 else 's'}" if runs else "",
            (
                f"{turns} AutoPilot {'reply' if turns == 1 else 'replies'}"
                if turns
                else ""
            ),
        ]
        return " and ".join(part for part in parts if part) or "nothing"

    def still(self, now: Activity) -> Activity:
        """What of this is in `now` as well."""
        return Activity(
            tuple(run for run in self.graph_runs if run in now.graph_runs),
            tuple(turn for turn in self.copilot_turns if turn in now.copilot_turns),
            tuple(
                pair for pair in self.copilot_turn_ids if pair[0] in now.copilot_turns
            ),
        )

    def merged(self, other: Activity) -> Activity:
        return Activity(
            tuple(dict.fromkeys([*self.graph_runs, *other.graph_runs])),
            tuple(dict.fromkeys([*self.copilot_turns, *other.copilot_turns])),
            tuple((dict(self.copilot_turn_ids) | dict(other.copilot_turn_ids)).items()),
        )


@dataclass(frozen=True)
class Cache:
    """The Valkey node, as the runtime reaches it."""

    port: int
    password: str
    # Set once the node answers, when it is given. While the app starts, the
    # hosts are started before the node is up; there is nothing to connect to
    # yet, and the supervisor clears the locks itself before any service
    # starts (supervisor.Stack.give_the_go).
    up: threading.Event | None = None
    session_meta_prefix: str = SESSION_META_PREFIX
    # One connection for the whole run of the app: what is running is asked
    # every two seconds.
    _kept: list = field(default_factory=list, compare=False, repr=False)
    _known_turns: dict[str, str] = field(
        default_factory=dict, compare=False, repr=False
    )

    def clear_stale_locks(self, services: Iterable[str]) -> None:
        """Before the process hosting `services` starts: any lock of theirs
        that exists belongs to a process that is gone."""
        patterns = [LOCK_PATTERNS[name] for name in services if name in LOCK_PATTERNS]
        if LOCK_PATTERNS["copilot-executor"] in patterns:
            self._known_turns.clear()
        if not patterns or (self.up is not None and not self.up.is_set()):
            return
        try:
            removed = self._delete_matching(patterns)
        except Exception as exc:  # a lock left behind costs one run, not the app
            logger.warning(f"could not clear stale run locks: {exc}")
            return
        if removed:
            logger.info(f"cleared {removed} lock(s) left by a process that is gone")

    def active(self) -> Activity | None:
        """Work held by a lock or recorded as running by AutoPilot.
        None when the node did not say."""
        try:
            graphs = self._locked(LOCK_PATTERNS["executor"])
            locked = self._locked(LOCK_PATTERNS["copilot-executor"])
            metadata = self.turn_metadata()
            sessions = tuple(
                sorted(
                    set(locked)
                    | {
                        session
                        for session, meta in metadata.items()
                        if meta.get("status") == "running"
                    }
                    | {
                        session
                        for session in self._known_turns
                        if session not in metadata
                    }
                )
            )
            identities = tuple(
                (
                    session,
                    metadata.get(session, {}).get("turn_id")
                    or self._known_turns.get(session, ""),
                )
                for session in sessions
                if metadata.get(session, {}).get("turn_id")
                or self._known_turns.get(session)
            )
            self._known_turns.clear()
            self._known_turns.update(
                (session, dict(identities).get(session, "")) for session in sessions
            )
            return Activity(graphs, sessions, identities)
        except Exception as exc:
            logger.debug("could not ask the cache what is running: %s", exc)
            return None

    def turn_metadata(self) -> dict[str, dict[str, str]]:
        client = self._client()
        result = {}
        for key in client.scan_iter(match=f"{self.session_meta_prefix}*", count=500):
            meta = {
                _text(name): _text(value) for name, value in client.hgetall(key).items()
            }
            session = (
                _text(key)[len(self.session_meta_prefix) :]
                .removeprefix("{")
                .removesuffix("}")
            )
            result[session] = meta
        return result

    def release(self, activity: Activity) -> None:
        """Once their executors are gone: the locks of runs that were stopped."""
        keys = [
            *(lock_key("executor", run) for run in activity.graph_runs),
            *(lock_key("copilot-executor", turn) for turn in activity.copilot_turns),
        ]
        try:
            for key in keys:
                self._delete(key)
        except Exception as exc:  # the next start clears them all anyway
            logger.warning(f"could not remove the locks of stopped runs: {exc}")

    def _locked(self, pattern: str) -> tuple[str, ...]:
        prefix, _, suffix = pattern.partition("*")
        keys = sorted(_text(key) for key in self._matching([pattern]))
        return tuple(key[len(prefix) : len(key) - len(suffix)] for key in keys)

    def _delete_matching(self, patterns: list[str]) -> int:
        return sum(self._delete(key) for key in self._matching(patterns))

    def _matching(self, patterns: list[str]) -> list:
        client = self._client()
        return [
            key
            for pattern in patterns
            for key in client.scan_iter(match=pattern, count=500)
        ]

    def _delete(self, key) -> int:
        # One key per command: the node is in cluster mode, where a command
        # naming keys from different slots is refused.
        return int(self._client().delete(key))

    def _client(self):
        if self._kept:
            return self._kept[0]
        import redis
        from redis.backoff import NoBackoff
        from redis.retry import Retry

        # No retries: this runs in the supervisor's own loop, and against a
        # node that is down each refused connection costs seconds on Windows.
        client = redis.Redis(
            host="127.0.0.1",
            port=self.port,
            password=self.password,
            socket_timeout=5,
            socket_connect_timeout=3,
            retry=Retry(NoBackoff(), 0),
        )
        self._kept.append(client)
        return client

    def close(self) -> None:
        while self._kept:
            with suppress(Exception):
                self._kept.pop().close()


def lock_key(service: str, held: str) -> str:
    return LOCK_PATTERNS[service].replace("*", held)


def _text(value: object) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


@dataclass(frozen=True)
class Queue:
    """The broker, as the runtime reaches it."""

    port: int
    user: str
    password: str

    def cancel(self, activity: Activity) -> None:
        """Ask the executors to stop what they are working on, the way the
        platform's own Stop does."""
        requests = (
            (GRAPH_CANCEL_EXCHANGE, GRAPH_CANCEL_FIELD, activity.graph_runs),
            (COPILOT_CANCEL_EXCHANGE, COPILOT_SESSION_FIELD, activity.copilot_turns),
        )
        try:
            with closing(self._connect()) as connection:
                for exchange, name, ids in requests:
                    if ids:
                        self._publish(
                            connection,
                            exchange,
                            [json.dumps({name: held}) for held in ids],
                        )
        except Exception as exc:  # they are ended in the database all the same
            logger.warning(f"could not ask the executors to cancel: {exc}")

    def _publish(self, connection, exchange: str, bodies: list[str]) -> None:
        """Without declaring the exchange: the one for agent runs deletes
        itself when no executor listens, and declaring it any other way than
        the backend does is refused. Published to an exchange that is not
        there, a request closes its channel, which is why each exchange has a
        channel of its own; nobody was listening, so nothing is lost."""
        import pika.exceptions

        try:
            channel = connection.channel()
            for body in bodies:
                channel.basic_publish(
                    exchange=exchange, routing_key="", body=body.encode()
                )
            channel.close()  # where the broker's refusal, if any, is heard
        except pika.exceptions.AMQPChannelError as exc:
            logger.info(
                f"no executor listens on {exchange} ({exc}); nothing to cancel there"
            )

    def drop_turns(
        self, sessions: Iterable[str], turn_ids: dict[str, str] | None = None
    ) -> set[str] | None:
        """Take the messages of these sessions' AutoPilot turns off their
        queue, so that turns the user stopped are not run again. Only while
        nothing consumes the queue. The sessions a message was found for;
        None when the broker could not be asked."""
        stopped = set(sessions)
        if not stopped:
            return set()
        import pika.exceptions

        try:
            with closing(self._connect()) as connection:
                found = _drop(
                    connection.channel(),
                    COPILOT_TURN_QUEUE,
                    COPILOT_SESSION_FIELD,
                    stopped,
                    turn_ids,
                )
        except pika.exceptions.ChannelClosedByBroker as exc:
            if exc.reply_code == 404:  # no turn was ever queued on this install
                return set()
            logger.warning(
                f"could not take stopped AutoPilot turns off the queue: {exc}"
            )
            return None
        except Exception as exc:
            logger.warning(
                f"could not take stopped AutoPilot turns off the queue: {exc}"
            )
            return None
        if found:
            logger.info(
                f"took the stopped AutoPilot turn(s) of {len(found)} chat(s) off the queue"
            )
        return found

    def _connect(self):
        import pika

        # pika logs at ERROR what is handled here.
        logging.getLogger("pika").setLevel(logging.CRITICAL)
        return pika.BlockingConnection(
            pika.ConnectionParameters(
                host="127.0.0.1",
                port=self.port,
                credentials=pika.PlainCredentials(self.user, self.password),
                connection_attempts=1,
                socket_timeout=5,
                blocked_connection_timeout=5,
            )
        )


def _drop(
    channel,
    queue: str,
    name: str,
    values: set[str],
    turn_ids: dict[str, str] | None = None,
) -> set[str]:
    """Every message is taken before any is given back: one that is given
    back could otherwise be taken again and another never looked at."""
    taken = []
    while True:
        method, _, body = channel.basic_get(queue=queue, auto_ack=False)
        if method is None:
            break
        named = _named(body, name, values)
        if (
            named is not None
            and turn_ids
            and named in turn_ids
            and json.loads(body).get("turn_id") != turn_ids[named]
        ):
            named = None
        taken.append((method.delivery_tag, named))
    for tag, stopped in taken:
        if stopped is None:
            channel.basic_nack(tag, requeue=True)
        else:
            channel.basic_ack(tag)
    return {stopped for _, stopped in taken if stopped is not None}


def _named(body: bytes, name: str, values: set[str]) -> str | None:
    """Which of `values` the message names, if any."""
    try:
        message = json.loads(body)
    except ValueError:
        return None
    named = message.get(name) if isinstance(message, dict) else None
    return named if isinstance(named, str) and named in values else None


def reconcile(
    connect: Callable, stopped_runs: Path, queue: Queue | None = None
) -> bool:
    """Settle unfinished runs before the executor sees their messages. It
    skips a message whose run has already ended."""
    stopped = read_stopped(stopped_runs)
    try:
        with closing(connect()) as connection:
            cursor = connection.cursor()
            _end(
                cursor,
                list(stopped.graph_runs),
                "TERMINATED",
                "stopped as AutoGPT closed",
            )
            cursor.execute(UNFINISHED_AND_OLD)
            stale = [row[0] for row in cursor.fetchall()]
            _end(
                cursor,
                stale,
                "FAILED",
                f"interrupted over {RESUME_WINDOW_HOURS} hours ago",
            )
            connection.commit()
    except Exception as exc:  # upstream moved a table: runs stay as they are
        logger.warning(f"could not settle interrupted agent runs: {exc}")
        return not stopped
    # Nothing consumes the queue yet: a message that is not found now is
    # not there. The original turn identity guards subsequent chat cleanup.
    turns = list(stopped.copilot_turns)
    if _end_turns(
        queue,
        stopped_runs.with_name(ENDED_TURNS_FILE),
        turns,
        lambda found: True,
        dict(stopped.copilot_turn_ids),
    ):
        _forget(stopped_runs)
        return True
    return False


def end_stopped(
    connect: Callable,
    stopped_runs: Path,
    cache: Cache,
    queue: Queue | None,
    chosen: Activity | None = None,
) -> None:
    """As the app closes, once the services are down and while the database
    and the broker are still up: end what the user chose to stop and its
    executor did not get to end. Written any earlier, it would race the
    executor's own last write.

    `chosen` is what this process was told to stop. The record on disk is
    for a process that dies before it gets here; one that could not be
    written must not cost the choice."""
    stopped = read_stopped(stopped_runs).merged(chosen or Activity())
    if not stopped:
        return
    try:
        with closing(connect()) as connection:
            _end(
                connection.cursor(),
                list(stopped.graph_runs),
                "TERMINATED",
                "stopped as AutoGPT closed",
            )
            connection.commit()
    except Exception as exc:  # the record stays: the next start applies it
        logger.warning(f"could not end the stopped agent runs: {exc}")
        return
    unfinished = list(stopped.copilot_turns)
    if not _end_turns(
        queue,
        stopped_runs.with_name(ENDED_TURNS_FILE),
        unfinished,
        set(unfinished).issubset,
        dict(stopped.copilot_turn_ids),
    ):
        return
    cache.release(stopped)
    _forget(stopped_runs)


def _end_turns(
    queue: Queue | None,
    ended_turns: Path,
    sessions: list[str],
    complete: Callable[[set[str]], bool],
    turn_ids: dict[str, str] | None = None,
) -> bool:
    """End AutoPilot turns whose executor is gone: see the top of this
    file. `complete` says, of the sessions a message was found for, whether
    those are all there can be. If so every chat is listed to be closed;
    if not, only those whose message is gone. False when not all of it is
    done."""
    if not sessions:
        return True
    found = (
        set(sessions)
        if queue is None
        else _taken_off(queue, sessions, complete, turn_ids)
    )
    done = found is not None and complete(found)
    to_close = (
        sessions
        if done
        else [session for session in sessions if session in (found or ())]
    )
    if to_close:
        old = _read(ended_turns)
        records = old if isinstance(old, list) else []
        for session in to_close:
            record = (
                {"session_id": session, "turn_id": turn_ids[session]}
                if turn_ids and session in turn_ids
                else session
            )
            if record not in records:
                records.append(record)
        if not _write(ended_turns, records):
            return False
    return done


def _taken_off(
    queue: Queue,
    sessions: list[str],
    complete: Callable[[set[str]], bool],
    turn_ids: dict[str, str] | None = None,
) -> set[str] | None:
    """The sessions whose messages were taken off the queue, asking again
    for a while (DROP_TRIES) when the broker did not answer or `complete`
    says some are still to come. None when it never answered."""
    found: set[str] | None = None
    for attempt in range(DROP_TRIES):
        if attempt:
            time.sleep(DROP_RETRY_SECONDS)
        remaining = set(sessions) - (found or set())
        dropped = (
            queue.drop_turns(remaining, turn_ids=turn_ids)
            if turn_ids
            else queue.drop_turns(remaining)
        )
        if dropped is None:
            continue
        found = (found or set()) | dropped
        if complete(found):
            break
    return found


def running(
    connect: Callable, among: Iterable[str] | None = None
) -> tuple[str, ...] | None:
    """The agent runs the database has as RUNNING: all of them, or those of
    `among`. None when it did not say."""
    ids = None if among is None else list(among)
    if ids is not None and not ids:
        return ()
    try:
        with closing(connect()) as connection:
            cursor = connection.cursor()
            if ids is None:
                cursor.execute(RUNNING)
            else:
                cursor.execute(RUNNING_AMONG, {"ids": ids})
            return tuple(row[0] for row in cursor.fetchall())
    except Exception as exc:
        logger.debug("could not ask the database what is running: %s", exc)
        return None


class Watch:
    """What is in progress, from one look at the locks to the next: what is
    locked, and the agent runs that lost their lock and go on (see the top
    of this file). `still_running` is `running` for the ids it is given.

    RUNNING is also what the database has for a run whose executor died.
    That one is not in progress: its lock is cleared before the next
    executor starts, and it is locked again when its message comes round,
    or never, when there is no message (`sweep_orphans`). `executor_gone`
    is how the watch is told."""

    def __init__(
        self,
        still_running: Callable[[tuple[str, ...]], tuple[str, ...] | None],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.still_running = still_running
        self.clock = clock
        self.seen = Activity()
        self.unlocked: tuple[str, ...] = ()
        self.asked = 0.0

    def look(self, locked: Activity) -> Activity:
        """`locked` is what the cache says now."""
        gone = tuple(
            run for run in self.seen.graph_runs if run not in locked.graph_runs
        )
        fresh = any(run not in self.unlocked for run in gone)
        if gone and (fresh or self.clock() - self.asked >= UNLOCKED_RUNS_EVERY_SECONDS):
            self.asked = self.clock()
            answer = self.still_running(gone)
            if answer is not None:  # not known: they count until it is
                gone = tuple(run for run in gone if run in answer)
        self.unlocked = gone
        self.seen = locked.merged(Activity(gone))
        return self.seen

    def executor_gone(self) -> None:
        """The process that hosted the executor has ended: no agent run seen
        so far is being worked on. What is, is locked at the next look."""
        self.unlocked = ()
        self.seen = Activity((), self.seen.copilot_turns, self.seen.copilot_turn_ids)


def sweep_orphans(
    connect: Callable,
    cache: Cache,
    queue_counts: Callable[[], dict[str, int] | None],
    before: datetime,
) -> bool:
    """Fail the runs from before this start (`before`, UTC without a zone)
    that nothing will ever finish: unfinished, held by no executor, and with
    no message left to stand for them. True once it is done, also when there
    was nothing to do; False to be asked again.

    Whether a run has a message cannot be asked, only how many messages
    there are. So it is done only when an executor is consuming, no message
    waits, and every message it has been handed is a run it holds (as many
    unsettled messages as locks, with the locks the same before and after the
    count): then no message is left for a run nobody holds. A message that
    is handed back and forth, as one is while every worker is busy, makes
    the numbers differ, and the question is put again a minute later."""
    try:
        held = cache.active()
        if held is None:
            return False
        with closing(connect()) as connection:
            cursor = connection.cursor()
            cursor.execute(UNFINISHED_BEFORE, {"before": before})
            lost = [
                row[0] for row in cursor.fetchall() if row[0] not in held.graph_runs
            ]
            if not lost:
                return True
            counts = queue_counts()
            if not counts or cache.active() != held or not _all_held(counts, held):
                return False
            _end(
                cursor,
                lost,
                "FAILED",
                "left unfinished by an earlier start, with nothing to resume them",
            )
            connection.commit()
            return True
    except Exception as exc:
        logger.warning(f"could not look for agent runs that nothing will finish: {exc}")
        return False


def _all_held(counts: dict[str, int], held: Activity) -> bool:
    return (
        counts.get("consumers", 0) >= 1
        and counts.get("messages_ready") == 0
        and counts.get("messages_unacknowledged") == len(held.graph_runs)
    )


class OrphanSweep:
    """When `sweep_orphans` is due: see the constants. It asks the broker's
    own tool, which can take many seconds to answer, so each try runs on a
    thread of its own. `tend`, which the supervisor's loop calls, starts one
    and on a later call collects what it said: the loop that restarts
    processes and hears a stop is never held up by it."""

    def __init__(
        self, sweep: Callable[[], bool], clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.sweep = sweep
        self.clock = clock
        self.due: float | None = None
        self.tries = 0
        self.trying: threading.Thread | None = None
        self.began = 0.0
        self.done = False

    def app_is_up(self) -> None:
        self.due = self.clock() + ORPHAN_SWEEP_AFTER_SECONDS

    def tend(self) -> None:
        if self.trying:
            if self.trying.is_alive():
                return
            self.trying = None
            self.settle()
        if self.due is None or self.clock() < self.due:
            return
        self.due = None
        self.began = self.clock()
        self.tries += 1
        self.done = False
        self.trying = threading.Thread(
            target=self.attempt, name="orphan-sweep", daemon=True
        )
        self.trying.start()

    def attempt(self) -> None:
        try:
            self.done = self.sweep()
        except Exception as exc:
            logger.warning(
                f"could not look for agent runs that nothing will finish: {exc}"
            )

    def settle(self) -> None:
        if self.done:
            return
        if self.tries >= ORPHAN_SWEEP_TRIES:
            logger.warning(
                "gave up looking for agent runs that nothing will finish: the queue never "
                "came to rest. The next start looks again."
            )
        else:
            self.due = self.began + ORPHAN_SWEEP_EVERY_SECONDS

    def wait(self, timeout: float | None = None) -> None:
        """For a try that is under way to end."""
        if self.trying:
            self.trying.join(timeout)


def record_stopped(path: Path, activity: Activity) -> None:
    """Persist stop intent before cancellation; a write failure is logged.
    The live supervisor also retains the choice for shutdown cleanup."""
    stopped = read_stopped(path).merged(activity)
    record: dict[str, object] = {
        "graph_runs": list(stopped.graph_runs),
        "copilot_turns": list(stopped.copilot_turns),
    }
    if stopped.copilot_turn_ids:
        record["copilot_turn_ids"] = dict(stopped.copilot_turn_ids)
    _write(path, record)


def read_stopped(path: Path) -> Activity:
    recorded = _read(path)
    if isinstance(recorded, list):  # as it was written before turns were recorded
        recorded = {"graph_runs": recorded}
    if not isinstance(recorded, dict):
        return Activity()
    identities = recorded.get("copilot_turn_ids", {})
    identities = identities if isinstance(identities, dict) else {}
    return Activity(
        _ids(recorded.get("graph_runs")),
        _ids(recorded.get("copilot_turns")),
        tuple(
            (session, turn)
            for session, turn in identities.items()
            if isinstance(session, str) and isinstance(turn, str) and turn
        ),
    )


def _write(path: Path, record: object) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fresh = path.with_name(path.name + ".new")
        # On the disk before it takes the name: a power cut must not leave
        # an empty file where the choice was.
        with open(fresh, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(record))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(fresh, path)
        return True
    except OSError as exc:
        logger.warning(f"could not write {path.name}: {exc}")
        return False


def _read(path: Path) -> object:
    """What is recorded in `path`; None when there is nothing to read."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        logger.warning(f"ignoring an unreadable {path.name}: {exc}")
        return None


def _ids(value: object) -> tuple[str, ...]:
    return (
        tuple(item for item in value if isinstance(item, str))
        if isinstance(value, list)
        else ()
    )


def _forget(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        # Applied again next time, which changes nothing: only runs that are
        # still unfinished are ended.
        logger.warning(f"could not remove {path.name}: {exc}")


def _end(cursor, ids: list[str], status: str, why: str) -> None:
    if not ids:
        return
    arguments = {"ids": ids, "status": status}
    cursor.execute(END_NODES, arguments)
    cursor.execute(END_RUNS, arguments)
    if cursor.rowcount:
        logger.info(f"marked {cursor.rowcount} agent run(s) {status}: {why}")
