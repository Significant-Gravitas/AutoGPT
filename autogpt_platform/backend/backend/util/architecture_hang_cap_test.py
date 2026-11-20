"""Tests cap the waits they expect to finish only generously.

Motivation (#16920, #17157): a CI runner can stall a test's thread for a
second or more, so a wait the test expects to finish, capped at a second,
fails while the code under test is fine. The wall-clock rule
(``architecture_wall_clock_bounds_test.py``) reads asserts on elapsed time,
which a cap is not.

The rule fails a cap a test puts on its own wait (``asyncio.wait_for``,
``asyncio.timeout``, ``asyncio.timeout_at``) under
``MIN_WALL_CLOCK_BOUND_SECONDS``, unless the test handles the cap's expiry
there or the cap says why: ``# hang-cap: <reason>`` on its statement or right
above. ``architecture_hang_cap_scan_test.py`` says which forms it reads.
Files that had such caps when the rule was written are held at their counts
in ``_HANG_CAP_COUNTS``, to be paid down.

It does not read a timeout passed into the code under test, such as
``connect_timeout=1``. Whether the test expects such a call to finish is in
what it asserts afterwards, not in the call:
``copilot/tools/bash_exec_test.py`` runs ``_run(box, timeout=2)`` and asserts
``timed_out is False``, then ``timeout=1`` and asserts ``timed_out is True``;
``copilot/sdk/tool_adapter_test.py`` expects ``wait_for_stash(timeout=1.0)``
to find the stash and ``timeout=0.05`` to return ``False``.
"""

import ast
import collections
import functools
import pathlib
import textwrap

import pytest

import backend.util.architecture_hang_cap_scan_test as reader
import backend.util.architecture_wall_clock_bounds_test as bounds
import backend.util.architecture_wall_clock_scan_test as scan

# Pre-existing short caps: "<path> -> how many the file has". Burn this down;
# do not add to it. A count above it fails the rule, a count below it fails
# the staleness check, so a fixed site lowers the number.
_HANG_CAP_COUNTS = {
    "backend/backend/api/external/rate_limit_test.py": 1,
    "backend/backend/api/features/account_deletion/autopilot_work_test.py": 4,
    "backend/backend/api/features/account_deletion/scheduled_work_test.py": 4,
    "backend/backend/api/features/experts/memory_schedule_hooks_test.py": 1,
    "backend/backend/api/features/onboarding_dump/rate_limit_test.py": 1,
    "backend/backend/api/features/onboarding_dump/service_test.py": 2,
    "backend/backend/api/features/workspace/embeddings_test.py": 1,
    "backend/backend/api/middleware/compression_test.py": 1,
    "backend/backend/api/ws_auth_transport_test.py": 3,
    "backend/backend/blocks/conductor/test_account_reads.py": 1,
    "backend/backend/blocks/discord/bot_blocks_test.py": 1,
    "backend/backend/blocks/slant3d/order_preparation_test.py": 5,
    "backend/backend/blocks/slant3d/slant3d_review_test.py": 1,
    "backend/backend/blocks/slant3d/slant3d_webhook_review_test.py": 2,
    "backend/backend/blocks/typesafe/_client_test.py": 3,
    "backend/backend/copilot/dream/scope_reconcile_test.py": 6,
    "backend/backend/copilot/gate/shadow_test.py": 1,
    "backend/backend/copilot/graphiti/context_refresh_test.py": 2,
    "backend/backend/copilot/graphiti/falkordb_probe_test.py": 1,
    "backend/backend/copilot/graphiti/ingest_test.py": 4,
    "backend/backend/copilot/graphiti/scope_lock_test.py": 1,
    "backend/backend/copilot/integration_refresh_test.py": 5,
    "backend/backend/copilot/model_test.py": 3,
    "backend/backend/copilot/sdk/codex_usage_persistence_test.py": 1,
    "backend/backend/copilot/sdk/codex_usage_test.py": 3,
    "backend/backend/copilot/sdk/first_turn_context_test.py": 1,
    "backend/backend/copilot/sdk/service_helpers_test.py": 2,
    "backend/backend/copilot/sdk/service_test.py": 1,
    "backend/backend/copilot/sdk/tool_display_integration_test.py": 2,
    "backend/backend/copilot/stream_registry_terminal_failure_test.py": 3,
    "backend/backend/copilot/tools/local_pc_shim_llm_test.py": 2,
    "backend/backend/data/subscription_lookup_cache_test.py": 1,
    "backend/backend/data/workspace_scope_reads_test.py": 1,
    "backend/backend/integrations/codex/device_login_test.py": 1,
    "backend/backend/integrations/codex/login_test.py": 6,
    "backend/backend/integrations/credential_lease_test.py": 1,
    "backend/backend/integrations/credentials_store_lock_test.py": 1,
    "backend/backend/util/feature_flag/posthog_test.py": 1,
    "backend/backend/util/request_body_test.py": 1,
    "backend/backend/util/request_url_logging_test.py": 1,
    "backend/backend/util/text_test.py": 1,
    "backend/test/native_mobile/auth_process_test.py": 1,
}


def test_tests_cap_the_waits_they_expect_to_finish_generously():
    found = _short_caps()
    counts = collections.Counter(rel for rel, _, _ in found)
    over = {rel for rel, n in counts.items() if n > _HANG_CAP_COUNTS.get(rel, 0)}
    short = [
        f"  {rel}:{line} caps a wait at {seconds:g} s"
        for rel, line, seconds in found
        if rel in over
    ]
    assert not short, (
        "These tests cap a wait they expect to finish below "
        f"{bounds.MIN_WALL_CLOCK_BOUND_SECONDS:g} s. A CI runner can stall a "
        "test for seconds, so they fail while the code is fine (#16920).\n"
        "- Give the wait a cap that only stops a hung test: a named constant "
        "like the file's _HANG_SECONDS = 10.\n"
        "- A wait meant to run out: catch its TimeoutError where it is "
        "awaited, with pytest.raises(TimeoutError), "
        "contextlib.suppress(TimeoutError) or `except TimeoutError:`.\n"
        "- If a short cap really is the point, say why in a comment on its "
        "line or right above: `# hang-cap: <reason>`.\n"
        "A file over its count in _HANG_CAP_COUNTS lists all its sites:\n"
        + "\n".join(short)
    )


def test_hang_cap_counts_are_not_stale():
    counts = collections.Counter(rel for rel, _, _ in _short_caps())
    stale = sorted(
        f"{rel}: {allowed} allowed, {counts[rel]} found"
        for rel, allowed in _HANG_CAP_COUNTS.items()
        if counts[rel] < allowed
    )
    assert not stale, (
        "These _HANG_CAP_COUNTS counts are above what the files still have; "
        "lower them (or drop the entry at 0):\n  " + "\n  ".join(stale)
    )


def test_the_scan_reads_test_modules_and_only_those(tmp_path: pathlib.Path):
    # A scan that finds nothing proves nothing; this one must find these two.
    source = "import asyncio\nasyncio.wait_for(asyncio.sleep(0), 1)\n"
    package = tmp_path / "pkg"
    package.mkdir()
    for name in ("checks_test.py", "test_checks.py", "checks.py"):
        (package / name).write_text(source)
    assert _scan(((package, "pkg"),), tmp_path) == [
        ("pkg/checks_test.py", 2, 1.0),
        ("pkg/test_checks.py", 2, 1.0),
    ]


@pytest.mark.parametrize(
    "body, expected",
    [
        # The caps, and the calls that are not one.
        ("await asyncio.wait_for(work(), 1)", [1.0]),
        ("await asyncio.wait_for(work(), timeout=2)", [2.0]),
        ("await asyncio.wait_for(work(), 4.5)", [4.5]),
        ("await asyncio.wait_for(work(), 5)", []),
        ("await asyncio.wait_for(work(), None)", []),
        ("await asyncio.wait_for(work(), budget())", []),
        ("await asyncio.tasks.wait_for(work(), 1)", [1.0]),
        ("await aio.wait_for(work(), 1)", [1.0]),
        ("await w(work(), 1)", [1.0]),
        ("await manager.wait_for('session', 1)", []),
        ("async with asyncio.timeout(1):\n    pass", [1.0]),
        ("async with asyncio.timeout(delay=2):\n    pass", [2.0]),
        ("async with asyncio.timeouts.timeout(1):\n    pass", [1.0]),
        ("async with asyncio.timeout(None):\n    pass", []),
        ("async with asyncio.timeout(10):\n    pass", []),
        (
            "async with asyncio.timeout_at(asyncio.get_running_loop().time() + 1):"
            "\n    pass",
            [1.0],
        ),
        (
            "loop = asyncio.get_running_loop()\ndeadline = loop.time() + 2\n"
            "async with asyncio.timeout_at(deadline):\n    pass",
            [2.0],
        ),
        ("async with asyncio.timeout_at(deadline):\n    pass", []),
        (
            "loop = asyncio.get_running_loop()\n"
            "async with asyncio.timeout_at(when=loop.time() + 1):\n    pass",
            [1.0],
        ),
        (
            "started = asyncio.get_running_loop().time()\n"
            "async with asyncio.timeout_at(started + 3):\n    pass",
            [3.0],
        ),
        ("await asyncio.wait({task}, timeout=1)", []),
        ("await LocalPCShim.for_session('s', connect_timeout=1)", []),
        # Its seconds, through names, constants and defaults.
        ("CAP = 0.5\nawait asyncio.wait_for(work(), CAP)", [0.5]),
        ("CAP = 0.5\nawait asyncio.wait_for(work(), 2 * CAP)", [1.0]),
        ("_HANG_SECONDS = 10\nawait asyncio.wait_for(work(), _HANG_SECONDS)", []),
        ("await asyncio.wait_for(work(), BOUND)", [1.0]),
        ("CAP = 1\nCAP = 2\nawait asyncio.wait_for(work(), CAP)", []),
        (
            "async def poll(timeout=2.0):\n"
            "    async with asyncio.timeout(timeout):\n        pass",
            [2.0],
        ),
        (
            "async def poll(*, timeout: float = 3):\n"
            "    await asyncio.wait_for(work(), timeout)",
            [3.0],
        ),
        (
            "async def poll(first, timeout=1, last=20):\n"
            "    await asyncio.wait_for(work(), timeout)",
            [1.0],
        ),
        ("async def poll(timeout):\n    await asyncio.wait_for(work(), timeout)", []),
        (
            "async def poll(timeout=1):\n    timeout = timeout * scale()\n"
            "    await asyncio.wait_for(work(), timeout)",
            [],
        ),
        (
            "async def poll(timeout=10):\n    await asyncio.wait_for(work(), timeout)",
            [],
        ),
        (
            "for timeout in (1, 2):\n    await asyncio.wait_for(work(), timeout)",
            [],
        ),
        (
            "CAP = 1\ndef poll(waiter=asyncio.wait_for(work(), CAP)):\n    CAP = 10",
            [1.0],
        ),
        # The handlers of its expiry, and what is not one.
        (
            "with pytest.raises(TimeoutError):\n"
            "    await asyncio.wait_for(work(), 1)",
            [],
        ),
        (
            "with pytest.raises(asyncio.TimeoutError):\n"
            "    await asyncio.wait_for(work(), 1)",
            [],
        ),
        (
            "with pytest.raises((ValueError, TimeoutError)):\n"
            "    await asyncio.wait_for(work(), 1)",
            [],
        ),
        (
            "with pytest.raises(expected_exception=TimeoutError):\n"
            "    await asyncio.wait_for(work(), 1)",
            [],
        ),
        (
            "with pytest.raises(FutureTimeout):\n"
            "    await asyncio.wait_for(work(), 1)",
            [],
        ),
        (
            "with pytest.raises(ValueError):\n" "    await asyncio.wait_for(work(), 1)",
            [1.0],
        ),
        (
            "with pytest.raises(TimeoutError, match='did not connect'):\n"
            "    await asyncio.wait_for(work(), 1)",
            [1.0],
        ),
        (
            "with pytest.raises(TimeoutError, match=None):\n"
            "    await asyncio.wait_for(work(), 1)",
            [],
        ),
        (
            "with pytest.raises(TimeoutError, check=lambda exc: bool(str(exc))):\n"
            "    await asyncio.wait_for(work(), 1)",
            [1.0],
        ),
        (
            "with pytest.raises(TimeoutError, check=None):\n"
            "    await asyncio.wait_for(work(), 1)",
            [],
        ),
        (
            "with pytest.raises(builtins.TimeoutError):\n"
            "    await asyncio.wait_for(work(), 1)",
            [],
        ),
        ("with pytest.raises(Expired):\n    await asyncio.wait_for(work(), 1)", []),
        (
            "with pytest.raises(AsyncTimeout):\n    await asyncio.wait_for(work(), 1)",
            [],
        ),
        (
            "try:\n    await asyncio.wait_for(work(), 1)\n"
            "except asyncio.exceptions.TimeoutError:\n    pass",
            [],
        ),
        (
            "TimeoutError = ValueError\nwith pytest.raises(TimeoutError):\n"
            "    await asyncio.wait_for(work(), 1)",
            [1.0],
        ),
        (
            "with contextlib.suppress(TimeoutError, asyncio.BrokenBarrierError):\n"
            "    await asyncio.wait_for(work(), 1)",
            [],
        ),
        ("with suppress(TimeoutError):\n    await asyncio.wait_for(work(), 1)", []),
        (
            "with contextlib.suppress(ValueError):\n"
            "    await asyncio.wait_for(work(), 1)",
            [1.0],
        ),
        (
            "with pytest.raises(TimeoutError), contextlib.suppress(ValueError):\n"
            "    await asyncio.wait_for(work(), 1)",
            [],
        ),
        (
            "try:\n    await asyncio.wait_for(work(), 1)\n"
            "except (TimeoutError, asyncio.BrokenBarrierError):\n    pass",
            [],
        ),
        (
            "try:\n    await asyncio.wait_for(work(), 1)\n"
            "except Exception:\n    pass",
            [1.0],
        ),
        ("try:\n    await asyncio.wait_for(work(), 1)\nexcept:\n    pass", [1.0]),
        (
            "try:\n    await asyncio.wait_for(work(), 1)\n"
            "except BaseException:\n    pass",
            [1.0],
        ),
        (
            "try:\n    await asyncio.wait_for(work(), 1)\n"
            "except (TimeoutError, Exception):\n    pass",
            [],
        ),
        (
            "with contextlib.suppress(Exception):\n"
            "    await asyncio.wait_for(work(), 1)",
            [1.0],
        ),
        (
            "with pytest.raises(Exception):\n    await asyncio.wait_for(work(), 1)",
            [1.0],
        ),
        (
            "try:\n    await asyncio.wait_for(work(), 1)\n"
            "except ValueError:\n    pass",
            [1.0],
        ),
        (
            "try:\n    await asyncio.wait_for(work(), 1)\n"
            "except TimeoutError:\n    raise AssertionError('hung')",
            [1.0],
        ),
        (
            "try:\n    await asyncio.wait_for(work(), 1)\n"
            "except TimeoutError:\n    pytest.fail('hung')",
            [1.0],
        ),
        (
            "try:\n    await asyncio.wait_for(work(), 1)\n"
            "except TimeoutError:\n    assert done.is_set()",
            [1.0],
        ),
        (
            "try:\n    await asyncio.wait_for(work(), 1)\n"
            "except TimeoutError:\n    def later():\n        raise ValueError",
            [],
        ),
        (
            "try:\n    await asyncio.wait_for(work(), 1)\n"
            "except TimeoutError:\n    raise\nexcept Exception:\n    pass",
            [1.0],
        ),
        (
            "try:\n    await asyncio.wait_for(work(), 1)\n"
            "except TimeoutError:\n    raise\nexcept asyncio.TimeoutError:\n    pass",
            [1.0],
        ),
        (
            "try:\n    await asyncio.wait_for(work(), 1)\n"
            "except OSError:\n    pass\nexcept TimeoutError:\n    pass",
            [1.0],
        ),
        (
            "try:\n    await asyncio.wait_for(work(), 1)\n"
            "except builtins.OSError:\n    pass\nexcept TimeoutError:\n    pass",
            [1.0],
        ),
        (
            "try:\n    await asyncio.wait_for(work(), 1)\n"
            "except ValueError:\n    raise\nexcept TimeoutError:\n    pass",
            [],
        ),
        (
            "try:\n    pass\n"
            "except TimeoutError:\n    await asyncio.wait_for(work(), 1)",
            [1.0],
        ),
        (
            "try:\n    pass\nexcept TimeoutError:\n    pass\n"
            "else:\n    await asyncio.wait_for(work(), 1)",
            [1.0],
        ),
        (
            "try:\n    pass\nexcept TimeoutError:\n    pass\n"
            "finally:\n    await asyncio.wait_for(work(), 1)",
            [1.0],
        ),
        (
            "with pytest.raises(TimeoutError):\n    async def inner():\n"
            "        await asyncio.wait_for(work(), 1)\n    await inner()",
            [1.0],
        ),
        # Its reason, in a comment on its statement or right above.
        ("await asyncio.wait_for(work(), 1)  # hang-cap: why", []),
        ("# hang-cap: why\nawait asyncio.wait_for(work(), 1)", []),
        ("# hang-cap: why\n# more\nawait asyncio.wait_for(work(), 1)", []),
        (
            "result = await asyncio.wait_for(\n    work(), 1\n)  # hang-cap: why",
            [],
        ),
        (
            "assert (\n    await asyncio.wait_for(work(), 1)\n) == 3  # hang-cap: why",
            [],
        ),
        (
            "# hang-cap: why\nfirst, second = await asyncio.gather(\n"
            "    asyncio.wait_for(work(), 1), work()\n)",
            [],
        ),
        ("async with asyncio.timeout(1):  # hang-cap: why\n    pass", []),
        (
            "async with asyncio.timeout(1), aclosing(\n    stream()\n"
            ") as frames:  # hang-cap: why\n    pass",
            [],
        ),
        (
            "async with (\n    asyncio.timeout(1),\n    aclosing(stream()) as frames,\n"
            "):  # hang-cap: why\n    pass",
            [],
        ),
        (
            "async with asyncio.timeout(1):\n    # hang-cap: no\n    pass",
            [1.0],
        ),
        (
            "if (\n    await asyncio.wait_for(work(), 1)\n):  # hang-cap: why\n    pass",
            [],
        ),
        ("async with asyncio.timeout(1):\n    pass  # hang-cap: no", [1.0]),
        ("await asyncio.wait_for(work(), 1)  # hang-cap:", [1.0]),
        ("await asyncio.wait_for(work(), 1)  # wall-clock: no", [1.0]),
        ("log('# hang-cap: no', await asyncio.wait_for(work(), 1))", [1.0]),
    ],
)
def test_the_scan_reads_each_form(
    body: str, expected: list[float], monkeypatch: pytest.MonkeyPatch
):
    # A synthetic module, so a change to real code cannot move this test.
    fake = {"backend.fake.limits": "BOUND = 1.0\n"}
    monkeypatch.setattr(
        scan,
        "_source_tree",
        lambda module: (
            (ast.parse(fake[module]), "backend.fake") if module in fake else None
        ),
    )
    source = (
        "import asyncio\nimport asyncio as aio\nimport contextlib\nimport pytest\n"
        "from asyncio import wait_for as w\nfrom contextlib import suppress\n"
        "from concurrent.futures import TimeoutError as FutureTimeout\n"
        "import builtins\nfrom builtins import TimeoutError as Expired\n"
        "from asyncio import TimeoutError as AsyncTimeout\n"
        "from backend.fake.limits import BOUND\n\n"
        "async def case():\n" + textwrap.indent(body, "    ") + "\n"
    )
    caps = reader.hang_caps(source, "backend.util")
    short = [seconds for _, seconds, reason in caps if _too_short(seconds, reason)]
    assert short == pytest.approx(expected)


@pytest.mark.parametrize(
    "source",
    [
        "import asyncio\nasyncio.wait_for(work(), 1)\n",
        "from asyncio import wait_for\nwait_for(work(), 1)\n",
        "from asyncio import timeout as cap\ncap(1)\n",
        "from asyncio import get_running_loop, timeouts\n"
        "timeouts.timeout_at(get_running_loop().time() + 1)\n",
    ],
)
def test_the_scan_reads_a_module_however_it_imports_its_cap(source: str):
    # The scan skips a module that names no cap, unparsed: each of these does.
    assert [seconds for _, seconds, _ in reader.hang_caps(source, "pkg")] == [1.0]


def _too_short(seconds: float, reason: bool) -> bool:
    return seconds < bounds.MIN_WALL_CLOCK_BOUND_SECONDS and not reason


@functools.cache
def _short_caps() -> list[tuple[str, int, float]]:
    return _scan(bounds._SCAN_ROOTS, bounds._PLATFORM_ROOT)


def _scan(
    roots: tuple[tuple[pathlib.Path, str], ...], base: pathlib.Path
) -> list[tuple[str, int, float]]:
    """Each cap of the test modules under *roots* on work the test expects to
    finish, shorter than the bound and with no reason: its file, line and
    seconds."""
    found: list[tuple[str, int, float]] = []
    for root, package in roots:
        assert root.is_dir(), f"{root} moved; point _SCAN_ROOTS at it"
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts or not bounds._is_test_module(path):
                continue
            module_package = ".".join([package, *path.relative_to(root).parts[:-1]])
            rel = path.relative_to(base).as_posix()
            caps = reader.hang_caps(path.read_text(encoding="utf-8"), module_package)
            found += [
                (rel, line, seconds)
                for line, seconds, reason in caps
                if _too_short(seconds, reason)
            ]
    return found
