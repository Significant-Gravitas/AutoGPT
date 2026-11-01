"""
Architectural tests for the backend package.

Each rule here exists to prevent a *class* of bug, not to police style.
When adding a rule, document the incident or failure mode that motivated
it so future maintainers know whether the rule still earns its keep.
"""

import ast
import collections
import functools
import pathlib
import sys
from collections.abc import Iterable

import pytest

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Rule: no process-wide @cached(...) around event-loop-bound async clients
# ---------------------------------------------------------------------------
#
# Motivation: `backend.util.cache.cached` stores its result in a process-wide
# dict for ttl_seconds. Async clients (AsyncOpenAI, httpx.AsyncClient,
# AsyncRabbitMQ, ...) wrap connection pools whose internal
# asyncio primitives lazily bind to the first event loop that uses them. The
# executor runs two long-lived loops on separate threads; once the cache is
# populated from loop A, any subsequent call from loop B raises
# `RuntimeError: ... bound to a different event loop`, surfaced as an opaque
# `APIConnectionError: Connection error.` and poisons the cache for a full
# TTL window.
#
# Use `per_loop_cached` (keyed on id(running loop)) or construct per-call.

LOOP_BOUND_TYPES = frozenset(
    {
        "AsyncOpenAI",
        "LangfuseAsyncOpenAI",
        "AsyncClient",  # httpx, openai internal
        "AsyncRabbitMQ",
        "AsyncRedisExecutionEventBus",
    }
)

# Pre-existing offenders tracked for future cleanup. Exclude from this test
# so the rule can still catch NEW violations without blocking unrelated PRs.
_KNOWN_OFFENDERS = frozenset(
    {
        "util/clients.py get_openai_client",
        # Inner helper extracted from ``get_openai_client`` so the local-
        # transport branch shares one cached client across both
        # ``prefer_openrouter`` arms (no semantic change vs the parent
        # offender — same loop-binding caveat applies, will be migrated
        # together when ``get_openai_client`` is moved to ``per_loop_cached``).
        "util/clients.py _get_local_openai_client",
    }
)


def _decorator_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _annotation_names(annotation: ast.expr | None) -> set[str]:
    if annotation is None:
        return set()
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        try:
            parsed = ast.parse(annotation.value, mode="eval").body
        except SyntaxError:
            return set()
        return _annotation_names(parsed)
    names: set[str] = set()
    for child in ast.walk(annotation):
        if isinstance(child, ast.Name):
            names.add(child.id)
        elif isinstance(child, ast.Attribute):
            names.add(child.attr)
    return names


def _iter_backend_py_files():
    for path in BACKEND_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        yield path


# ---------------------------------------------------------------------------
# Rule: backend code uses pydantic, not stdlib dataclasses
# ---------------------------------------------------------------------------
#
# Motivation: pydantic ``BaseModel`` gives us validation, (de)serialization,
# and JSON-schema generation for free, and the whole backend relies on that
# consistency across the API, the executor, and blocks. A plain ``@dataclass``
# silently bypasses all of it. Shared, framework-agnostic helpers living under
# a ``lib``/``libs`` directory are exempt. Pydantic dataclasses
# (``pydantic.dataclasses``) are a different module and are intentionally
# allowed.

# Path segments whose subtrees are exempt: shared helpers under a `lib`/`libs`
# folder may use stdlib dataclasses; everything else must not.
_DATACLASS_EXEMPT_DIR_NAMES = frozenset({"lib", "libs"})

# Pre-existing offenders (present in `dev` as of merge-base 9822fe4b) tracked
# for future migration to pydantic. Excluded so the rule catches NEW violations
# without blocking unrelated PRs. Paths are posix-relative to the backend
# package root. Burn this list down; do not add to it.
_DATACLASS_KNOWN_OFFENDERS = frozenset(
    {
        "api/features/builder/db.py",
        # ``search/`` siblings to the original ``store/`` offenders — landed
        # in dev's global-search rollout (#13217) after this allowlist was
        # last snapshotted. Tracked here so the rule still catches genuinely
        # new violations; migrate alongside the ``store/`` originals.
        "api/features/search/content_handlers.py",
        "api/features/search/hybrid_search.py",
        "api/features/store/content_handlers.py",
        "api/features/store/hybrid_search.py",
        "blocks/codex.py",
        "copilot/baseline/service.py",
        "copilot/bot/adapters/base.py",
        "copilot/bot/bot_backend.py",
        "copilot/sdk/compaction.py",
        "copilot/sdk/file_ref.py",
        "copilot/sdk/service.py",
        "copilot/sdk/service_test.py",
        "copilot/sdk/session_waiter.py",
        "copilot/stream_registry.py",
        # Self-distilled skills registry added by dev's copilot-skills PR
        # series; predates this allowlist refresh, same migration plan.
        "copilot/tools/skills.py",
        "copilot/tools/helpers.py",
        "copilot/transcript.py",
        "util/cache.py",
        "util/prompt.py",
        "util/sandbox_files.py",
        "util/tool_call_loop.py",
    }
)


def _dataclass_imports(path: pathlib.Path) -> list[tuple[int, str]]:
    """Return (lineno, description) for every stdlib ``dataclasses`` import."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return []
    offenses: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "dataclasses" or alias.name.startswith("dataclasses."):
                    offenses.append((node.lineno, "imports the `dataclasses` module"))
        elif isinstance(node, ast.ImportFrom):
            # `from dataclasses import ...`; `from pydantic.dataclasses import ...`
            # has a different module and is intentionally allowed.
            if node.module == "dataclasses":
                names = ", ".join(alias.name for alias in node.names)
                offenses.append((node.lineno, f"imports `{names}` from `dataclasses`"))
    return offenses


def test_backend_uses_pydantic_not_dataclasses():
    offenders: list[str] = []
    for py in _iter_backend_py_files():
        rel = py.relative_to(BACKEND_ROOT)
        if set(rel.parts) & _DATACLASS_EXEMPT_DIR_NAMES:
            continue
        if rel.as_posix() in _DATACLASS_KNOWN_OFFENDERS:
            continue
        for lineno, reason in _dataclass_imports(py):
            offenders.append(f"  backend/{rel}:{lineno} {reason}")

    assert not offenders, (
        "Backend code must use a pydantic `BaseModel` instead of stdlib "
        "`dataclasses` (only shared `lib`/`libs` helpers are exempt). "
        "Replace these with `pydantic.BaseModel`:\n" + "\n".join(sorted(offenders))
    )


# ---------------------------------------------------------------------------
# Rule: tests never patch the shared clock or sleep
# ---------------------------------------------------------------------------
#
# Motivation: ``patch("asyncio.sleep")``, ``patch.object(time, "time")`` and
# ``patch("backend.x.y.asyncio.sleep")`` (where ``y`` does ``import asyncio``)
# all replace the attribute on the one shared module, so every coroutine and
# library in the test's event loop sleeps or reads the clock through the stub.
# That made the copilot dispatch-retry test flaky (OPEN-3475), stalled the
# desktop opener test (#14633), and expired an ``ExpiringDict`` lock table
# under ``creds_manager_lock_test.py``. Patch a name the module under test bound
# itself (``from asyncio import sleep``, then ``patch("backend.x.y.sleep")``),
# or inject the clock or sleep.

_GLOBAL_CLOCK_PATCH_TARGETS = frozenset(
    {"time.sleep", "asyncio.sleep", "time.time", "time.monotonic"}
)

# Pre-existing offenders: "<path> <target>" -> how many such patches the file
# makes. Burn this down; do not add to it. A count above it fails the rule, a
# count below it fails the staleness check, so a fixed site lowers the number.
_GLOBAL_CLOCK_PATCH_OFFENDERS = {
    "api/features/onboarding_dump/transcription_test.py asyncio.sleep": 2,
    "api/features/orgs/routes_test.py asyncio.sleep": 1,
    "blocks/desktop/desktop_api_test.py asyncio.sleep": 1,
    "blocks/exa/cost_tracking_test.py asyncio.sleep": 4,
    "blocks/rss_test.py asyncio.sleep": 2,
    "blocks/time_blocks_test.py asyncio.sleep": 2,
    "copilot/baseline/reasoning_test.py time.monotonic": 1,
    "copilot/bot/handler_test.py asyncio.sleep": 3,
    "copilot/bot/turn_stream_test.py time.monotonic": 6,
    "copilot/executor/manager_block_warmup_test.py time.sleep": 1,
    "copilot/expert_context_test.py asyncio.sleep": 2,
    "copilot/graphiti/falkordb_driver_test.py asyncio.sleep": 4,
    "copilot/optimize_blocks_test.py asyncio.sleep": 3,
    "copilot/sdk/openrouter_cost_test.py asyncio.sleep": 1,
    "copilot/sdk/p0_guardrails_test.py asyncio.sleep": 7,
    "copilot/sdk/provider_out_of_credits_test.py asyncio.sleep": 1,
    "copilot/sdk/retry_scenarios_test.py asyncio.sleep": 3,
    "copilot/tools/e2b_sandbox_test.py asyncio.sleep": 11,
    "copilot/tools/e2b_sandbox_test.py time.time": 1,
    "copilot/tools/graphiti_forget_retry_test.py asyncio.sleep": 3,
    "copilot/tools/skills_learning_test.py asyncio.sleep": 1,
    "data/credential_refresh_test.py time.time": 1,
    "data/subscription_upgrade_fixtures.py time.time": 1,
    "data/subscription_upgrade_test.py time.time": 1,
    "data/user_test.py time.time": 1,
    "executor/manager_block_warmup_test.py time.sleep": 1,
    "executor/scheduler_index_recovery_test.py time.sleep": 1,
    "integrations/creds_manager_lock_test.py time.time": 1,
    "integrations/creds_manager_test.py time.time": 2,
    "integrations/oauth/microsoft_365_copilot_test.py time.time": 1,
    "integrations/oauth/todoist_test.py time.time": 1,
    "monitoring/auth_identity_monitor_test.py time.monotonic": 1,
    "notifications/checkout_backfill_test.py asyncio.sleep": 6,
    "notifications/consumer_test.py asyncio.sleep": 5,
    "notifications/mailerlite_backfill_test.py asyncio.sleep": 1,
    "notifications/mailerlite_field_backfill_test.py asyncio.sleep": 1,
    "notifications/role_backfill_test.py asyncio.sleep": 1,
    "notifications/subscriber_fields_test.py asyncio.sleep": 1,
    "notifications/trial_shared_delivery_test.py asyncio.sleep": 1,
    "util/e2b_template_test.py asyncio.sleep": 6,
    "util/retry_test.py time.time": 1,
}


def test_no_global_clock_or_sleep_patches():
    found = _global_clock_patches()
    counts = collections.Counter((rel, target) for rel, target, _ in found)
    over = {
        key
        for key, n in counts.items()
        if n > _GLOBAL_CLOCK_PATCH_OFFENDERS.get(f"{key[0]} {key[1]}", 0)
    }
    new = sorted(
        f"  backend/{rel}:{lineno} patches {target}"
        for rel, target, lineno in found
        if (rel, target) in over
    )
    assert not new, (
        "Tests must not patch time.sleep, asyncio.sleep, time.time or "
        "time.monotonic on the shared module, by any path that reaches it: "
        "the stub reaches every coroutine and library in the process. Patch "
        "a name the module under test bound itself (`from asyncio import "
        'sleep`, then patch("backend.x.y.sleep")) or inject the clock or '
        "sleep. An allowlisted file over its count lists all its sites:\n"
        + "\n".join(new)
    )


def test_global_clock_patch_offenders_are_not_stale():
    counts = collections.Counter(
        f"{rel} {target}" for rel, target, _ in _global_clock_patches()
    )
    stale = sorted(
        f"{key}: {allowed} allowed, {counts[key]} found"
        for key, allowed in _GLOBAL_CLOCK_PATCH_OFFENDERS.items()
        if counts[key] < allowed
    )
    assert not stale, (
        "These _GLOBAL_CLOCK_PATCH_OFFENDERS counts are above what the files "
        "still patch; lower them (or drop the entry at 0):\n  " + "\n  ".join(stale)
    )


@pytest.mark.parametrize(
    "source, expected",
    [
        ('patch("asyncio.sleep")', ["asyncio.sleep"]),
        ('patch("backend.fake.shares.asyncio.sleep")', ["asyncio.sleep"]),
        ('patch("backend.fake.binds.sleep")', []),
        ('import time\npatch.object(time, "sleep")', ["time.sleep"]),
        (
            "import time\npatch.multiple(time, sleep=1, monotonic=2)",
            ["time.sleep", "time.monotonic"],
        ),
        ('from unittest.mock import patch as p\np("time.time")', ["time.time"]),
        ('patch(target="time.sleep")', ["time.sleep"]),
        (
            'def t():\n    import time as clock\n    patch.object(clock, "sleep")',
            ["time.sleep"],
        ),
        (
            '_M = "backend.fake.shares"\npatch(f"{_M}.asyncio.sleep")',
            ["asyncio.sleep"],
        ),
        ('_T = "time.sleep"\npatch(_T)', ["time.sleep"]),
        ('patch(f"{unknown()}.asyncio.sleep")', []),
    ],
)
def test_clock_patch_resolver_reads_each_form(
    source: str, expected: list[str], monkeypatch: pytest.MonkeyPatch
):
    # Synthetic modules, so an import change in real code cannot move this test.
    fake = {
        "backend.fake.shares": "import asyncio\n",
        "backend.fake.binds": "from time import sleep\n",
    }
    monkeypatch.setattr(
        sys.modules[__name__],
        "_module_tree",
        lambda module: (
            (ast.parse(fake[module]), "backend.fake") if module in fake else None
        ),
    )
    tree = ast.parse(source)
    names = _import_bindings(tree.body, "backend.util")
    consts = _string_constants(tree)
    found = [
        target
        for call, scope in _calls_in_scope(tree, names, "backend.util")
        for target in _patched_clock_targets(call, scope, consts)
    ]
    assert found == expected


@functools.cache
def _global_clock_patches() -> tuple[tuple[str, str, int], ...]:
    found: list[tuple[str, str, int]] = []
    for py in _iter_backend_py_files():
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        except SyntaxError:
            continue
        rel = py.relative_to(BACKEND_ROOT).as_posix()
        package = _package_of(rel)
        names = _import_bindings(tree.body, package)
        consts = _string_constants(tree)
        for call, scope in _calls_in_scope(tree, names, package):
            for target in _patched_clock_targets(call, scope, consts):
                found.append((rel, target, call.lineno))
    return tuple(found)


def _calls_in_scope(node: ast.AST, names: dict[str, str], package: str):
    """Each call with the import bindings visible to it, so a test-local
    `import time as clock` resolves inside that test."""
    for child in ast.iter_child_nodes(node):
        scope = names
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scope = names | _import_bindings(ast.walk(child), package)
        elif isinstance(child, ast.Call):
            yield child, names
        yield from _calls_in_scope(child, scope, package)


def _patched_clock_targets(
    call: ast.Call, names: dict[str, str], consts: dict[str, str]
) -> list[str]:
    """Shared-module clock targets of `patch(...)`, `patch.object(...)`,
    `patch.multiple(...)` and `setattr(...)`, aliases included. A target built
    from anything but literals and module-level string constants is skipped."""
    func = _qualified(call.func, names) or ""
    kind = func.rsplit(".", 2)[-2:]
    first = _argument(call, 0, "target")
    second = _argument(call, 1, "attribute", "name")
    dotted = _str_value(first, consts) if first else None
    if kind[-1] in ("patch", "setattr") and dotted:
        owner, _, attr = dotted.rpartition(".")
        pairs = [(owner, attr)]
    elif (kind == ["patch", "object"] or kind[-1] == "setattr") and first and second:
        owner, attr = _owner_path(first, names, consts), _str_value(second, consts)
        pairs = [(owner, attr)] if owner and attr else []
    elif kind == ["patch", "multiple"] and first:
        owner = _owner_path(first, names, consts)
        keywords = [kw.arg for kw in call.keywords if kw.arg != "target"]
        pairs = [(owner, attr) for attr in keywords if owner and attr]
    else:
        return []
    targets = [f"{_resolve(owner)}.{attr}" for owner, attr in pairs]
    return [t for t in targets if t in _GLOBAL_CLOCK_PATCH_TARGETS]


def _argument(call: ast.Call, index: int, *keywords: str) -> ast.expr | None:
    if len(call.args) > index:
        return call.args[index]
    return next((kw.value for kw in call.keywords if kw.arg in keywords), None)


def _owner_path(
    node: ast.expr, names: dict[str, str], consts: dict[str, str]
) -> str | None:
    return _str_value(node, consts) or _qualified(node, names)


def _qualified(node: ast.expr, names: dict[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        return names.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        base = _qualified(node.value, names)
        return f"{base}.{node.attr}" if base else None
    return None


def _resolve(path: str, depth: int = 0) -> str:
    """Follow module-level import bindings, so
    "backend.blocks.exa.research.asyncio" -> "asyncio" when research.py
    does `import asyncio`."""
    parts = path.split(".")
    for i in range(len(parts), 0, -1):
        found = _module_tree(".".join(parts[:i]))
        if found is None:
            continue
        if i == len(parts) or depth > 5:
            return path
        tree, package = found
        binding = _import_bindings(tree.body, package).get(parts[i])
        if binding is None:
            return path
        return _resolve(".".join([binding, *parts[i + 1 :]]), depth + 1)
    return path


@functools.cache
def _module_tree(module: str) -> tuple[ast.Module, str] | None:
    """The parsed backend module and the package its relative imports use."""
    if module != "backend" and not module.startswith("backend."):
        return None
    base = BACKEND_ROOT.joinpath(*module.split(".")[1:])
    for path, package in (
        (base / "__init__.py", module),
        (base.with_suffix(".py"), module.rpartition(".")[0]),
    ):
        if path.is_file():
            try:
                return ast.parse(path.read_text(encoding="utf-8")), package
            except SyntaxError:
                return None
    return None


def _import_bindings(nodes: Iterable[ast.AST], package: str) -> dict[str, str]:
    """Names the given import statements bind, mapped to their dotted origin."""
    names: dict[str, str] = {}
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    names[alias.asname] = alias.name
                else:
                    top = alias.name.split(".")[0]
                    names[top] = top
        elif isinstance(node, ast.ImportFrom):
            origin = _absolute_module(node, package)
            for alias in node.names:
                names[alias.asname or alias.name] = f"{origin}.{alias.name}"
    return names


def _absolute_module(node: ast.ImportFrom, package: str) -> str:
    if not node.level:
        return node.module or ""
    base = package.split(".")[: len(package.split(".")) - node.level + 1]
    return ".".join([*base, *([node.module] if node.module else [])])


def _package_of(rel: str) -> str:
    return ".".join(["backend", *rel.split("/")[:-1]])


def _string_constants(tree: ast.Module) -> dict[str, str]:
    """Module-level `NAME = "..."` (or an f-string of such) assignments."""
    consts: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target, value = node.target, node.value
        else:
            continue
        resolved = _str_value(value, consts)
        if isinstance(target, ast.Name) and resolved is not None:
            consts[target.id] = resolved
    return consts


def _str_value(node: ast.expr, consts: dict[str, str]) -> str | None:
    """A literal, a module-level string constant, or an f-string of those."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return consts.get(node.id)
    if isinstance(node, ast.JoinedStr):
        parts = []
        for part in node.values:
            expr = part
            if isinstance(part, ast.FormattedValue):
                if part.conversion != -1 or part.format_spec is not None:
                    return None
                expr = part.value
            value = _str_value(expr, consts)
            if value is None:
                return None
            parts.append(value)
        return "".join(parts)
    return None


# ---------------------------------------------------------------------------
# Rule: copilot code touches the chat's working directory only through host_fs
# ---------------------------------------------------------------------------
#
# Motivation: copilot code reaches the chat's working directory only through
# ``backend/copilot/host_fs.py``, which opens a checked path without following a
# link. A listed call acts on a path outside that directory, or on all of it.

_WORKDIR_NAMES = frozenset(
    {
        "get_sdk_cwd",
        "get_workspace_dir",
        "is_allowed_local_path",
        "make_session_path",
        "resolve_allowed_local_path",
        "sdk_cwd",
        "validate_ephemeral_path",
        "workdir_root",
        "working_dir",
    }
)
_BY_NAME_FILE_CALLS = {
    "os": frozenset(
        {
            "access", "chmod", "chown", "lchown", "link", "listdir", "lstat",
            "makedirs", "mkdir", "mkfifo", "open", "readlink", "remove",
            "removedirs", "rename", "renames", "replace", "rmdir", "scandir",
            "stat", "symlink", "truncate", "unlink", "utime", "walk",
        }
    ),
    "os.path": frozenset(
        {
            "exists", "getatime", "getctime", "getmtime", "getsize", "isdir",
            "isfile", "islink", "lexists", "samefile",
        }
    ),
    "shutil": frozenset(
        {"chown", "copy", "copy2", "copyfile", "copymode", "copystat", "copytree",
         "move", "rmtree"}
    ),
    "io": frozenset({"open"}),
    "glob": frozenset({"glob", "iglob"}),
}  # fmt: skip
_PATH_METHODS = frozenset(
    {
        "chmod", "exists", "glob", "hardlink_to", "is_dir", "is_file",
        "is_symlink", "iterdir", "lstat", "mkdir", "open", "read_bytes",
        "read_text", "readlink", "rglob", "rmdir", "samefile", "stat",
        "symlink_to", "touch", "unlink", "write_bytes", "write_text",
    }
)  # fmt: skip
# Path's forms take one argument; str.replace takes two.
_ONE_ARG_PATH_METHODS = frozenset({"rename", "replace"})

# "<path> <function> <call>" -> how many such calls it makes, and what they act on.
_CLI_PROJECTS = "the CLI's projects directory"
_WHOLE_DIR = "a whole directory; shutil.rmtree follows no link inside it"
_CLI_TMPDIR = "the CLI's temp directory, beside the working directory"
_CREATES_IT = "creates the working directory itself"
_WORKDIR_FILE_CALL_OFFENDERS: dict[str, tuple[int, str]] = {
    "copilot/baseline/service.py stream_chat_completion_baseline shutil.rmtree": (
        1,
        _WHOLE_DIR,
    ),
    "copilot/sdk/cost_tracking.py read_cli_session_usage .read_text": (
        1,
        _CLI_PROJECTS,
    ),
    "copilot/sdk/service.py _cleanup_sdk_tool_results shutil.rmtree": (2, _WHOLE_DIR),
    "copilot/sdk/service.py _make_cli_tmpdir os.lstat": (1, _CLI_TMPDIR),
    "copilot/sdk/service.py _make_cli_tmpdir os.mkdir": (1, _CLI_TMPDIR),
    "copilot/sdk/service.py _scrub_resumed_memory_for_retry .read_bytes": (
        1,
        _CLI_PROJECTS,
    ),
    "copilot/sdk/service.py _write_cli_session_to_disk .write_bytes": (
        1,
        _CLI_PROJECTS,
    ),
    "copilot/sdk/service.py _write_cli_session_to_disk os.makedirs": (1, _CLI_PROJECTS),
    "copilot/sdk/service.py delete_stale_cli_session_file .unlink": (1, _CLI_PROJECTS),
    "copilot/sdk/service.py read_cli_session_from_disk .read_bytes": (1, _CLI_PROJECTS),
    "copilot/sdk/service.py read_cli_session_from_disk .write_bytes": (
        1,
        _CLI_PROJECTS,
    ),
    "copilot/sdk/service.py stream_chat_completion_sdk os.makedirs": (1, _CREATES_IT),
    "copilot/tools/sandbox.py _build_bwrap_command os.path.exists": (
        1,
        "host system paths bound into the sandbox",
    ),
    "copilot/tools/sandbox.py _build_bwrap_command os.path.islink": (
        1,
        "host system paths bound into the sandbox",
    ),
    "copilot/tools/sandbox.py get_workspace_dir os.makedirs": (1, _CREATES_IT),
    "copilot/tools/skills.py _load_default_body .read_text": (
        1,
        "default skill bodies shipped with the backend",
    ),
    "copilot/transcript.py cleanup_stale_project_dirs .is_dir": (2, _CLI_PROJECTS),
    "copilot/transcript.py cleanup_stale_project_dirs .iterdir": (1, _CLI_PROJECTS),
    "copilot/transcript.py cleanup_stale_project_dirs os.path.isdir": (
        1,
        _CLI_PROJECTS,
    ),
    "copilot/transcript.py cleanup_stale_project_dirs shutil.rmtree": (
        2,
        _CLI_PROJECTS,
    ),
    "copilot/transcript.py cleanup_stale_project_dirs .stat": (2, _CLI_PROJECTS),
    "copilot/transcript.py read_compacted_entries .read_text": (1, _CLI_PROJECTS),
}


def test_workdir_modules_use_host_fs():
    found = _workdir_file_calls()
    counts = collections.Counter(key for key, _ in found)
    new = sorted(
        f"  backend/{key.split()[0]}:{lineno} {key.split()[1]}() calls {key.split()[2]}"
        for key, lineno in found
        if counts[key] > _WORKDIR_FILE_CALL_OFFENDERS.get(key, (0, ""))[0]
    )
    assert not new, (
        "A copilot module that names the chat's working directory opens, checks "
        "and changes files there through backend/copilot/host_fs.py (pass "
        "`opener=open_no_follow` to open()). A call on a path outside that "
        "directory goes in _WORKDIR_FILE_CALL_OFFENDERS with what it acts on:\n"
        + "\n".join(new)
    )


def test_workdir_file_call_offenders_are_not_stale():
    counts = collections.Counter(key for key, _ in _workdir_file_calls())
    stale = sorted(
        f"{key}: {allowed} allowed, {counts[key]} found"
        for key, (allowed, _) in _WORKDIR_FILE_CALL_OFFENDERS.items()
        if counts[key] < allowed
    )
    assert not stale, (
        "These _WORKDIR_FILE_CALL_OFFENDERS counts are above what the modules "
        "still call; lower them (or drop the entry at 0):\n  " + "\n  ".join(stale)
    )


@pytest.mark.parametrize(
    "source, expected",
    [
        ("def f(sdk_cwd):\n    open(sdk_cwd)", ["f open"]),
        ("def f(sdk_cwd):\n    open(sdk_cwd, opener=open_no_follow)", []),
        ("def f(sdk_cwd):\n    open(sdk_cwd, opener=partial(open_no_follow))", []),
        ("def f(sdk_cwd):\n    open(sdk_cwd, opener=other)", ["f open"]),
        ("def f(sdk_cwd):\n    to_thread(shutil.rmtree, sdk_cwd)", ["f shutil.rmtree"]),
        ("def f(sdk_cwd):\n    os.path.exists(sdk_cwd)", ["f os.path.exists"]),
        ("def f(sdk_cwd):\n    Path(sdk_cwd).write_text('x')", ["f .write_text"]),
        ("def f(sdk_cwd):\n    os.path.join(sdk_cwd, 'x')", []),
        ("def f(path):\n    os.remove(path)", []),
        ("def f(sdk_cwd):\n    Path(sdk_cwd).open()", ["f .open"]),
        ("def f(sdk_cwd):\n    Path(sdk_cwd).exists()", ["f .exists"]),
        ("def f(sdk_cwd):\n    Path(sdk_cwd).stat()", ["f .stat"]),
        ("def f(sdk_cwd):\n    Path(sdk_cwd).rename('x')", ["f .rename"]),
        ("def f(sdk_cwd):\n    Path(sdk_cwd).replace('x')", ["f .replace"]),
        ("def f(sdk_cwd):\n    Path(sdk_cwd).readlink()", ["f .readlink"]),
        ("def f(sdk_cwd):\n    Path(sdk_cwd).samefile('x')", ["f .samefile"]),
        ("def f(sdk_cwd):\n    Path(sdk_cwd).glob('*')", ["f .glob"]),
        ("def f(sdk_cwd):\n    glob.glob(sdk_cwd)", ["f glob.glob"]),
        ("def f(sdk_cwd):\n    glob.iglob(sdk_cwd)", ["f glob.iglob"]),
        ("def f(sdk_cwd):\n    codecs.open(sdk_cwd)", ["f .open"]),
        ("def f(sdk_cwd):\n    Image.open(sdk_cwd)", ["f .open"]),
        ("from os import remove\ndef f(sdk_cwd):\n    remove(sdk_cwd)", ["f os.remove"]),
        ("from os import remove as rm\ndef f(sdk_cwd):\n    rm(sdk_cwd)", ["f os.remove"]),
        ("from shutil import rmtree\ndef f(sdk_cwd):\n    to_thread(rmtree, sdk_cwd)", ["f shutil.rmtree"]),
        ("from os.path import exists\ndef f(sdk_cwd):\n    exists(sdk_cwd)", ["f os.path.exists"]),
        ("from glob import glob\ndef f(sdk_cwd):\n    glob(sdk_cwd)", ["f glob.glob"]),
        ("from os import path\ndef f(sdk_cwd):\n    path.isfile(sdk_cwd)", ["f os.path.isfile"]),
        ("import shutil as sh\ndef f(sdk_cwd):\n    sh.copy(sdk_cwd, 'x')", ["f shutil.copy"]),
        ("import os.path as osp\ndef f(sdk_cwd):\n    osp.exists(sdk_cwd)", ["f os.path.exists"]),
        ("from os import path\ndef f(sdk_cwd):\n    path.join(sdk_cwd, 'x')", []),
        ("def f(sdk_cwd):\n    sdk_cwd.replace('/', '_')", []),
        ("def f(sdk_cwd):\n    os.replace(sdk_cwd, 'x')", ["f os.replace"]),
    ],
)  # fmt: skip
def test_workdir_file_call_scan_reads_each_form(source: str, expected: list[str]):
    assert [f"{fn} {call}" for fn, call, _ in _file_calls_in(ast.parse(source))] == (
        expected
    )


def _workdir_file_calls() -> list[tuple[str, int]]:
    found: list[tuple[str, int]] = []
    for py in sorted((BACKEND_ROOT / "copilot").rglob("*.py")):
        rel = py.relative_to(BACKEND_ROOT).as_posix()
        if py.name.endswith("_test.py") or rel == "copilot/host_fs.py":
            continue
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        found += [
            (f"{rel} {fn} {call}", lineno) for fn, call, lineno in _file_calls_in(tree)
        ]
    return found


def _file_calls_in(tree: ast.Module) -> list[tuple[str, str, int]]:
    """By-name file calls in *tree*, if it names the working directory at all."""
    names = {
        getattr(n, "id", None) or getattr(n, "arg", None) or getattr(n, "attr", None)
        for n in ast.walk(tree)
        if isinstance(n, (ast.Name, ast.arg, ast.Attribute))
    }
    if not names & _WORKDIR_NAMES:
        return []
    owner = _enclosing_functions(tree)
    imported = _imported_names(tree)
    calls = []
    for node in ast.walk(tree):
        call = _by_name_file_call(node, imported)
        if call:
            calls.append((owner.get(node, "<module>"), call, node.lineno))
    return sorted(calls, key=lambda c: c[2])


def _by_name_file_call(node: ast.AST, imported: dict[str, str]) -> str | None:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id != "open":
            return None
        opener = next((k.value for k in node.keywords if k.arg == "opener"), None)
        safe = opener is not None and any(
            isinstance(n, ast.Name) and n.id == "open_no_follow"
            for n in ast.walk(opener)
        )
        return None if safe else "open"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        attr = node.func.attr
        if attr in _ONE_ARG_PATH_METHODS and _owner(node.func, imported) is None:
            return f".{attr}" if len(node.args) == 1 else None
        return None
    if isinstance(node, ast.Name):
        full = imported.get(node.id)
        return full if full and full not in _BY_NAME_FILE_CALLS else None
    if not isinstance(node, ast.Attribute):
        return None
    owner = _owner(node, imported)
    if owner in _BY_NAME_FILE_CALLS:
        return (
            f"{owner}.{node.attr}" if node.attr in _BY_NAME_FILE_CALLS[owner] else None
        )
    return f".{node.attr}" if node.attr in _PATH_METHODS else None


def _owner(node: ast.Attribute, imported: dict[str, str]) -> str | None:
    owner = _dotted(node.value)
    owner = imported.get(owner, owner) if owner else None
    return owner if owner in _BY_NAME_FILE_CALLS else None


def _imported_names(tree: ast.Module) -> dict[str, str]:
    """Local name -> the listed module or function it was imported as."""
    names: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname and alias.name in _BY_NAME_FILE_CALLS:
                    names[alias.asname] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                full = f"{node.module}.{alias.name}"
                listed = _BY_NAME_FILE_CALLS.get(node.module, frozenset())
                if full in _BY_NAME_FILE_CALLS or alias.name in listed:
                    names[alias.asname or alias.name] = full
    return names


def _dotted(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return f"{node.value.id}.{node.attr}"
    return None


def _enclosing_functions(tree: ast.Module) -> dict[ast.AST, str]:
    owner: dict[ast.AST, str] = {}
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(fn):
                owner[child] = fn.name  # inner functions overwrite outer ones
    return owner


def test_known_offenders_use_posix_separators():
    """_KNOWN_OFFENDERS must use forward slashes since the comparison key
    is built from pathlib.Path.relative_to() which uses OS-native separators.
    On Windows this would be backslashes, causing false positives.

    Ensure the key construction normalises to forward slashes.
    """
    for entry in (
        _KNOWN_OFFENDERS
        | _DATACLASS_KNOWN_OFFENDERS
        | set(_GLOBAL_CLOCK_PATCH_OFFENDERS)
    ):
        path_part = entry.split()[0]
        assert "\\" not in path_part, (
            f"known-offenders entry uses backslash: {entry!r}. "
            "Use forward slashes — the test should normalise Path separators."
        )


def test_no_process_cached_loop_bound_clients():
    offenders: list[str] = []
    for py in _iter_backend_py_files():
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            decorators = {_decorator_name(d) for d in node.decorator_list}
            if "cached" not in decorators:
                continue
            bound = _annotation_names(node.returns) & LOOP_BOUND_TYPES
            if bound:
                rel = py.relative_to(BACKEND_ROOT)
                key = f"{rel.as_posix()} {node.name}"
                if key in _KNOWN_OFFENDERS:
                    continue
                offenders.append(
                    f"{rel}:{node.lineno} {node.name}() -> {sorted(bound)}"
                )

    assert not offenders, (
        "Process-wide @cached(...) must not wrap functions returning event-"
        "loop-bound async clients. These objects lazily bind their connection "
        "pool to the first event loop that uses them; caching them across "
        "loops poisons the cache and surfaces as opaque connection errors.\n\n"
        "Offenders:\n  " + "\n  ".join(offenders) + "\n\n"
        "Fix: construct the client per-call, or introduce a per-loop factory "
        "keyed on id(asyncio.get_running_loop()). See "
        "backend/util/clients.py::get_openai_client for context."
    )


# ---------------------------------------------------------------------------
# Rule: only cli_io.py puts a line on the Claude Code CLI's stdin
# ---------------------------------------------------------------------------
#
# Motivation: that stream carries the SDK's control protocol as well as the
# conversation, so every send goes through cli_io.py, which writes user turns only.

_CLI_IO_MODULE = "copilot/sdk/cli_io.py"


def _imports_claude_agent_sdk(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name.split(".")[0] == "claude_agent_sdk" for a in node.names):
                return True
        elif (
            isinstance(node, ast.ImportFrom)
            and (node.module or "").split(".")[0] == "claude_agent_sdk"
        ):
            return True
    return False


def test_only_cli_io_writes_to_the_cli_stdin():
    offenders: list[str] = []
    for py in _iter_backend_py_files():
        rel = py.relative_to(BACKEND_ROOT).as_posix()
        if rel == _CLI_IO_MODULE or py.name.endswith("_test.py"):
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        except SyntaxError:
            continue
        if not _imports_claude_agent_sdk(tree):
            continue
        for node in ast.walk(tree):
            # `self._transport` is a module's own attribute; `<client>._transport`
            # reaches into the SDK's.
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "_transport"
                and not (isinstance(node.value, ast.Name) and node.value.id == "self")
            ):
                offenders.append(f"{rel}:{node.lineno} ._transport")
            if (
                isinstance(node, ast.ImportFrom)
                and (node.module or "").split(".")[0] == "claude_agent_sdk"
                and any(a.name == "query" for a in node.names)
            ):
                offenders.append(
                    f"{rel}:{node.lineno} import of claude_agent_sdk.query"
                )
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute):
                name = func.attr
            elif isinstance(func, ast.Name):
                name = func.id
            else:
                continue
            # Every SDK entry point that writes a prompt: the client's query()
            # and connect(prompt), the module-level query(), and a transport's write().
            if name in ("query", "stream_input"):
                offenders.append(f"{rel}:{node.lineno} {name}(...)")
            elif name == "connect" and (node.args or node.keywords):
                offenders.append(f"{rel}:{node.lineno} connect(<prompt>)")
            elif (
                name == "write"
                and isinstance(func, ast.Attribute)
                and "transport" in ast.unparse(func.value).lower()
            ):
                offenders.append(f"{rel}:{node.lineno} <transport>.write(...)")

    assert not offenders, (
        "Only backend/copilot/sdk/cli_io.py may send a message to the Claude "
        "Code CLI; it writes user turns only.\n\n"
        "Offenders:\n  " + "\n  ".join(offenders) + "\n\n"
        "Fix: call cli_io.send_user_text() or cli_io.send_user_blocks()."
    )
