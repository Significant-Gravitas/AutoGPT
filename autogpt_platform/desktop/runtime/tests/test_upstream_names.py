"""Upstream names the service host, the profiles and the run settling use,
read from the backend's source without importing it.

These run wherever the unit tests run, including the gate that an upstream
sync has to pass, so a rename upstream fails here with what to update.
backend_contract.py checks the same dependencies, and the ones that need the
backend imported, against the backend in a bundle.
"""

import ast
import re
from pathlib import Path

import pytest

from autogpt_desktop import apps, ports, resources, runs, settings

PLATFORM = Path(__file__).resolve().parents[3]
BACKEND = PLATFORM / "backend" / "backend"
HOST = "desktop/runtime/autogpt_desktop/servicehost.py"


def source(*parts: str) -> str:
    return BACKEND.joinpath(*parts).read_text(encoding="utf-8")


def entry_module(service: apps.Service) -> Path:
    module = service.entry.partition(":")[0]
    root = (
        PLATFORM / "desktop" / "runtime"
        if module.startswith("autogpt_desktop.")
        else BACKEND.parent
    )
    return root.joinpath(*module.split(".")).with_suffix(".py")


def entry_tree(service: apps.Service) -> ast.AST:
    tree = ast.parse(entry_module(service).read_text(encoding="utf-8"))
    name = service.entry.partition(":")[2]
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def classes_handed_over(tree: ast.AST) -> set[str]:
    """The classes instantiated inside a call to run_processes."""
    return {
        node.func.id
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and getattr(call.func, "id", "") == "run_processes"
        for node in ast.walk(call)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    } - {"run_processes"}


def test_backend_app_still_has_the_seam_the_services_are_taken_from():
    assert re.search(r"^def run_processes\(\*processes", source("app.py"), re.M), (
        f"backend.app.run_processes(*processes) is gone or changed. {HOST} puts a collector "
        "in its place to be handed each service; find where an entry point hands its service "
        "over now and collect it there."
    )


@pytest.mark.parametrize("service", apps.SERVICES, ids=lambda service: service.name)
def test_each_entry_point_hands_one_service_to_run_processes(service: apps.Service):
    path = entry_module(service)
    update = f"Update {service.name}'s entry in SERVICES in desktop/runtime/autogpt_desktop/apps.py."
    assert path.is_file(), f"{service.entry} no longer exists upstream. {update}"
    text = path.read_text(encoding="utf-8")
    assert "from backend.app import run_processes" in text, (
        f"{path.name} no longer imports run_processes from backend.app, so the collector in "
        f"{HOST} never sees its service."
    )
    assert f"def {service.entry.partition(':')[2]}(" in text, update
    handed = classes_handed_over(entry_tree(service))
    assert (
        len(handed) == 1
    ), f"{path.name} hands over {sorted(handed)}; the host expects one."


def test_the_desktop_runs_every_service_upstream_runs_but_the_chat_bridges():
    upstream = classes_handed_over(ast.parse(source("app.py")))
    ours = set()
    for service in apps.SERVICES:
        ours |= classes_handed_over(entry_tree(service))
    assert upstream - {"PlatformLinkingManager", "CoPilotChatBridge"} == ours, (
        f"backend.app.main runs {sorted(upstream)}; the desktop runs {sorted(ours)}. Add a new "
        "service to SERVICES and to a group in apps.py (and its port to ports.py), or remove "
        "one that is gone."
    )


def test_database_adapters_preserve_upstream_service_construction():
    def arguments(tree):
        return sorted(
            ast.dump(argument)
            for call in ast.walk(tree)
            if isinstance(call, ast.Call)
            and getattr(call.func, "id", "") == "run_processes"
            for argument in call.args
        )

    expected = arguments(ast.parse(source("db.py")))
    actual = [
        argument
        for name in ("database-manager", "swap-credentials")
        for argument in arguments(entry_tree(apps.service(name)))
    ]
    assert sorted(actual) == expected


@pytest.mark.parametrize(
    "name,module", [("websocket", "ws_api.py"), ("rest", "rest_api.py")]
)
def test_api_database_ownership_matches_upstream(name: str, module: str):
    tree = ast.parse(source("api", module))
    connects = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "connect"
        and isinstance(node.func.value, (ast.Name, ast.Attribute))
        and (
            node.func.value.id
            if isinstance(node.func.value, ast.Name)
            else node.func.value.attr
        )
        in {"db", "database", "prisma"}
        for node in ast.walk(tree)
    )
    assert apps.service(name).database == connects, (
        f"{name}'s database ownership changed in {module}. "
        "Update SERVICES and keep direct-database services in separate hosts "
        "from services that query through database-manager."
    )


def test_a_service_can_still_be_run_in_a_thread_and_cleaned_up():
    process = source("util", "process.py")
    update = f"Update run_service, clean_up and check_service in {HOST}."
    assert "class AppProcess(" in process, update
    assert re.search(
        r"def start\(self, background: bool = False", process
    ), f"AppProcess.start(background=False) changed. {update}"
    assert re.search(
        r"^    def cleanup\(self\):", process, re.M
    ), f"AppProcess.cleanup() changed. {update}"
    assert re.search(r"@property\s+def service_name\(self\)", process), update
    # Why the host makes signal.signal thread-aware. If this goes, so can that.
    assert "signal.signal(signal.SIGTERM, self._self_terminate)" in process, (
        "AppProcess no longer installs signal handlers in execute_run_command; "
        f"make_signals_thread_aware in {HOST} may no longer be needed."
    )


def test_the_database_switch_the_watchdog_reads_is_still_there():
    assert re.search(r"^def is_connected\(\)", source("data", "db.py"), re.M), (
        f"backend.data.db.is_connected() is gone. The watchdog in {HOST} (database_is_connected) "
        "uses it to notice a worker service connecting to the database itself."
    )


@pytest.mark.parametrize("module", ["rest_api.py", "ws_api.py"])
def test_the_api_servers_still_serve_through_uvicorn_run(module: str):
    assert "uvicorn.run(" in source("api", module), (
        f"api/{module} no longer calls uvicorn.run. {HOST} replaces that call to serve both API "
        "servers on one event loop (serve_on_one_loop); without it the second server never "
        "finishes starting. Intercept what is called now, or split the api group in apps.py."
    )


def test_services_answer_where_the_health_table_says():
    update = "Update the health paths in SERVICES in desktop/runtime/autogpt_desktop/apps.py."
    assert '"/health_check"' in source("util", "service.py"), update
    assert '@app.get("/health")' in source("api", "ws_api.py"), update
    assert '@app.get(path="/health"' in source("api", "rest_api.py"), update
    for manager in (("executor", "manager.py"), ("copilot", "executor", "manager.py")):
        assert "start_http_server(" in source(*manager), update


# Service -> the backend setting its server's port is read from, and the
# module that reads it. The four AppService services say theirs through
# get_port, which backend_contract.py asks.
PORT_SETTINGS = {
    "swap-credentials": (
        "swap_credential_service_port",
        ("copilot", "swap_service.py"),
    ),
    "executor": ("execution_manager_port", ("executor", "manager.py")),
    "copilot-executor": (
        "copilot_executor_port",
        ("copilot", "executor", "manager.py"),
    ),
    "websocket": ("websocket_server_port", ("api", "ws_api.py")),
    "rest": ("agent_api_port", ("api", "rest_api.py")),
}


@pytest.mark.parametrize("name", PORT_SETTINGS)
def test_a_service_still_listens_on_the_port_the_runtime_gives_it(name: str):
    """The supervisor waits for each service on that port; the port numbers
    differ per install, so a renamed setting is a service that never answers."""
    setting, module = PORT_SETTINGS[name]
    update = (
        f"Update the *_PORT names in settings._service_addresses and {name}'s entry in "
        "SERVICES in desktop/runtime/autogpt_desktop/apps.py."
    )
    assert re.search(
        rf"^    {setting}: int = Field\(", source("util", "settings.py"), re.M
    ), f"backend.util.settings.Config has no {setting} any more. {update}"
    assert f"config.{setting}" in source(*module) or f"Config().{setting}" in source(
        *module
    ), f"{'/'.join(module)} no longer serves on {setting}. {update}"
    port = {name: 20000 + index for index, name in enumerate(ports.PORT_NAMES)}
    given = settings._service_addresses(port)
    assert given[setting.upper()] == str(port[apps.service(name).port])


def test_the_scheduler_still_holds_the_connections_the_limit_counts():
    scheduler = source("executor", "scheduler.py")
    update = (
        "Count the scheduler's connections again and update SCHEDULER_POOLS and "
        "POSTGRES_LIMITS in desktop/runtime/autogpt_desktop/resources.py."
    )
    assert scheduler.count("create_engine(") == resources.SCHEDULER_POOLS, update
    assert (
        scheduler.count("pool_size=self.db_pool_size()") == resources.SCHEDULER_POOLS
    ), update
    assert scheduler.count("max_overflow=0") == resources.SCHEDULER_POOLS, update
    assert "return config.scheduler_db_pool_size" in scheduler, update
    default = re.search(
        r"^    scheduler_db_pool_size: int = Field\(\s+default=(\d+)",
        source("util", "settings.py"),
        re.M,
    )
    assert (
        default
    ), f"backend.util.settings.Config has no scheduler_db_pool_size any more. {update}"
    assert int(default.group(1)) == resources.SCHEDULER_DB_POOL_SIZE, (
        f"Upstream's scheduler pool is now {default.group(1)}; the desktop pins "
        f"{resources.SCHEDULER_DB_POOL_SIZE}. Follow it if the connection limit has room. {update}"
    )


def test_the_pool_sizes_a_profile_sets_are_backend_settings():
    config = source("util", "settings.py")
    for profile in resources.PROFILES:
        for name in resources.choose(
            {resources.PROFILE_SETTING: profile}, 1, 1
        ).backend_env:
            assert re.search(rf"^    {name.lower()}: int = Field\(", config, re.M), (
                f"backend.util.settings.Config has no {name.lower()} any more. "
                "Update desktop/runtime/autogpt_desktop/resources.py."
            )


def test_the_lock_keys_cleared_before_an_executor_starts_are_the_executors():
    update = "Update LOCK_PATTERNS in desktop/runtime/autogpt_desktop/runs.py."
    assert 'key=f"exec_lock:{graph_exec_id}"' in source(
        "executor", "manager.py"
    ), update
    assert runs.LOCK_PATTERNS["executor"] == "exec_lock:*"
    assert 'return f"copilot:session:{session_id}:lock"' in source(
        "copilot", "executor", "utils.py"
    ), update
    assert runs.LOCK_PATTERNS["copilot-executor"] == "copilot:session:*:lock"
    assert set(runs.LOCK_PATTERNS) <= {service.name for service in apps.SERVICES}


RUNS = "desktop/runtime/autogpt_desktop/runs.py"


@pytest.mark.parametrize(
    "module, constant, queue, exchange",
    [
        (
            ("executor", "utils.py"),
            "GRAPH_EXECUTION",
            runs.GRAPH_RUN_QUEUE,
            runs.GRAPH_CANCEL_EXCHANGE,
        ),
        (
            ("copilot", "executor", "utils.py"),
            "COPILOT",
            runs.COPILOT_TURN_QUEUE,
            runs.COPILOT_CANCEL_EXCHANGE,
        ),
    ],
    ids=["agent runs", "AutoPilot turns"],
)
def test_the_queue_and_the_cancel_exchange_are_named_as_the_runtime_names_them(
    module: tuple[str, ...], constant: str, queue: str, exchange: str
):
    """The runtime asks the executors to cancel what the user chose to stop,
    and takes a stopped turn's message off its queue."""
    text = source(*module)
    update = f"Update the names at the top of {RUNS}."
    queue_name = f"{constant.removesuffix('_EXECUTION')}_EXECUTION_QUEUE_NAME"
    assert (
        f'{queue_name} = "{queue}"' in text
    ), f"{queue_name} is no longer {queue!r}. {update}"
    declared = re.search(
        rf'{constant.removesuffix("_EXECUTION")}(?:_EXECUTION)?_CANCEL_EXCHANGE = Exchange\(\s+'
        r'name="([^"]+)",\s+type=ExchangeType\.(\w+),',
        text,
    )
    assert (
        declared
    ), f"The exchange for requests to cancel is declared differently now. {update}"
    assert declared.groups() == (exchange, "FANOUT"), (
        f"A request to cancel now goes to the {declared.group(2)} exchange {declared.group(1)!r}. "
        f"{update} Queue.cancel publishes it with an empty routing key."
    )


def model_fields(module: tuple[str, ...], model: str) -> dict[str, bool]:
    """A pydantic model's fields, read from its source: name -> required."""
    found = re.search(
        rf"^class {model}\(BaseModel\):\n(.*?)(?=^\S|\Z)",
        source(*module),
        re.DOTALL | re.MULTILINE,
    )
    assert found, f"{model} is gone from backend/{'/'.join(module)}. Update {RUNS}."
    fields = re.findall(r"^    (\w+): [^=\n]+?( = .*)?$", found.group(1), re.MULTILINE)
    return {name: not default for name, default in fields}


@pytest.mark.parametrize(
    "module, model, name",
    [
        (("executor", "utils.py"), "CancelExecutionEvent", runs.GRAPH_CANCEL_FIELD),
        (
            ("copilot", "executor", "utils.py"),
            "CancelCoPilotEvent",
            runs.COPILOT_SESSION_FIELD,
        ),
    ],
)
def test_a_request_to_cancel_needs_only_the_field_the_runtime_gives_it(
    module: tuple[str, ...], model: str, name: str
):
    fields = model_fields(module, model)
    assert [field for field, required in fields.items() if required] == [name], (
        f"{model} now needs {sorted(fields)}; the runtime's request gives it {name!r} alone. "
        f"Update Queue.cancel in {RUNS}."
    )


def test_a_turns_message_still_names_its_session():
    fields = model_fields(("copilot", "executor", "utils.py"), "CoPilotExecutionEntry")
    assert runs.COPILOT_SESSION_FIELD in fields, (
        f"CoPilotExecutionEntry has no {runs.COPILOT_SESSION_FIELD} any more. Queue.drop_turns "
        f"in {RUNS} finds the turns of a stopped session by it."
    )
    assert (
        "turn_id" in fields
    ), f"Queue.drop_turns in {RUNS} guards against removing a later turn."


def test_the_executors_still_take_a_cancel_request_as_json_and_stop_the_run():
    for module, model in (
        (("executor", "manager.py"), "CancelExecutionEvent"),
        (("copilot", "executor", "manager.py"), "CancelCoPilotEvent"),
    ):
        text = source(*module)
        assert f"request = {model}.model_validate_json(body)" in text, (
            f"backend/{'/'.join(module)} reads a request to cancel differently now. "
            f"Update Queue.cancel in {RUNS}."
        )
        assert "cancel_event.set()" in text


def test_the_executor_still_skips_the_message_of_a_run_that_has_ended():
    """How a run ended in the database stays ended: stopped by the user as
    the app closed, older than a day, or left with nothing to resume it."""
    assert re.search(
        r'else:\s+log_metadata\.warning\(\s+f"Skipped graph execution',
        source("executor", "manager.py"),
    ), (
        "on_graph_execution in backend/executor/manager.py no longer skips a run whose status "
        f"is not QUEUED, RUNNING or REVIEW. {RUNS} ends runs in the database and relies on "
        "the executor leaving them ended; without it a run the user stopped starts again."
    )


def test_a_run_is_locked_for_as_long_as_it_runs():
    """The locks are how the runtime counts what is in progress."""
    manager = source("executor", "manager.py")
    assert "cluster_lock.refresh()" in manager, (
        "The executor no longer renews a run's lock while the run goes on, so a long run "
        f"would stop counting as in progress. See Cache.active in {RUNS}."
    )
    assert "self._execution_locks[graph_exec_id].release()" in manager, (
        f"The executor releases a run's lock differently now. See Cache.active in {RUNS}: a run "
        "that has ended must not count as in progress."
    )


def test_a_turn_is_locked_for_as_long_as_it_runs():
    """Locks supplement session metadata when counting AutoPilot work."""
    assert "cluster_lock.refresh()" in source("copilot", "executor", "processor.py"), (
        "The copilot executor no longer renews a turn's lock while the turn goes on "
        "(backend/copilot/executor/processor.py), so a turn of more than five minutes would "
        f"stop counting as in progress and be left out of a stop. See Cache.active in {RUNS}."
    )
    assert re.search(
        r"def on_run_done\(f: Future\):.*?finally:\s+if session_id in self\._task_locks:\s+"
        r"(?:logger\.\w+\([^\n]*\)\s+)?self\._task_locks\[session_id\]\.release\(\)",
        source("copilot", "executor", "manager.py"),
        re.DOTALL,
    ), (
        "The copilot executor releases a turn's lock differently now "
        "(on_run_done in backend/copilot/executor/manager.py). A turn that has ended must not "
        f"count as in progress: see Cache.active in {RUNS}."
    )


def function_source(text: str, name: str) -> str:
    found = re.search(
        rf"^(?:async )?def {name}\(.*?(?=^(?:async def|def|class|@)\b|\Z)",
        text,
        re.DOTALL | re.MULTILINE,
    )
    return found.group(0) if found else ""


def test_closing_a_chat_changes_only_one_that_is_busy():
    """Completion must be guarded by both running status and turn identity."""
    closing = function_source(
        source("copilot", "stream_registry.py"), "mark_session_completed"
    )
    assert re.search(
        r'hash_compare_and_set\(\s*redis,\s*meta_key,\s*"status",\s*'
        r'expected="running",\s*new=status',
        closing,
    ), (
        "stream_registry.mark_session_completed no longer swaps a chat's status only from "
        '"running" (the hash_compare_and_set call with expected="running"). runs.reconcile '
        f"in {RUNS} lists every recorded turn's chat for {HOST} (close_ended_turns) to close, "
        "finished or not: if closing now marks a finished chat failed, list only the chats "
        "whose message was found on the queue."
    )
    assert re.search(
        r"if not swapped:\s+(?:logger\.\w+\([^\n]*\)\s+)?return False", closing
    ), (
        "stream_registry.mark_session_completed goes on after finding the chat not busy. "
        f"See close_ended_turns in {HOST} and runs.reconcile in {RUNS}."
    )
    assert 'guard = ("turn_id", turn_id) if turn_id else None' in closing
    assert re.search(r'expected="running",\s*new=status,\s*guard=guard', closing)


def test_running_session_metadata_has_the_identity_used_for_sleep_recovery():
    registry = source("copilot", "stream_registry.py")
    config = source("copilot", "config.py")
    assert re.search(
        rf'session_meta_prefix: str = Field\(\s+default="{re.escape(runs.SESSION_META_PREFIX)}"',
        config,
    )
    assert 'return f"{config.session_meta_prefix}{{{session_id}}}"' in registry
    assert 'status: Literal["running", "completed", "failed"] = "running"' in registry
    creating = function_source(registry, "create_session")
    assert '"turn_id": turn_id' in creating
    assert '"status": session.status' in creating


def test_a_chat_can_still_be_closed_the_way_upstreams_stop_closes_it():
    """The copilot executor's host does it for a turn that was ended without
    the executor; until then the chat takes no new message."""
    update = f"Update close_ended_turns in {HOST}."
    registry = source("copilot", "stream_registry.py")
    closing = next(
        node
        for node in ast.parse(registry).body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "mark_session_completed"
    )
    arguments = closing.args
    required = [
        arg.arg
        for arg in arguments.args[: len(arguments.args) - len(arguments.defaults)]
    ]
    required += [
        arg.arg
        for arg, default in zip(arguments.kwonlyargs, arguments.kw_defaults)
        if default is None
    ]
    assert required == ["session_id"], update
    assert {"error_message", "skip_error_publish", "turn_id"} <= {
        arg.arg for arg in [*arguments.args, *arguments.kwonlyargs]
    }, update
    assert re.search(
        r'^CANCELLED_MESSAGE = "', registry, re.MULTILINE
    ), f"stream_registry.CANCELLED_MESSAGE is gone. {update}"
    assert re.search(
        r"await stream_registry\.mark_session_completed\(\s+session_id,\s+"
        r"error_message=stream_registry\.CANCELLED_MESSAGE,\s+skip_error_publish=True,",
        source("api", "features", "chat", "routes.py"),
    ), (
        "Upstream's Stop no longer closes a chat this way when the executor does not confirm "
        f"the cancel. The host copies that call: find what it does now. {update}"
    )


def prisma_model(schema: str, name: str) -> str:
    found = re.search(
        rf"^model {name} \{{\n(.*?)^\}}", schema, re.DOTALL | re.MULTILINE
    )
    return found.group(1) if found else ""


@pytest.mark.parametrize(
    "table, columns",
    [
        (runs.GRAPH_EXECUTION_TABLE, runs.GRAPH_EXECUTION_COLUMNS),
        (runs.NODE_EXECUTION_TABLE, runs.NODE_EXECUTION_COLUMNS),
    ],
)
def test_the_run_tables_still_have_the_columns_settling_uses(
    table: str, columns: tuple[str, ...]
):
    schema = (PLATFORM / "backend" / "schema.prisma").read_text(encoding="utf-8")
    model = prisma_model(schema, table)
    update = "Update the names and the SQL in desktop/runtime/autogpt_desktop/runs.py."
    assert model, f"backend/schema.prisma has no model {table} any more. {update}"
    assert (
        "@@map(" not in model
    ), f"{table} is now stored under another table name. {update}"
    fields = set(re.findall(r"^\s+(\w+)\s+\w", model, re.MULTILINE))
    missing = set(columns) - fields
    assert not missing, f"{table} no longer has {sorted(missing)}. {update}"
    for column in columns:
        assert column in runs.END_NODES + runs.END_RUNS + runs.UNFINISHED_AND_OLD
    for statement in (runs.UNFINISHED_AND_OLD, runs.UNFINISHED_BEFORE, runs.END_RUNS):
        assert f'platform."{runs.GRAPH_EXECUTION_TABLE}"' in statement


def test_the_statuses_settling_writes_are_execution_statuses():
    schema = (PLATFORM / "backend" / "schema.prisma").read_text(encoding="utf-8")
    statuses = re.search(
        r"^enum AgentExecutionStatus \{\n(.*?)^\}", schema, re.DOTALL | re.MULTILINE
    )
    assert statuses, "backend/schema.prisma has no AgentExecutionStatus enum any more."
    missing = set(runs.EXECUTION_STATUSES) - set(statuses.group(1).split())
    assert not missing, f"AgentExecutionStatus lost {sorted(missing)}; update runs.py."


def test_the_skills_catalog_publisher_is_still_where_the_runtime_calls_it():
    module, _, function = apps.SKILLS_CATALOG_ENTRY.partition(":")
    text = source(
        *module.removeprefix("backend.").split(".")[:-1], module.split(".")[-1] + ".py"
    )
    assert f"def {function}(" in text
    assert (
        "--skip-missing-preloads" in text
    ), "supervisor.publish_skills_catalog passes this flag."
