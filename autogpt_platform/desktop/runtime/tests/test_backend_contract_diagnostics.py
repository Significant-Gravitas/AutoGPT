"""Contract diagnostics without importing or starting the backend."""

import builtins
import importlib.util
import io
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

PREFIX = "DESKTOP_SMOKE_DIAGNOSTIC "
CANARIES = (
    "secret-contract-canary-7f19",
    r"C:\private-contract-canary\owner\settings.json",
    "DATABASE_URL=postgresql://contract-canary:private@invalid/db",
)
PRIVATE = " | ".join(CANARIES)
CONTRACT_NAMES = (
    "service_names_and_classes",
    "nobody_is_connected",
    "api_servers_call_uvicorn_run",
    "services_answer_where_the_health_table_says",
    "the_other_services_listen_on_the_ports_they_are_given",
    "uvicorn_can_serve_on_the_hosts_loop",
    "pool_sizes_come_from_the_environment",
    "the_scheduler_holds_the_connections_the_limit_counts",
    "lock_keys_are_the_ones_cleared",
    "runs_are_cancelled_and_settled_under_the_runtimes_names",
    "a_chat_can_be_closed_from_outside_its_turn",
    "turns_are_locked_while_they_run_and_only_busy_chats_are_closed",
)
EXCEPTIONS = (
    (ModuleNotFoundError, "module_not_found"),
    (ImportError, "import_error"),
    (AttributeError, "attribute_error"),
    (TypeError, "type_error"),
    (ValueError, "value_error"),
    (OSError, "os_error"),
    (TimeoutError, "timeout"),
    (RuntimeError, "other"),
    (AssertionError, "other"),
)


@pytest.fixture
def contract(monkeypatch, tmp_path):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name.split(".")[0] not in {"backend", "autogpt_desktop"}
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    spec = importlib.util.spec_from_file_location(
        "backend_contract_diagnostics_subject",
        Path(__file__).with_name("backend_contract.py"),
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.original_collect = module.collect
    monkeypatch.setattr(module.sys, "argv", ["backend_contract.py", str(tmp_path)])
    monkeypatch.setattr(
        module.tempfile,
        "TemporaryDirectory",
        lambda **kwargs: nullcontext(str(tmp_path / "scratch")),
    )

    def collect(bundle_root, scratch):
        assert bundle_root == tmp_path
        assert scratch == tmp_path / "scratch"
        module.expect(True, PRIVATE, "")
        return {"service": object()}

    monkeypatch.setattr(module, "collect", collect)
    monkeypatch.setattr(module, "CHECKS", ())
    return module


def events(output):
    result = []
    for line in output.splitlines():
        if not line.startswith(PREFIX):
            continue
        assert all(canary not in line for canary in CANARIES)
        event = json.loads(line[len(PREFIX) :])
        assert set(event) == {
            "version",
            "stage",
            "status",
            "reason",
            "exception",
            "contract",
            "child_exit",
            "failures",
            "start",
        }
        assert type(event["version"]) is int and event["version"] == 1
        assert event["stage"] in {"contract_collect", "contract_check"}
        assert event["status"] in {"started", "passed", "failed", "error"}
        assert event["reason"] in {"none", "assertion_failed", "exception", "timeout"}
        assert event["exception"] in {
            "none",
            "module_not_found",
            "import_error",
            "attribute_error",
            "type_error",
            "value_error",
            "os_error",
            "timeout",
            "keyboard_interrupt",
            "system_exit",
            "other",
        }
        assert event["contract"] in {"collect", "unknown", *CONTRACT_NAMES}
        assert event["child_exit"] is None
        assert type(event["failures"]) is int and 0 <= event["failures"] <= 65535
        assert type(event["start"]) is int and event["start"] == 0
        result.append(event)
    assert result, "no safe contract diagnostics emitted"
    return result


def test_all_named_checks_pass_with_flushed_diagnostics(
    contract, monkeypatch, capsys, tmp_path
):
    checks = []
    ids = {}
    for name in CONTRACT_NAMES:

        def check(services, backend):
            assert set(services) == {"service"}
            assert backend == tmp_path / "backend" / "backend"
            contract.expect(True, PRIVATE, "")

        checks.append(check)
        ids[check] = name
    monkeypatch.setattr(contract, "CHECKS", checks)
    monkeypatch.setattr(contract, "_CONTRACT_IDS", ids, raising=False)
    printed = []

    def print_spy(*args, **kwargs):
        printed.append((args, kwargs))
        builtins.print(*args, **kwargs)

    monkeypatch.setattr(contract, "print", print_spy, raising=False)
    assert contract.main() == 0
    output = capsys.readouterr().out
    diagnostics = events(output)
    assert [
        (event["stage"], event["contract"], event["status"]) for event in diagnostics
    ] == [
        (stage, name, status)
        for stage, name in [("contract_collect", "collect")]
        + [("contract_check", name) for name in CONTRACT_NAMES]
        for status in ("started", "passed")
    ]
    assert all(event["failures"] == 0 for event in diagnostics)
    assert all(event["reason"] == event["exception"] == "none" for event in diagnostics)
    diagnostic_prints = [
        kwargs for args, kwargs in printed if args and args[0].startswith(PREFIX)
    ]
    assert len(diagnostic_prints) == 26
    assert all(kwargs.get("flush") is True for kwargs in diagnostic_prints)
    assert output.count(f"ok: {PRIVATE}") == 13
    assert output.endswith("13 of 13 dependencies hold\n")


@pytest.mark.parametrize("services_present", [False, True])
@pytest.mark.parametrize("preexisting", [False, True])
def test_collection_failure_never_becomes_a_pass(
    contract, monkeypatch, capsys, services_present, preexisting
):
    if preexisting:
        contract.expect(False, "collect", PRIVATE)

    def collect(bundle_root, scratch):
        if not preexisting:
            contract.expect(False, "collect", PRIVATE)
        return {"service": object()} if services_present else {}

    def forbidden_check(services, backend):
        pytest.fail("checks ran after empty collection")

    monkeypatch.setattr(contract, "collect", collect)
    if not services_present:
        monkeypatch.setattr(contract, "CHECKS", (forbidden_check,))
    assert contract.main() == 1
    output = capsys.readouterr().out
    diagnostics = events(output)
    assert [(event["stage"], event["status"]) for event in diagnostics] == [
        ("contract_collect", "started"),
        ("contract_collect", "failed"),
    ]
    assert diagnostics[-1]["reason"] == "assertion_failed"
    assert diagnostics[-1]["exception"] == "none"
    assert diagnostics[-1]["failures"] == 1
    assert contract.findings == [(False, f"collect. {PRIVATE}")]
    assert f"BROKEN: collect. {PRIVATE}" in output
    assert output.endswith("0 of 1 dependencies hold\n")


def test_failed_check_and_later_pass_have_separate_counts(
    contract, monkeypatch, capsys
):
    def failing(services, backend):
        contract.expect(False, "first", PRIVATE)
        contract.expect(True, PRIVATE, "")
        contract.expect(False, "second", PRIVATE)

    def passing(services, backend):
        contract.expect(True, PRIVATE, "")

    monkeypatch.setattr(contract, "CHECKS", (failing, passing))
    assert contract.main() == 1
    output = capsys.readouterr().out
    diagnostics = events(output)
    assert [event["status"] for event in diagnostics] == [
        "started",
        "passed",
        "started",
        "failed",
        "started",
        "passed",
    ]
    assert diagnostics[3]["contract"] == "unknown"
    assert diagnostics[3]["reason"] == "assertion_failed"
    assert diagnostics[3]["exception"] == "none"
    assert diagnostics[3]["failures"] == 2
    assert diagnostics[-1]["failures"] == 0
    assert output.endswith("3 of 5 dependencies hold\n")
    assert [message for ok, message in contract.findings if not ok] == [
        f"first. {PRIVATE}",
        f"second. {PRIVATE}",
    ]


@pytest.mark.parametrize("claimed_name", [CONTRACT_NAMES[0], PRIVATE])
def test_replacement_check_cannot_supply_its_own_contract_id(
    contract, monkeypatch, capsys, claimed_name
):
    def replacement(services, backend):
        contract.expect(True, PRIVATE, "")

    replacement.__name__ = claimed_name
    monkeypatch.setattr(contract, "CHECKS", (replacement,))
    assert contract.main() == 0
    diagnostics = events(capsys.readouterr().out)
    assert [event["contract"] for event in diagnostics] == [
        "collect",
        "collect",
        "unknown",
        "unknown",
    ]


def test_original_callable_name_is_frozen_at_import(contract, monkeypatch, capsys):
    original = contract.service_names_and_classes

    def harmless(services, backend):
        pass

    monkeypatch.setattr(original, "__code__", harmless.__code__)
    monkeypatch.setattr(original, "__name__", PRIVATE)
    monkeypatch.setattr(contract, "CHECKS", (original,))
    assert contract.main() == 0
    diagnostics = events(capsys.readouterr().out)
    assert [event["contract"] for event in diagnostics[2:]] == [
        CONTRACT_NAMES[0],
        CONTRACT_NAMES[0],
    ]


@pytest.mark.parametrize("exception_type, category", EXCEPTIONS)
@pytest.mark.parametrize("stage", ["contract_collect", "contract_check"])
def test_exception_categories_do_not_expose_names_messages_or_findings(
    contract, monkeypatch, capsys, exception_type, category, stage
):
    error = type("private_exception_canary", (exception_type,), {})(PRIVATE)
    calls = []

    def failing(*args):
        if stage == "contract_check":
            contract.expect(False, "before exception", PRIVATE)
        raise error

    failing.__name__ = "private_check_canary"

    def later(*args):
        calls.append(True)

    if stage == "contract_collect":
        monkeypatch.setattr(contract, "collect", failing)
        with pytest.raises(exception_type) as raised:
            contract.main()
        assert raised.value is error
        assert contract.findings == []
    else:
        monkeypatch.setattr(contract, "CHECKS", (failing, later))
        assert contract.main() == 1
        assert calls == [True]
        assert contract.findings[-1] == (
            False,
            "private check canary. The check itself failed: "
            f"private_exception_canary: {PRIVATE}",
        )
    output = capsys.readouterr().out
    diagnostics = events(output)
    errors = [event for event in diagnostics if event["status"] == "error"]
    assert len(errors) == 1
    event = errors[0]
    assert event["stage"] == stage
    assert event["contract"] == (
        "collect" if stage == "contract_collect" else "unknown"
    )
    assert event["exception"] == category
    assert event["reason"] == ("timeout" if category == "timeout" else "exception")
    assert event["failures"] == (0 if stage == "contract_collect" else 2)
    assert all(
        "private_exception_canary" not in json.dumps(item) for item in diagnostics
    )
    assert all("private_check_canary" not in json.dumps(item) for item in diagnostics)
    if stage == "contract_collect":
        assert [item["status"] for item in diagnostics] == ["started", "error"]
    else:
        assert PRIVATE in output
        assert diagnostics[-1]["status"] == "passed"


@pytest.mark.parametrize(
    "exception_type, category",
    [
        (KeyboardInterrupt, "keyboard_interrupt"),
        (SystemExit, "system_exit"),
        (BaseException, "other"),
    ],
)
@pytest.mark.parametrize("stage", ["contract_collect", "contract_check"])
def test_base_exceptions_emit_an_error_then_reraise_unchanged(
    contract, monkeypatch, capsys, exception_type, category, stage
):
    error = exception_type(PRIVATE)

    def failing(*args):
        contract.expect(False, "before interruption", PRIVATE)
        raise error

    def forbidden(*args):
        pytest.fail("continued after interruption")

    if stage == "contract_collect":
        monkeypatch.setattr(contract, "collect", failing)
        monkeypatch.setattr(contract, "CHECKS", (forbidden,))
    else:
        monkeypatch.setattr(contract, "CHECKS", (failing, forbidden))
    with pytest.raises(exception_type) as raised:
        contract.main()
    assert raised.value is error
    diagnostics = events(capsys.readouterr().out)
    assert diagnostics[-1]["stage"] == stage
    assert diagnostics[-1]["status"] == "error"
    assert diagnostics[-1]["exception"] == category
    assert diagnostics[-1]["reason"] == "exception"
    assert diagnostics[-1]["failures"] == 1
    assert contract.findings[-1] == (False, f"before interruption. {PRIVATE}")


@pytest.mark.parametrize("point", ["bundle_path", "scratch_create", "scratch_enter"])
def test_setup_exceptions_are_collection_errors(contract, monkeypatch, capsys, point):
    error = OSError(PRIVATE)

    def fail(*args, **kwargs):
        raise error

    class FailingScratch:
        __enter__ = fail

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(contract, "collect", lambda *args: pytest.fail("collected"))
    if point == "bundle_path":
        monkeypatch.setattr(contract.sys, "argv", ["backend_contract.py", object()])
        expected_exception = TypeError
        category = "type_error"
    else:
        monkeypatch.setattr(
            contract.tempfile,
            "TemporaryDirectory",
            fail if point == "scratch_create" else lambda **kwargs: FailingScratch(),
        )
        expected_exception = OSError
        category = "os_error"
    with pytest.raises(expected_exception) as raised:
        contract.main()
    if point != "bundle_path":
        assert raised.value is error
    diagnostics = events(capsys.readouterr().out)
    assert [event["status"] for event in diagnostics] == ["started", "error"]
    assert all(event["stage"] == "contract_collect" for event in diagnostics)
    assert diagnostics[-1]["exception"] == category
    assert diagnostics[-1]["reason"] == "exception"
    assert diagnostics[-1]["failures"] == 0


@pytest.mark.parametrize(
    "count, bounded", [(-1, 0), (0, 0), (65535, 65535), (65536, 65535)]
)
def test_diagnostic_failure_counts_are_bounded(contract, capsys, count, bounded):
    contract._diagnostic("contract_check", "failed", "unknown", failures=count)
    diagnostic = events(capsys.readouterr().out)
    assert len(diagnostic) == 1
    assert diagnostic[0]["failures"] == bounded


@pytest.mark.parametrize(
    "site, category",
    [
        ("collect", "other"),
        ("nobody_is_connected", "module_not_found"),
        ("runs_are_cancelled_and_settled_under_the_runtimes_names", "value_error"),
        ("a_chat_can_be_closed_from_outside_its_turn", "import_error"),
    ],
)
def test_nested_catches_preserve_exception_categories(
    contract, monkeypatch, capsys, site, category
):
    guarded_import = builtins.__import__
    imports = {}

    def fake_import(name, *args, **kwargs):
        if name in imports:
            value = imports[name]
            if isinstance(value, BaseException):
                raise value
            return value
        return guarded_import(name, *args, **kwargs)

    if site == "collect":

        class ContractError(Exception):
            pass

        def collect_host():
            raise ContractError(PRIVATE)

        host = SimpleNamespace(collect=collect_host)
        imports["autogpt_desktop"] = SimpleNamespace(
            apps=SimpleNamespace(SERVICES=[]),
            servicehost=SimpleNamespace(
                Host=lambda **kwargs: host, ContractError=ContractError
            ),
        )
        monkeypatch.setattr(contract, "as_a_service_sees_it", lambda *args: None)
        monkeypatch.setattr(contract, "collect", contract.original_collect)
        contract_id = "collect"
    else:
        monkeypatch.setattr(contract, "CHECKS", (getattr(contract, site),))
        contract_id = site
        if site == "nobody_is_connected":
            imports["autogpt_desktop"] = SimpleNamespace(apps=object())
            imports["backend.data"] = ModuleNotFoundError(PRIVATE)
        elif site == "a_chat_can_be_closed_from_outside_its_turn":
            imports["backend.copilot"] = ImportError(PRIVATE)
        else:

            class CancelEvent:
                @staticmethod
                def model_validate_json(payload):
                    raise ValueError(PRIVATE)

            graph = SimpleNamespace(
                GRAPH_EXECUTION_QUEUE_NAME="graph",
                GRAPH_EXECUTION_CANCEL_EXCHANGE=SimpleNamespace(
                    name="graph", type="fanout"
                ),
                CancelExecutionEvent=CancelEvent,
            )
            copilot = SimpleNamespace(
                COPILOT_EXECUTION_QUEUE_NAME="copilot",
                COPILOT_CANCEL_EXCHANGE=SimpleNamespace(name="copilot", type="fanout"),
                CancelCoPilotEvent=CancelEvent,
                CoPilotExecutionEntry=SimpleNamespace(
                    model_fields={"session_id": None, "turn_id": None}
                ),
            )
            runs = SimpleNamespace(
                GRAPH_RUN_QUEUE="graph",
                COPILOT_TURN_QUEUE="copilot",
                GRAPH_CANCEL_EXCHANGE="graph",
                COPILOT_CANCEL_EXCHANGE="copilot",
                GRAPH_CANCEL_FIELD="graph_id",
                COPILOT_SESSION_FIELD="session_id",
            )
            imports["backend.copilot.executor"] = SimpleNamespace(utils=copilot)
            imports["backend.executor"] = SimpleNamespace(utils=graph)
            imports["autogpt_desktop"] = SimpleNamespace(runs=runs)
            monkeypatch.setattr(
                contract.Path,
                "read_text",
                lambda *args, **kwargs: 'else: log_metadata.warning( f"Skipped graph execution',
            )
    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert contract.main() == 1
    output = capsys.readouterr().out
    diagnostics = events(output)
    assert diagnostics[-1]["contract"] == contract_id
    assert diagnostics[-1]["status"] == "failed"
    assert diagnostics[-1]["exception"] == category
    assert diagnostics[-1]["reason"] == "exception"
    assert diagnostics[-1]["failures"] == (2 if site.startswith("runs_") else 1)
    assert len(diagnostics) == (2 if site == "collect" else 4)
    assert PRIVATE in output
    assert any(PRIVATE in message for ok, message in contract.findings if not ok)


class FailingDiagnosticStream(io.StringIO):
    def __init__(self, operation, error_type, stage, status, *, once=True):
        super().__init__()
        self.operation = operation
        self.error_type = error_type
        self.target = (stage, status)
        self.once = once
        self.current = None
        self.failures = 0

    def fail_if_targeted(self):
        if self.current == self.target and (not self.once or not self.failures):
            self.failures += 1
            raise self.error_type(PRIVATE)

    def write(self, text):
        if text.startswith(PREFIX):
            event = json.loads(text[len(PREFIX) :])
            self.current = (event["stage"], event["status"])
            if self.operation == "write":
                self.fail_if_targeted()
        return super().write(text)

    def flush(self):
        if self.operation == "flush":
            self.fail_if_targeted()
        return super().flush()


@pytest.fixture
def scratch_exits(contract, monkeypatch, tmp_path):
    exits = []

    class Scratch:
        def __enter__(self):
            return str(tmp_path / "scratch")

        def __exit__(self, exc_type, exc, traceback):
            exits.append(exc)

    monkeypatch.setattr(contract.tempfile, "TemporaryDirectory", lambda **kw: Scratch())
    return exits


@pytest.mark.parametrize("operation", ["write", "flush"])
@pytest.mark.parametrize("error_type", [OSError, BrokenPipeError, ValueError])
@pytest.mark.parametrize(
    "stage, status",
    [
        ("contract_collect", "started"),
        ("contract_collect", "passed"),
        ("contract_check", "started"),
        ("contract_check", "passed"),
    ],
)
def test_diagnostic_io_failure_latches_until_cleanup_and_resets_next_main(
    contract, monkeypatch, scratch_exits, operation, error_type, stage, status
):
    checked = []

    def check(*args):
        checked.append(True)
        contract.expect(True, "check holds", "")

    monkeypatch.setattr(contract, "CHECKS", (check,))
    stream = FailingDiagnosticStream(operation, error_type, stage, status)
    with monkeypatch.context() as output:
        output.setattr(contract.sys, "stdout", stream)
        assert contract.main() == 1
        assert scratch_exits == [None]
        assert checked == [True]
        assert stream.failures == 1
        assert stream.getvalue().endswith("2 of 2 dependencies hold\n")
        assert contract.main() == 0
    assert scratch_exits == [None, None]
    assert checked == [True, True]
    assert stream.failures == 1
    assert all(ok for ok, _ in contract.findings)
    diagnostics = events(stream.getvalue())
    assert len(diagnostics) == (7 if operation == "write" else 8)
    assert diagnostics[-1]["status"] == "passed"


@pytest.mark.parametrize("operation", ["write", "flush"])
@pytest.mark.parametrize("error_type", [OSError, BrokenPipeError, ValueError])
@pytest.mark.parametrize("check_raises", [False, True])
def test_diagnostic_io_failure_preserves_failed_check_and_continuation(
    contract, monkeypatch, scratch_exits, operation, error_type, check_raises
):
    primary = ValueError(PRIVATE)
    checked = []

    def failing(*args):
        if check_raises:
            raise primary
        contract.expect(False, "check failed", PRIVATE)

    def later(*args):
        checked.append(True)

    monkeypatch.setattr(contract, "CHECKS", (failing, later))
    status = "error" if check_raises else "failed"
    stream = FailingDiagnosticStream(operation, error_type, "contract_check", status)
    with monkeypatch.context() as output:
        output.setattr(contract.sys, "stdout", stream)
        assert contract.main() == 1
    assert scratch_exits == [None]
    assert checked == [True]
    assert stream.failures == 1
    message = (
        f"failing. The check itself failed: ValueError: {PRIVATE}"
        if check_raises
        else f"check failed. {PRIVATE}"
    )
    assert contract.findings == [(True, PRIVATE), (False, message)]
    assert f"BROKEN: {message}\n" in stream.getvalue()
    assert stream.getvalue().endswith("1 of 2 dependencies hold\n")
    assert events(stream.getvalue())[-1]["status"] == "passed"


@pytest.mark.parametrize("operation", ["write", "flush"])
@pytest.mark.parametrize("io_error", [OSError, BrokenPipeError, ValueError])
@pytest.mark.parametrize("primary_type", [BaseException, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("stage", ["contract_collect", "contract_check"])
def test_diagnostic_io_failure_preserves_primary_exception_and_scratch_cleanup(
    contract, monkeypatch, scratch_exits, operation, io_error, primary_type, stage
):
    primary = primary_type(PRIVATE)
    original_collect = contract.collect

    def fail(*args):
        raise primary

    if stage == "contract_collect":
        monkeypatch.setattr(contract, "collect", fail)
    else:
        monkeypatch.setattr(contract, "CHECKS", (fail,))
    stream = FailingDiagnosticStream(operation, io_error, stage, "error", once=False)
    with monkeypatch.context() as output:
        output.setattr(contract.sys, "stdout", stream)
        with pytest.raises(primary_type) as raised:
            contract.main()
        assert raised.value is primary
        assert scratch_exits == [primary]
        assert stream.failures == 1
        monkeypatch.setattr(contract, "collect", original_collect)
        monkeypatch.setattr(contract, "CHECKS", ())
        assert contract.main() == 0
    assert scratch_exits == [primary, None]
    assert events(stream.getvalue())[-1]["status"] == "passed"


def test_diagnostic_io_closed_stdout_preserves_collection_value_error(
    contract, monkeypatch, scratch_exits
):
    primary = ValueError(PRIVATE)
    original_collect = contract.collect
    closed = io.StringIO()
    closed.close()

    def fail(*args):
        raise primary

    monkeypatch.setattr(contract, "collect", fail)
    with monkeypatch.context() as output:
        output.setattr(contract.sys, "stdout", closed)
        contract._diagnostic("contract_collect", "started", "collect")
        with pytest.raises(ValueError) as raised:
            contract.main()
        assert raised.value is primary
    assert scratch_exits == [primary]
    monkeypatch.setattr(contract, "collect", original_collect)
    assert contract.main() == 0
    assert scratch_exits == [primary, None]


def test_diagnostic_io_does_not_swallow_json_value_error(contract, monkeypatch):
    primary = ValueError(PRIVATE)

    def fail(*args, **kwargs):
        raise primary

    monkeypatch.setattr(contract.json, "dumps", fail)
    with pytest.raises(ValueError) as raised:
        contract._diagnostic("contract_collect", "started", "collect")
    assert raised.value is primary
