"""Operational CLI contracts without a database or external service."""

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock, call

import pytest
from pydantic import ValidationError

from backend.api.features.store import catalog_release as cli
from backend.api.features.store import catalog_release_load_test
from backend.api.features.store.catalog_release_model import Adoption, Preview, digest

checkout = catalog_release_load_test.checkout


@pytest.fixture(scope="session", autouse=True)
def graph_cleanup():
    """This module cannot create server records; all DB boundaries are mocked."""
    yield


@pytest.fixture(autouse=True)
def boundaries(monkeypatch):
    boundary = Mock()
    for name in ("connect", "disconnect"):
        operation = AsyncMock()
        boundary.attach_mock(operation, name)
        monkeypatch.setattr(cli.database, name, operation)
    for name in (
        "preview_release",
        "apply_release",
        "preview_rollback",
        "rollback_release",
    ):
        operation = AsyncMock(side_effect=AssertionError("unexpected operation"))
        boundary.attach_mock(operation, name)
        monkeypatch.setattr(cli, name, operation)
    return boundary


@pytest.fixture
def adoption():
    return Adoption(skills={"demo": "listing-id"}, experts={"max": "expert-id"})


@pytest.fixture
def approved(adoption):
    return Preview(
        database_target="isolated-test-target",
        rollback_release_id="backup-release",
        release_id="saved-release",
        revision="a" * 40,
        previous_release_id="previous-release",
        generation=7,
        state_sha256="b" * 64,
        adoption_sha256=digest(adoption.model_dump()),
        create_skills=[],
        update_skills=["demo"],
        retire_skills=[],
        expert_keys=["max"],
        activate_experts=[],
    )


@pytest.fixture
def documents(tmp_path_factory, adoption, approved):
    # Approval artifacts must be outside the catalogue's clean Git checkout.
    root = tmp_path_factory.mktemp("catalogue-cli-documents")
    (root / "adoption.json").write_text(adoption.model_dump_json(), encoding="utf-8")
    (root / "approved.json").write_text(approved.model_dump_json(), encoding="utf-8")
    return root


def invoke(monkeypatch, *arguments):
    monkeypatch.setattr(sys, "argv", ["catalogue-release", *map(str, arguments)])
    cli.main()


def operation_arguments(command: str, documents: Path, checkout) -> list[str | Path]:
    arguments: list[str | Path] = [
        command,
        "--adoption",
        documents / "adoption.json",
    ]
    if command in {"preview", "apply"}:
        root, revision = checkout
        arguments.extend(["--catalogue", root, "--revision", revision])
    else:
        arguments.extend(["--release-id", "saved-release"])
    return arguments


@pytest.mark.parametrize("command", ["apply", "rollback"])
def test_mutations_require_approval_argument_before_connecting(
    command, checkout, documents, boundaries, monkeypatch, capsys
):
    with pytest.raises(SystemExit) as error:
        invoke(monkeypatch, *operation_arguments(command, documents, checkout))
    assert error.value.code == 2
    assert "--approved" in capsys.readouterr().err
    assert boundaries.mock_calls == []


@pytest.mark.parametrize("document", ["adoption", "approved"])
@pytest.mark.parametrize("invalid", ["missing", "malformed", "unknown-field"])
def test_invalid_input_files_never_connect(
    document, invalid, checkout, documents, boundaries, monkeypatch
):
    path = documents / f"{document}.json"
    if invalid == "missing":
        path.unlink()
    elif invalid == "malformed":
        path.write_text("{invalid", encoding="utf-8")
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        value["unreviewed_override"] = True
        path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises((FileNotFoundError, ValidationError)):
        invoke(
            monkeypatch,
            *operation_arguments("apply", documents, checkout),
            "--approved",
            documents / "approved.json",
        )
    assert boundaries.mock_calls == []


@pytest.mark.parametrize("changed", ["dirty", "wrong-revision"])
def test_checkout_changes_never_connect(
    changed, checkout, documents, boundaries, monkeypatch
):
    root, revision = checkout
    if changed == "dirty":
        (root / "skills/demo/SKILL.md").write_text("unreviewed", encoding="utf-8")
        message = "must be clean"
    else:
        revision = "0" * 40
        message = "does not match"
    with pytest.raises(ValueError, match=message):
        invoke(
            monkeypatch,
            *operation_arguments("apply", documents, (root, revision)),
            "--approved",
            documents / "approved.json",
        )
    assert boundaries.mock_calls == []


def test_rollback_uses_saved_release_without_a_catalogue_checkout(
    documents, adoption, approved, boundaries, monkeypatch
):
    loader = Mock(side_effect=AssertionError("rollback must use the saved release"))
    monkeypatch.setattr(cli, "load_release", loader)
    boundaries.preview_rollback.side_effect = None
    boundaries.preview_rollback.return_value = approved
    boundaries.rollback_release.side_effect = None
    boundaries.rollback_release.return_value = approved.release_id
    preview_path = documents / "rollback-preview.json"
    output_path = documents / "rollback-result.json"
    invoke(
        monkeypatch,
        *operation_arguments("preview-rollback", documents, None),
        "--output",
        preview_path,
    )
    invoke(
        monkeypatch,
        *operation_arguments("rollback", documents, None),
        "--approved",
        preview_path,
        "--output",
        output_path,
    )
    loader.assert_not_called()
    boundaries.rollback_release.assert_awaited_once_with(
        "saved-release", adoption, approved
    )
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "release_id": "saved-release"
    }
    assert boundaries.mock_calls == [
        call.connect(),
        call.preview_rollback("saved-release", adoption),
        call.disconnect(),
        call.connect(),
        call.rollback_release("saved-release", adoption, approved),
        call.disconnect(),
    ]


@pytest.mark.parametrize(
    ("command", "operation"),
    [
        ("preview", "preview_release"),
        ("apply", "apply_release"),
        ("preview-rollback", "preview_rollback"),
        ("rollback", "rollback_release"),
    ],
)
def test_operation_failure_disconnects_without_success_output(
    command, operation, checkout, documents, boundaries, monkeypatch, capsys
):
    service = AsyncMock(side_effect=RuntimeError("reviewed state changed"))
    monkeypatch.setattr(cli, operation, service)
    output_path = documents / "result.json"
    output_path.write_text("existing artifact", encoding="utf-8")
    with pytest.raises(RuntimeError, match="reviewed state changed"):
        invoke(
            monkeypatch,
            *operation_arguments(command, documents, checkout),
            *(
                ["--approved", documents / "approved.json"]
                if command in {"apply", "rollback"}
                else []
            ),
            "--output",
            output_path,
        )
    service.assert_awaited_once()
    assert boundaries.mock_calls == [call.connect(), call.disconnect()]
    assert output_path.read_text(encoding="utf-8") == "existing artifact"
    assert capsys.readouterr().out == ""


def test_preview_file_is_consumed_by_apply_unchanged(
    checkout, documents, adoption, approved, boundaries, monkeypatch
):
    release = cli.load_release(*checkout)
    preview = approved.model_copy(
        update={"release_id": release.release_id, "revision": release.revision}
    )
    boundaries.preview_release.side_effect = None
    boundaries.preview_release.return_value = preview
    boundaries.apply_release.side_effect = None
    boundaries.apply_release.return_value = release.release_id
    preview_path = documents / "review-this.json"
    output_path = documents / "apply-result.json"
    invoke(
        monkeypatch,
        *operation_arguments("preview", documents, checkout),
        "--output",
        preview_path,
    )
    preview_bytes = preview_path.read_bytes()
    assert Preview.model_validate_json(preview_bytes) == preview
    invoke(
        monkeypatch,
        *operation_arguments("apply", documents, checkout),
        "--approved",
        preview_path,
        "--output",
        output_path,
    )
    boundaries.apply_release.assert_awaited_once_with(release, adoption, preview)
    assert preview_path.read_bytes() == preview_bytes
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "release_id": release.release_id
    }
    assert boundaries.disconnect.await_count == 2


def test_preview_stdout_is_machine_readable(
    checkout, documents, approved, boundaries, monkeypatch, capsys
):
    boundaries.preview_release.side_effect = None
    boundaries.preview_release.return_value = approved
    invoke(monkeypatch, *operation_arguments("preview", documents, checkout))
    assert Preview.model_validate_json(capsys.readouterr().out) == approved
    boundaries.disconnect.assert_awaited_once()


def test_output_write_failure_still_disconnects(
    checkout, documents, approved, boundaries, monkeypatch, capsys
):
    boundaries.preview_release.side_effect = None
    boundaries.preview_release.return_value = approved
    with pytest.raises(OSError):
        invoke(
            monkeypatch,
            *operation_arguments("preview", documents, checkout),
            "--output",
            documents,
        )
    boundaries.disconnect.assert_awaited_once()
    assert capsys.readouterr().out == ""
