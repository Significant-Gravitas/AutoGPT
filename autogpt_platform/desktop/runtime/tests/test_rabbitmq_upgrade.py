import json
from pathlib import Path

import pytest

from autogpt_desktop import rabbitmq
from autogpt_desktop.layout import Bundle, DataDir


def make_bundle(
    root: Path, version: str = "4.3.6", directory: str = "rabbit-4.3.6"
) -> Bundle:
    bundle = Bundle(root)
    metadata = bundle.rabbitmq_home / "plugins" / directory / "ebin" / "rabbit.app"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(
        "{application, 'rabbit', [\n"
        '\t{description, "RabbitMQ"},\n'
        f'\t{{vsn, "{version}"}},\n'
        '\t{id, "4435ce2"},\n'
        "\t{modules, []}\n]}.\n",
        encoding="utf-8",
    )
    return bundle


def add_storage(data: DataDir) -> Path:
    storage = data.rabbitmq / "mnesia" / "rabbit@localhost-mnesia" / "schema.DAT"
    storage.parent.mkdir(parents=True, exist_ok=True)
    storage.write_bytes(b"opaque broker storage")
    return storage


def test_unversioned_storage_is_rejected_without_changes(tmp_path: Path):
    bundle = make_bundle(tmp_path / "bundle")
    data = DataDir(tmp_path / "data")
    storage = add_storage(data)
    with pytest.raises(rabbitmq.IncompatibleDataError, match="Doctor"):
        rabbitmq.check_compatible(bundle, data)

    assert storage.read_bytes() == b"opaque broker storage"
    assert not (data.rabbitmq / "desktop-version.json").exists()


def test_unproven_version_uses_plain_startup(tmp_path: Path, monkeypatch):
    bundle = make_bundle(tmp_path / "bundle")
    monkeypatch.setattr(rabbitmq, "_short", str)

    env = rabbitmq.environment(
        bundle, DataDir(tmp_path / "data"), {"rabbitmq_dist": 15002, "epmd": 15001}
    )

    assert env[rabbitmq.SERVER_FLAGS] == rabbitmq.QUIET_SYSLOG


def write_record(data: DataDir, version: str, state: str = "ready") -> Path:
    path = data.rabbitmq / "desktop-version.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema_version": 1, "version": version, "state": state}),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("layout", ["absent", "empty", "configured"])
def test_fresh_data_is_accepted_without_writes_or_subprocesses(
    tmp_path: Path, monkeypatch, layout
):
    bundle = make_bundle(tmp_path / "bundle")
    data = DataDir(tmp_path / "data")
    if layout != "absent":
        (data.rabbitmq / "mnesia").mkdir(parents=True)
    if layout == "configured":
        for name in (
            ".erlang.cookie",
            "rabbitmq.conf",
            "enabled_plugins",
            "erl_inetrc",
        ):
            (data.rabbitmq / name).write_text("do not read", encoding="utf-8")
    original_open = Path.open

    def metadata_only(path, *args, **kwargs):
        assert path.name in ("rabbit.app", "desktop-version.json")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", metadata_only)
    monkeypatch.setattr(
        rabbitmq, "run_tool", lambda *a, **kw: pytest.fail("subprocess launched")
    )

    assert rabbitmq.check_compatible(bundle, data) == "4.3.6"
    assert not (data.rabbitmq / "desktop-version.json").exists()
    if layout == "absent":
        assert not data.root.exists()


@pytest.mark.parametrize(
    ("old", "target", "accepted"),
    [
        ("4.3.6", "4.3.6", True),
        ("4.3.5", "4.3.6", True),
        ("4.3.9", "4.3.10", True),
        ("4.3.10", "4.3.9", False),
        ("4.3.6", "4.3.5", False),
        ("4.3.6", "4.2.9", False),
        ("4.1.8", "4.3.6", False),
        ("4.1.8", "4.2.6", False),
        ("4.2.6", "4.3.6", False),
        ("3.13.8", "4.3.6", False),
    ],
)
def test_version_boundary_is_read_only(
    tmp_path: Path, old: str, target: str, accepted: bool
):
    bundle = make_bundle(tmp_path / "bundle", target)
    data = DataDir(tmp_path / "data")
    storage = add_storage(data)
    record = write_record(data, old)
    before = record.read_bytes()

    if accepted:
        assert rabbitmq.check_compatible(bundle, data) == target
    else:
        with pytest.raises(rabbitmq.IncompatibleDataError, match="Doctor"):
            rabbitmq.check_compatible(bundle, data)

    assert record.read_bytes() == before
    assert storage.read_bytes() == b"opaque broker storage"


def test_version_named_flags_and_directories_do_not_identify_the_writer(tmp_path: Path):
    bundle = make_bundle(tmp_path / "bundle")
    data = DataDir(tmp_path / "data")
    add_storage(data)
    (data.rabbitmq / "mnesia" / "rabbit@localhost-feature_flags").write_text(
        "['rabbitmq_4.0.0', 'rabbitmq_4.1.0', 'rabbitmq_4.2.0', khepri_db].\n"
    )
    (data.rabbitmq / "mnesia" / "rabbit-4.3.6").mkdir()

    with pytest.raises(rabbitmq.IncompatibleDataError, match="unknown"):
        rabbitmq.check_compatible(bundle, data)


@pytest.mark.parametrize(
    "content",
    [
        "",
        "{",
        "[]",
        "null",
        '"4.3.6"',
        '{"schema_version":2,"version":"4.3.6","state":"ready"}',
        '{"schema_version":true,"version":"4.3.6","state":"ready"}',
        '{"schema_version":1,"version":"4.3.6","state":"unknown"}',
        '{"schema_version":1,"version":"4.3.6-rc.1","state":"ready"}',
        '{"schema_version":1,"version":null,"state":"ready"}',
        '{"schema_version":1,"version":"4.1.8","version":"4.3.6","state":"ready"}',
        '[["schema_version",1],["version","4.3.6"],["state","ready"]]',
        "x" * 1025,
    ],
)
def test_invalid_stored_metadata_is_rejected_even_without_storage(
    tmp_path: Path, content: str
):
    bundle = make_bundle(tmp_path / "bundle")
    data = DataDir(tmp_path / "data")
    record = write_record(data, "4.3.6")
    record.write_text(content, encoding="utf-8")

    with pytest.raises(rabbitmq.IncompatibleDataError, match="metadata.*invalid"):
        rabbitmq.check_compatible(bundle, data)

    assert record.read_text("utf-8") == content


def test_unreadable_storage_does_not_become_fresh(tmp_path: Path, monkeypatch):
    bundle = make_bundle(tmp_path / "bundle")
    data = DataDir(tmp_path / "data")
    (data.rabbitmq / "mnesia").mkdir(parents=True)
    original = rabbitmq.os.scandir

    def denied(path):
        if path == data.rabbitmq / "mnesia":
            raise PermissionError("denied")
        return original(path)

    monkeypatch.setattr(rabbitmq.os, "scandir", denied)

    with pytest.raises(rabbitmq.IncompatibleDataError, match="Doctor"):
        rabbitmq.check_compatible(bundle, data)


@pytest.mark.parametrize("entry", ["unexpected-storage", "mnesia"])
def test_unexpected_layout_or_non_directory_storage_is_rejected(
    tmp_path: Path, entry: str
):
    bundle = make_bundle(tmp_path / "bundle")
    data = DataDir(tmp_path / "data")
    data.rabbitmq.mkdir(parents=True)
    (data.rabbitmq / entry).write_bytes(b"opaque storage")

    with pytest.raises(rabbitmq.IncompatibleDataError, match="Doctor"):
        rabbitmq.check_compatible(bundle, data)


def test_interrupted_patch_upgrade_cannot_reopen_with_older_binary(tmp_path: Path):
    older = make_bundle(tmp_path / "older", "4.3.5")
    newer = make_bundle(tmp_path / "newer", "4.3.6")
    data = DataDir(tmp_path / "data")
    add_storage(data)
    record = write_record(data, "4.3.5")

    assert rabbitmq.record_starting(newer, data) == "4.3.6"
    assert json.loads(record.read_text("utf-8"))["state"] == "starting"
    with pytest.raises(rabbitmq.IncompatibleDataError, match="unsupported downgrade"):
        rabbitmq.check_compatible(older, data)
    assert rabbitmq.check_compatible(newer, data) == "4.3.6"
    assert json.loads(record.read_text("utf-8"))["state"] == "starting"


def test_success_is_recorded_separately_after_the_start_attempt(tmp_path: Path):
    bundle = make_bundle(tmp_path / "bundle")
    data = DataDir(tmp_path / "data")
    version = rabbitmq.record_starting(bundle, data)
    record = data.rabbitmq / "desktop-version.json"
    assert json.loads(record.read_text("utf-8"))["state"] == "starting"
    add_storage(data)

    rabbitmq.record_ready(data, version)

    assert json.loads(record.read_text("utf-8")) == {
        "schema_version": 1,
        "version": "4.3.6",
        "state": "ready",
    }
    assert rabbitmq.check_compatible(bundle, data) == "4.3.6"


@pytest.mark.parametrize("state", ["missing", "wrong_version", "no_storage"])
def test_ready_cannot_invent_or_advance_a_version_record(tmp_path: Path, state: str):
    data = DataDir(tmp_path / "data")
    if state != "no_storage":
        add_storage(data)
    if state != "missing":
        write_record(data, "4.3.5" if state == "wrong_version" else "4.3.6", "starting")

    with pytest.raises(rabbitmq.IncompatibleDataError, match="Cannot confirm"):
        rabbitmq.record_ready(data, "4.3.6")


def test_start_record_does_not_bypass_unknown_persisted_data(tmp_path: Path):
    bundle = make_bundle(tmp_path / "bundle")
    data = DataDir(tmp_path / "data")
    add_storage(data)

    with pytest.raises(rabbitmq.IncompatibleDataError, match="unknown"):
        rabbitmq.record_starting(bundle, data)

    assert not (data.rabbitmq / "desktop-version.json").exists()


@pytest.mark.parametrize("version", ["4.1.8", "4.3.6"])
def test_application_contents_determine_the_version_not_the_directory(
    tmp_path: Path, version: str
):
    bundle = make_bundle(tmp_path / "bundle", version, "rabbit-99.99.99")

    assert rabbitmq.bundled_version(bundle) == version
    flags = rabbitmq.server_flags(bundle, {"rabbitmq_dist": 15002})
    assert ("-sname rabbit@localhost" in flags) == (version == "4.1.8")


@pytest.mark.parametrize(
    "failure", ["absent", "malformed", "oversized", "ambiguous", "prerelease"]
)
def test_unknown_bundle_is_blocked_and_gets_no_startup_shortcut(
    tmp_path: Path, failure: str
):
    bundle = make_bundle(tmp_path / "bundle")
    app = bundle.rabbitmq_home / "plugins" / "rabbit-4.3.6" / "ebin" / "rabbit.app"
    if failure == "absent":
        app.unlink()
    elif failure == "malformed":
        app.write_text('{application, "not rabbit", [{vsn, "4.1.8"}]}.')
    elif failure == "oversized":
        app.write_bytes(b"x" * (rabbitmq.METADATA_BYTES + 1))
    elif failure == "ambiguous":
        make_bundle(bundle.root, "4.1.8", "rabbit-4.1.8")
    else:
        make_bundle(bundle.root, "4.3.6-rc.1")

    with pytest.raises(
        rabbitmq.IncompatibleDataError, match="bundled RabbitMQ version"
    ):
        rabbitmq.check_compatible(bundle, DataDir(tmp_path / "data"))
    assert (
        rabbitmq.server_flags(bundle, {"rabbitmq_dist": 15002}) == rabbitmq.QUIET_SYSLOG
    )


def test_metadata_directory_inspection_is_bounded(tmp_path: Path, monkeypatch):
    bundle = make_bundle(tmp_path / "bundle")
    for number in range(3):
        (bundle.rabbitmq_home / "plugins" / f"unrelated-{number}").mkdir()
    monkeypatch.setattr(rabbitmq, "METADATA_ENTRIES", 2)

    with pytest.raises(
        rabbitmq.IncompatibleDataError, match="bundled RabbitMQ version"
    ):
        rabbitmq.bundled_version(bundle)


def test_nested_version_does_not_authorize_an_incompatible_writer(tmp_path: Path):
    bundle = make_bundle(tmp_path / "bundle")
    app = bundle.rabbitmq_home / "plugins" / "rabbit-4.3.6" / "ebin" / "rabbit.app"
    app.write_text(
        '{application, rabbit, [{vsn, "4.3.6"},\n'
        '  {env, [\n    {vsn, "4.1.8"}\n  ]},\n'
        "  {modules, []}\n]}.\n",
        encoding="utf-8",
    )
    data = DataDir(tmp_path / "data")
    add_storage(data)
    write_record(data, "4.1.8")

    with pytest.raises(rabbitmq.IncompatibleDataError, match="staged upgrade"):
        rabbitmq.check_compatible(bundle, data)

    assert rabbitmq.bundled_version(bundle) == "4.3.6"
    assert (
        rabbitmq.server_flags(bundle, {"rabbitmq_dist": 15002}) == rabbitmq.QUIET_SYSLOG
    )


def test_truncated_application_is_rejected_before_any_profile_write(tmp_path: Path):
    bundle = make_bundle(tmp_path / "bundle")
    app = bundle.rabbitmq_home / "plugins" / "rabbit-4.3.6" / "ebin" / "rabbit.app"
    app.write_text('{application, rabbit, [\n{vsn, "4.3.6"},\n', encoding="utf-8")
    data = DataDir(tmp_path / "data")

    with pytest.raises(
        rabbitmq.IncompatibleDataError, match="bundled RabbitMQ version"
    ):
        rabbitmq.record_starting(bundle, data)

    assert not data.root.exists()


def test_interrupted_ready_replacement_preserves_retry_and_downgrade_floor(
    tmp_path: Path, monkeypatch
):
    older = make_bundle(tmp_path / "older", "4.3.5")
    newer = make_bundle(tmp_path / "newer", "4.3.6")
    data = DataDir(tmp_path / "data")
    add_storage(data)
    record = write_record(data, "4.3.5")
    version = rabbitmq.record_starting(newer, data)
    committed = record.read_bytes()

    def interrupted(*args):
        raise OSError("interrupted before replacement")

    with monkeypatch.context() as failure:
        failure.setattr(rabbitmq.os, "replace", interrupted)
        with pytest.raises(OSError, match="interrupted before replacement"):
            rabbitmq.record_ready(data, version)
    leftovers = list(data.rabbitmq.glob(".desktop-version.json.*.tmp"))
    assert len(leftovers) == 1
    temporary = leftovers[0].read_bytes()
    assert record.read_bytes() == committed

    assert rabbitmq.check_compatible(newer, data) == version
    with pytest.raises(rabbitmq.IncompatibleDataError, match="unsupported downgrade"):
        rabbitmq.check_compatible(older, data)
    assert record.read_bytes() == committed
    assert leftovers[0].read_bytes() == temporary
    rabbitmq.record_starting(newer, data)
    rabbitmq.record_ready(data, version)
    assert json.loads(record.read_text("utf-8"))["state"] == "ready"


@pytest.mark.parametrize(
    "name", ["enabled_plugins", "erl_inetrc", "rabbitmq.conf", ".erlang.cookie"]
)
def test_linked_scaffold_is_rejected_without_touching_its_target(
    tmp_path: Path, name: str
):
    bundle = make_bundle(tmp_path / "bundle")
    data = DataDir(tmp_path / "data")
    data.rabbitmq.mkdir(parents=True)
    outside = tmp_path / "outside-profile.txt"
    outside.write_bytes(b"keep outside-profile content")
    (data.rabbitmq / name).symlink_to(outside)

    with pytest.raises(rabbitmq.IncompatibleDataError, match="Doctor"):
        rabbitmq.check_compatible(bundle, data)

    assert outside.read_bytes() == b"keep outside-profile content"
    assert not (data.rabbitmq / "desktop-version.json").exists()


def test_linked_node_storage_is_not_authorized_by_the_profile_marker(tmp_path: Path):
    bundle = make_bundle(tmp_path / "bundle")
    data = DataDir(tmp_path / "data")
    mnesia = data.rabbitmq / "mnesia"
    mnesia.mkdir(parents=True)
    outside = tmp_path / "outside-node"
    outside.mkdir()
    (outside / "schema.DAT").write_bytes(b"unknown writer")
    (mnesia / "rabbit@localhost-mnesia").symlink_to(outside, target_is_directory=True)
    record = write_record(data, "4.3.6")
    committed = record.read_bytes()

    with pytest.raises(rabbitmq.IncompatibleDataError, match="Doctor"):
        rabbitmq.check_compatible(bundle, data)

    assert record.read_bytes() == committed
    assert (outside / "schema.DAT").read_bytes() == b"unknown writer"


@pytest.mark.parametrize(
    "content",
    [
        '{application, rabbit, [{env, [{vsn, "4.1.8"}]}]}.',
        '{application, rabbit, [{vsn, "4.3.6"}, {vsn, "4.1.8"}]}.',
        '{application, rabbit, [{vsn, "4.3.6"}, {\'vsn\', "4.3.6"}]}.',
        '{application, rabbit, [{vsn, "4.3.6"}, {modules, []}]}',
        '{application, rabbit, [{vsn, "4.3.6"}, {modules, []]}.',
        '{application, rabbit, [{vsn, "4.3.6"}]} garbage.',
        '{application, rabbit, [{vsn, "4.3.6"}]}. trailing',
        '{application, rabbit, [{vsn, "4.3.6"}]}. {vsn, "4.1.8"}.',
        '{application, rabbit, [{vsn, "4.3.6"}, {env, [unterminated,]}]}.',
        '{application, rabbit, [{vsn, "4.3.6"}, {description, "unterminated}]}.',
        '{application, rabbit, [{vsn, "4.3.6"}, {env, #{unsupported => syntax}}]}.',
        '{application, rabbit, [{vsn, "4.3.6"}, {env, '
        + "[" * 40
        + "x"
        + "]" * 40
        + "}]}.",
    ],
)
def test_application_metadata_must_be_one_complete_supported_term(
    tmp_path: Path, content: str
):
    bundle = make_bundle(tmp_path / "bundle")
    app = bundle.rabbitmq_home / "plugins" / "rabbit-4.3.6" / "ebin" / "rabbit.app"
    app.write_text(content, encoding="utf-8")

    with pytest.raises(
        rabbitmq.IncompatibleDataError, match="bundled RabbitMQ version"
    ):
        rabbitmq.check_compatible(bundle, DataDir(tmp_path / "data"))
    assert (
        rabbitmq.server_flags(bundle, {"rabbitmq_dist": 15002}) == rabbitmq.QUIET_SYSLOG
    )


@pytest.mark.parametrize("version", ["4.1.8", "4.3.6"])
def test_generated_application_literals_comments_and_quoted_delimiters(
    tmp_path: Path, version: str
):
    bundle = make_bundle(tmp_path / "bundle")
    app = bundle.rabbitmq_home / "plugins" / "rabbit-4.3.6" / "ebin" / "rabbit.app"
    app.write_text(
        '% {vsn, "9.9.9"} ]}. ignored comment\n'
        "{application, 'rabbit', [\n"
        f'{{vsn, "{version}"}},\n'
        r'{description, "escaped \" quote, \\ backslash, % and ]}."},' + "\n"
        "{env, [{vm_memory_high_watermark, 0.6}, %% numeric literal\n"
        "  {default_vhost, <<\"/\">>}, {auth_mechanisms, ['PLAIN']},\n"
        '  {nested, {vsn, "9.9.9"}}]}, {modules, []}]}. % final comment\n',
        encoding="utf-8",
    )

    assert rabbitmq.bundled_version(bundle) == version
    assert ("-sname" in rabbitmq.server_flags(bundle, {"rabbitmq_dist": 15002})) == (
        version == "4.1.8"
    )


@pytest.mark.parametrize(
    "temporary_content", ["invalid JSON", '{"version":"1.0.0"}', '{"version":"99.0.0"}']
)
def test_exact_marker_temporary_files_are_not_read_or_promoted(
    tmp_path: Path, monkeypatch, temporary_content: str
):
    bundle = make_bundle(tmp_path / "bundle")
    data = DataDir(tmp_path / "data")
    add_storage(data)
    record = write_record(data, "4.3.6", "starting")
    temporary = data.rabbitmq / ".desktop-version.json.0123456789abcdef.tmp"
    temporary.write_text(temporary_content, encoding="utf-8")
    committed = record.read_bytes()
    original_open = Path.open

    def no_temporary_read(path, *args, **kwargs):
        assert path != temporary
        return original_open(path, *args, **kwargs)

    with monkeypatch.context() as checking:
        checking.setattr(Path, "open", no_temporary_read)
        assert rabbitmq.check_compatible(bundle, data) == "4.3.6"
    assert record.read_bytes() == committed
    assert temporary.read_text("utf-8") == temporary_content


@pytest.mark.parametrize(
    "temporary_name",
    [
        ".desktop-version.json.0123456789abcde.tmp",
        ".desktop-version.json.0123456789abcdef0.tmp",
        ".desktop-version.json.0123456789ABCDEF.tmp",
        ".desktop-version.json.0123456789abcdeg.tmp",
        ".desktop-version.json.0123456789abcdef.tmp.extra",
        "desktop-version.json.0123456789abcdef.tmp",
        ".rabbitmq.conf.0123456789abcdef.tmp",
    ],
)
def test_temporary_file_lookalikes_are_rejected(tmp_path: Path, temporary_name: str):
    bundle = make_bundle(tmp_path / "bundle")
    data = DataDir(tmp_path / "data")
    add_storage(data)
    write_record(data, "4.3.6")
    (data.rabbitmq / temporary_name).write_text("untrusted", encoding="utf-8")

    with pytest.raises(rabbitmq.IncompatibleDataError, match="Doctor"):
        rabbitmq.check_compatible(bundle, data)


@pytest.mark.parametrize("kind", ["directory", "symlink", "hardlink", "dangling"])
def test_marker_temporary_names_must_be_regular_unlinked_files(
    tmp_path: Path, kind: str
):
    bundle = make_bundle(tmp_path / "bundle")
    data = DataDir(tmp_path / "data")
    add_storage(data)
    write_record(data, "4.3.6")
    temporary = data.rabbitmq / ".desktop-version.json.0123456789abcdef.tmp"
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"preserve")
    if kind == "directory":
        temporary.mkdir()
    elif kind == "hardlink":
        temporary.hardlink_to(outside)
    else:
        temporary.symlink_to(outside if kind == "symlink" else tmp_path / "missing")

    with pytest.raises(rabbitmq.IncompatibleDataError, match="Doctor"):
        rabbitmq.check_compatible(bundle, data)
    assert outside.read_bytes() == b"preserve"


@pytest.mark.parametrize("populated", [False, True])
def test_temporary_marker_is_not_writer_evidence(tmp_path: Path, populated: bool):
    bundle = make_bundle(tmp_path / "bundle")
    data = DataDir(tmp_path / "data")
    data.rabbitmq.mkdir(parents=True)
    (data.rabbitmq / ".desktop-version.json.0123456789abcdef.tmp").write_text(
        '{"schema_version":1,"version":"4.3.6","state":"ready"}', encoding="utf-8"
    )
    if populated:
        add_storage(data)
        with pytest.raises(rabbitmq.IncompatibleDataError, match="unknown"):
            rabbitmq.check_compatible(bundle, data)
    else:
        assert rabbitmq.check_compatible(bundle, data) == "4.3.6"
    assert not (data.rabbitmq / "desktop-version.json").exists()


@pytest.mark.parametrize("name", ["enabled_plugins", "erl_inetrc"])
def test_hard_linked_scaffold_cannot_overwrite_another_file(tmp_path: Path, name: str):
    bundle = make_bundle(tmp_path / "bundle")
    data = DataDir(tmp_path / "data")
    data.rabbitmq.mkdir(parents=True)
    outside = tmp_path / "outside-profile.txt"
    outside.write_bytes(b"preserve")
    (data.rabbitmq / name).hardlink_to(outside)

    with pytest.raises(rabbitmq.IncompatibleDataError, match="Doctor"):
        rabbitmq.check_compatible(bundle, data)
    assert outside.read_bytes() == b"preserve"


@pytest.mark.parametrize(
    "name",
    [
        "enabled_plugins",
        "erl_inetrc",
        "rabbitmq.conf",
        ".erlang.cookie",
        "desktop-version.json",
    ],
)
def test_scaffold_files_cannot_be_directories(tmp_path: Path, name: str):
    bundle = make_bundle(tmp_path / "bundle")
    data = DataDir(tmp_path / "data")
    (data.rabbitmq / name).mkdir(parents=True)

    with pytest.raises(rabbitmq.IncompatibleDataError, match="Doctor"):
        rabbitmq.check_compatible(bundle, data)


def test_all_direct_mnesia_entries_are_checked_without_reading_storage(
    tmp_path: Path, monkeypatch
):
    bundle = make_bundle(tmp_path / "bundle")
    data = DataDir(tmp_path / "data")
    add_storage(data)
    write_record(data, "4.3.6")
    mnesia = data.rabbitmq / "mnesia"
    (mnesia / "rabbit@localhost-feature_flags").write_text(
        "opaque flags", encoding="utf-8"
    )
    original_open = Path.open
    original_scandir = rabbitmq.os.scandir

    def metadata_only(path, *args, **kwargs):
        assert path.name in ("rabbit.app", "desktop-version.json")
        return original_open(path, *args, **kwargs)

    def no_recursive_storage_walk(path):
        assert path not in (mnesia / "rabbit@localhost-mnesia", tmp_path / "outside")
        return original_scandir(path)

    monkeypatch.setattr(Path, "open", metadata_only)
    monkeypatch.setattr(rabbitmq.os, "scandir", no_recursive_storage_walk)
    assert rabbitmq.check_compatible(bundle, data) == "4.3.6"
    (mnesia / "z-linked-node").symlink_to(
        tmp_path / "outside", target_is_directory=True
    )

    with pytest.raises(rabbitmq.IncompatibleDataError, match="Doctor"):
        rabbitmq.check_compatible(bundle, data)


@pytest.mark.parametrize("location", ["mnesia", "temporary"])
def test_storage_and_ignored_temporary_files_count_toward_entry_bounds(
    tmp_path: Path, monkeypatch, location: str
):
    bundle = make_bundle(tmp_path / "bundle")
    data = DataDir(tmp_path / "data")
    write_record(data, "4.3.6")
    if location == "mnesia":
        mnesia = data.rabbitmq / "mnesia"
        mnesia.mkdir()
        for number in range(3):
            (mnesia / str(number)).touch()
    else:
        for number in range(2):
            (data.rabbitmq / f".desktop-version.json.{number:016x}.tmp").touch()
    monkeypatch.setattr(rabbitmq, "METADATA_ENTRIES", 2)

    with pytest.raises(rabbitmq.IncompatibleDataError, match="Doctor"):
        rabbitmq.check_compatible(bundle, data)
