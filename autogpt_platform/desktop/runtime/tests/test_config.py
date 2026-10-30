import json
import re
import socket
import stat
import sys
import threading
from pathlib import Path

import pytest

from autogpt_desktop import ports, rabbitmq, settings, valkey
from autogpt_desktop.layout import Bundle, DataDir

DESKTOP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DESKTOP / "build"))

import frontend_role_sql  # noqa: E402
import lock_export  # noqa: E402


@pytest.fixture
def data(tmp_path: Path) -> DataDir:
    data = DataDir(tmp_path / "data with space")
    data.prepare()
    return data


@pytest.fixture
def bundle(tmp_path: Path) -> Bundle:
    return Bundle(tmp_path / "runtime")


@pytest.mark.parametrize(
    "raw, expected",
    [
        ('"quoted value"', "quoted value"),
        ("'quoted value'", "quoted value"),
        ('" spaces inside "', " spaces inside "),
        ('""', ""),
        ("''", ""),
        ('"unmatched', '"unmatched'),
        ("'mismatched\"", "'mismatched\""),
        ("unquoted=value#literal", "unquoted=value#literal"),
        (r'"C:\\path\\folder"', r"C:\\path\\folder"),
    ],
)
def test_settings_values_strip_one_matching_quote_pair(tmp_path, raw, expected):
    path = tmp_path / "settings.env"
    path.write_text(f" TEST_VALUE = {raw} \n", encoding="utf-8")
    assert settings._read_env(path) == {"TEST_VALUE": expected}


def test_ports_are_unique_in_range_and_remembered(tmp_path: Path):
    path = tmp_path / "ports.json"
    first = ports.allocate(path)
    assert set(first) == set(ports.PORT_NAMES)
    assert len(set(first.values())) == len(first)
    assert all(port in ports.PORT_RANGE for port in first.values())
    assert ports.allocate(path) == first


def test_a_taken_port_is_replaced_and_the_rest_are_kept(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(ports, "PUBLIC_PORT_PATIENCE_SECONDS", 0)
    path = tmp_path / "ports.json"
    first = ports.allocate(path)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as squatter:
        squatter.bind(("127.0.0.1", first["public"]))
        second = ports.allocate(path)
    assert second["public"] != first["public"]
    assert {k: v for k, v in second.items() if k != "public"} == {
        k: v for k, v in first.items() if k != "public"
    }
    assert json.loads(path.read_text()) == second


def test_a_replacement_port_never_takes_another_services_port(
    tmp_path: Path, monkeypatch
):
    """Seen on CI: the new public port was drawn from the whole range, landed
    on the port Valkey had on record, and Valkey was moved as well."""
    monkeypatch.setattr(ports, "PUBLIC_PORT_PATIENCE_SECONDS", 0)
    path = tmp_path / "ports.json"
    first = ports.allocate(path)
    draws = iter([first["valkey"], first["postgres"]])
    choose = ports.random.choice
    monkeypatch.setattr(
        ports.random, "choice", lambda options: next(draws, None) or choose(options)
    )
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as squatter:
        squatter.bind(("127.0.0.1", first["public"]))
        second = ports.allocate(path)

    assert second["public"] not in first.values()
    assert {k: v for k, v in second.items() if k != "public"} == {
        k: v for k, v in first.items() if k != "public"
    }


def test_the_public_port_is_waited_for_before_it_is_given_up(tmp_path: Path):
    """It is the app's origin: sessions and OAuth redirect URIs hang off it,
    and its usual holder is a previous run that is still shutting down."""
    path = tmp_path / "ports.json"
    first = ports.allocate(path)
    squatter = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    squatter.bind(("127.0.0.1", first["public"]))
    threading.Timer(1, squatter.close).start()

    assert ports.allocate(path) == first


def free_port() -> int:
    return ports._free_port(exclude=set())


def test_a_new_install_takes_the_preferred_public_port(tmp_path: Path, monkeypatch):
    preferred = free_port()
    monkeypatch.setattr(ports, "PREFERRED", {"public": preferred})
    path = tmp_path / "ports.json"

    first = ports.allocate(path)

    assert first["public"] == preferred
    assert list(first.values()).count(preferred) == 1
    assert ports.allocate(path) == first


def test_an_install_that_has_a_port_is_never_moved_to_the_preferred_one(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(ports, "PREFERRED", {})
    path = tmp_path / "ports.json"
    first = ports.allocate(path)

    monkeypatch.setattr(ports, "PREFERRED", {"public": free_port()})
    assert ports.allocate(path) == first

    # Not even when its own port is taken: the preference is for installs
    # that have nothing to lose yet.
    monkeypatch.setattr(ports, "PUBLIC_PORT_PATIENCE_SECONDS", 0)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as squatter:
        squatter.bind(("127.0.0.1", first["public"]))
        moved = ports.allocate(path)
    assert moved["public"] not in (first["public"], ports.PREFERRED["public"])


def test_a_busy_preferred_port_falls_back_to_a_free_one(tmp_path: Path, monkeypatch):
    preferred = free_port()
    monkeypatch.setattr(ports, "PREFERRED", {"public": preferred})
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as squatter:
        squatter.bind(("127.0.0.1", preferred))
        first = ports.allocate(tmp_path / "ports.json")

    assert first["public"] != preferred
    assert first["public"] in ports.PORT_RANGE
    # ...and that is now this install's port, even once the preferred one is free.
    assert ports.allocate(tmp_path / "ports.json") == first


def test_a_port_with_a_live_listener_is_not_free():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", free_port()))
        listener.listen()
        assert ports._is_free(listener.getsockname()[1]) is False


@pytest.mark.skipif(
    sys.platform == "win32", reason="Windows has no TIME_WAIT bind refusal"
)
def test_a_port_its_last_listener_just_left_is_free_at_once():
    """The proxy closes its connections first on every stop, which leaves the
    port in TIME_WAIT for a minute. A restart must not read that as taken and
    move the app's address."""
    port = free_port()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        # As asyncio sets it for the proxy, and every bundled server for itself.
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", port))
        listener.listen()
        with socket.create_connection(("127.0.0.1", port)) as client:
            accepted, _ = listener.accept()
            accepted.close()  # the server side closes first
            assert client.recv(1) == b""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as plain, pytest.raises(
        OSError
    ):
        plain.bind(("127.0.0.1", port))  # what the probe used to do
    assert ports._is_free(port) is True


def test_the_preferred_port_is_usable_and_not_a_well_known_default():
    port = ports.PREFERRED["public"]
    assert port in ports.PORT_RANGE  # _is_free rejects anything outside it
    assert port not in {15672, 16379, 17500, 18080, 19132, 25565, 27017, 28015}


def test_secrets_are_generated_once(bundle: Bundle, data: DataDir):
    first = settings.ensure_secrets(bundle, data)
    assert first == settings.ensure_secrets(bundle, data)
    assert len(first["AUTOGPT_FRONTEND_DB_PASSWORD"]) >= 32
    assert {"POSTGRES_PASSWORD", "ENCRYPTION_KEY", "BETTER_AUTH_SECRET"} <= set(first)


def test_backend_uses_persisted_internal_key_without_exposing_it_to_frontend(
    bundle: Bundle, data: DataDir
):
    secret = settings.ensure_secrets(bundle, data)
    port = ports.allocate(data.ports_file)
    backend = settings.backend_environment(
        bundle, data, port, secret, {"INTERNAL_API_KEY": "user-setting-must-not-win"}
    )
    frontend = settings.frontend_environment(backend, port, secret, data)

    assert backend["INTERNAL_API_KEY"] == secret["INTERNAL_API_KEY"]
    assert len(backend["INTERNAL_API_KEY"]) >= 32
    assert "INTERNAL_API_KEY" not in frontend


def test_v1_profile_upgrade_preserves_existing_secrets_and_internal_key_on_restart(
    bundle: Bundle, data: DataDir
):
    before = settings.ensure_secrets(bundle, data)
    legacy = "\n".join(
        line.replace(
            "AUTOGPT_RUNTIME_CONFIG_VERSION=2", "AUTOGPT_RUNTIME_CONFIG_VERSION=1"
        )
        for line in data.runtime_env.read_text().splitlines()
        if not line.startswith("INTERNAL_API_KEY=")
    )
    data.runtime_env.write_text(legacy + "\n")

    upgraded = settings.ensure_secrets(bundle, data)
    restarted = settings.ensure_secrets(bundle, data)
    assert upgraded == restarted
    assert upgraded["AUTOGPT_RUNTIME_CONFIG_VERSION"] == "2"
    assert {k: v for k, v in before.items() if k != "INTERNAL_API_KEY"} == {
        k: v for k, v in upgraded.items() if k != "INTERNAL_API_KEY"
    }
    backend = settings.backend_environment(
        bundle, data, ports.allocate(data.ports_file), restarted, {}
    )
    assert backend["INTERNAL_API_KEY"] == upgraded["INTERNAL_API_KEY"]


def test_environment_is_wired_to_the_allocated_ports(bundle: Bundle, data: DataDir):
    secret = settings.ensure_secrets(bundle, data)
    port = ports.allocate(data.ports_file)
    env = settings.backend_environment(bundle, data, port, secret, {})

    public = f"http://127.0.0.1:{port['public']}"
    assert env["AUTOGPT_PUBLIC_URL"] == env["BETTER_AUTH_URL"] == public
    assert env["BACKEND_CORS_ALLOW_ORIGINS"] == json.dumps([public])
    assert f":{port['postgres']}/postgres?schema=platform" in env["DATABASE_URL"]
    assert env["REDIS_CLUSTER_PORT"] == str(port["valkey"])
    assert env["RABBITMQ_PORT"] == str(port["rabbitmq"])
    assert env["AGENT_API_HOST"] == env["WEBSOCKET_SERVER_HOST"] == "127.0.0.1"
    assert env["FORCE_FLAG_GRAPHITI_MEMORY"] == "false"
    assert env["JWT_JWKS_URL"].startswith(f"http://127.0.0.1:{port['frontend']}/")


def test_user_settings_cannot_override_the_runtime_wiring(
    bundle: Bundle, data: DataDir
):
    secret = settings.ensure_secrets(bundle, data)
    port = ports.allocate(data.ports_file)
    user = {"OPEN_ROUTER_API_KEY": "sk-user", "DATABASE_URL": "postgresql://elsewhere"}
    env = settings.backend_environment(bundle, data, port, secret, user)
    assert env["OPEN_ROUTER_API_KEY"] == "sk-user"
    assert "elsewhere" not in env["DATABASE_URL"]


@pytest.mark.parametrize("override", [None, "", "outside-cache"])
def test_tokenizer_cache_follows_the_bundle_after_relocation(
    bundle: Bundle, data: DataDir, monkeypatch, override
):
    secret = settings.ensure_secrets(bundle, data)
    port = {name: 20000 + index for index, name in enumerate(ports.PORT_NAMES)}
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", "ambient-cache")
    monkeypatch.setenv("DATA_GYM_CACHE_DIR", "ambient-legacy-cache")
    user = {} if override is None else {"TIKTOKEN_CACHE_DIR": override}
    user["DATA_GYM_CACHE_DIR"] = "user-legacy-cache"
    defaults = {"TIKTOKEN_CACHE_DIR": "profile-cache"}

    for root in (bundle.root, bundle.root.parent / "Moved AutoGPT's runtime"):
        relocated = Bundle(root)
        env = settings.backend_environment(
            relocated, data, port, secret, user, defaults
        )
        expected = str(root / "assets" / "tiktoken")
        assert env["TIKTOKEN_CACHE_DIR"] == expected
        host_env = settings.claude_code_host_environment(
            env, data, {}, data.root / "host-home"
        )
        assert host_env["TIKTOKEN_CACHE_DIR"] == expected
        assert "TIKTOKEN_CACHE_DIR" not in settings.frontend_environment(
            env, port, secret, data
        )


def test_registration_is_open_until_an_owner_exists_or_the_user_decides(
    bundle: Bundle, data: DataDir
):
    """The gate is set once the accounts are counted (Stack.secure_owner,
    tests/test_owner.py); the frontend gets whatever it was set to."""
    secret = settings.ensure_secrets(bundle, data)
    port = ports.allocate(data.ports_file)

    def allowed(user: dict[str, str], identities: int) -> str:
        backend = settings.backend_environment(bundle, data, port, secret, user)
        backend["AUTH_ALLOW_NEW_ACCOUNTS"] = settings.registration_gate(
            backend.get("AUTH_ALLOW_NEW_ACCOUNTS"), identities
        )
        frontend = settings.frontend_environment(backend, port, secret, data)
        assert frontend["AUTH_ALLOW_NEW_ACCOUNTS"] == backend["AUTH_ALLOW_NEW_ACCOUNTS"]
        return backend["AUTH_ALLOW_NEW_ACCOUNTS"]

    assert allowed({}, identities=0) == "true"
    assert allowed({}, identities=1) == "false"
    assert allowed({"AUTH_ALLOW_NEW_ACCOUNTS": "false"}, identities=1) == "false"
    assert allowed({"AUTH_ALLOW_NEW_ACCOUNTS": "true"}, identities=1) == "true"
    # Closed before anybody exists, the install could never get its owner.
    assert allowed({"AUTH_ALLOW_NEW_ACCOUNTS": "false"}, identities=0) == "true"


def test_services_get_a_home_of_their_own(bundle: Bundle, data: DataDir):
    """AutoPilot's coding agents keep their sign-in under the home directory;
    the app must not find the user's personal one there."""
    secret = settings.ensure_secrets(bundle, data)
    port = ports.allocate(data.ports_file)
    backend = settings.backend_environment(bundle, data, port, secret, {})
    frontend = settings.frontend_environment(backend, port, secret, data)

    assert backend["HOME"] == str(data.home)
    assert backend["CLAUDE_CONFIG_DIR"] == str(data.home / ".claude")
    assert backend["CODEX_HOME"] == str(data.home / ".codex")
    assert frontend["HOME"] == str(data.frontend_home)
    if sys.platform == "win32":  # where HOME is ignored
        assert backend["USERPROFILE"] == str(data.home)
        assert frontend["USERPROFILE"] == str(data.frontend_home)


@pytest.mark.skipif(
    sys.platform == "win32", reason="the user profile's ACL covers this"
)
def test_files_holding_secrets_are_readable_by_their_owner_only(data: DataDir):
    settings.read_user_settings(data)
    valkey.write_config(data, 20000, 20001, "secret" * 8)
    rabbitmq.prepare(data, 20002, "autogpt", "secret" * 8)

    def mode(path: Path) -> int:
        return stat.S_IMODE(path.stat().st_mode)

    assert mode(data.root) == 0o700
    assert mode(data.config / "settings.env") == 0o600
    assert mode(data.valkey / valkey.CONFIG_NAME) == 0o600
    assert mode(data.rabbitmq / "rabbitmq.conf") == 0o600


def test_user_settings_file_is_created_and_blank_values_are_ignored(data: DataDir):
    assert settings.read_user_settings(data) == {}
    path = data.config / "settings.env"
    path.write_text(
        "OPEN_ROUTER_API_KEY=sk-1\nOPENAI_API_KEY=\n# note\n", encoding="utf-8"
    )
    assert settings.read_user_settings(data) == {"OPEN_ROUTER_API_KEY": "sk-1"}


def test_frontend_gets_its_own_database_role_and_no_backend_secrets(
    bundle: Bundle, data: DataDir
):
    secret = settings.ensure_secrets(bundle, data)
    port = ports.allocate(data.ports_file)
    backend = settings.backend_environment(bundle, data, port, secret, {})
    frontend = settings.frontend_environment(backend, port, secret, data)
    assert frontend["DATABASE_URL"].startswith("postgresql://autogpt_frontend:")
    assert frontend["PORT"] == str(port["frontend"])
    assert frontend["HOSTNAME"] == "127.0.0.1"
    for leaked in ("ENCRYPTION_KEY", "POSTGRES_PASSWORD", "REDIS_PASSWORD", "DB_PASS"):
        assert leaked not in frontend


def test_valkey_config_is_a_loopback_single_node_cluster(data: DataDir):
    valkey.write_config(data, 20000, 20001, "secret" * 8)
    config = (data.valkey / valkey.CONFIG_NAME).read_text()
    assert "cluster-enabled yes" in config
    assert "bind 127.0.0.1" in config
    assert "cluster-port 20001" in config
    assert "dir ./" in config


def test_lock_export_pins_main_dependencies_only():
    lines = lock_export.export(DESKTOP.parent / "backend" / "poetry.lock", "linux")
    names = {line.split("==")[0] for line in lines}
    assert {"fastapi", "prisma", "redis", "aio-pika"} <= names
    assert "autogpt-libs" not in names  # a path dependency, installed separately
    assert "pyright" not in names  # dev-only
    assert all(
        "==" in line or re.search(r" @ git\+https://[^\s]+@[a-f0-9]{40}(?: ; |$)", line)
        for line in lines
    )


def test_lock_export_gives_every_platform_the_locks_versions():
    """A platform that cannot install the locked version of a package is
    not given another one (tests/test_claude_cli_build.py has the one case
    there was)."""
    lock = DESKTOP.parent / "backend" / "poetry.lock"
    pins = [
        {line.split(" ; ")[0] for line in lock_export.export(lock, platform)}
        for platform in ("win32", "darwin", "linux")
    ]
    assert pins[0] == pins[1] == pins[2]


def test_frontend_role_policy_is_extracted_from_the_appliance_bootstrap():
    sql = frontend_role_sql.extract(
        DESKTOP.parent / "single-container" / "bootstrap.sh"
    )
    assert sql.lstrip().startswith("BEGIN;")
    assert sql.rstrip().endswith("COMMIT;")
    assert (
        'GRANT SELECT (id, email), INSERT (id, email, name, "updatedAt"), '
        'UPDATE (email, "updatedAt") ON TABLE platform."User" TO autogpt_frontend;'
    ) in " ".join(sql.split())


def test_frontend_role_extraction_fails_loudly_when_the_layout_changes(tmp_path: Path):
    script = tmp_path / "bootstrap.sh"
    script.write_text("configure_frontend_database_role() {\n  psql -f role.sql\n}\n")
    with pytest.raises(ValueError):
        frontend_role_sql.extract(script)
