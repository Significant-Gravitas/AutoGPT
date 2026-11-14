import importlib
import json
import sys
import types
from pathlib import Path

from autogpt_desktop import apps, ports, settings
from autogpt_desktop.layout import Bundle, DataDir


def test_database_services_are_collected_separately(monkeypatch):
    collected = []

    class DatabaseManager:
        pass

    class SwapCredentialService:
        def set_log_level(self, level):
            self.level = level
            return self

    for name, exports in {
        "backend": {},
        "backend.app": {"run_processes": lambda *services: collected.extend(services)},
        "backend.data": {},
        "backend.data.db_manager": {"DatabaseManager": DatabaseManager},
        "backend.copilot": {},
        "backend.copilot.swap_service": {
            "SwapCredentialService": SwapCredentialService
        },
    }.items():
        module = types.ModuleType(name)
        module.__dict__.update(exports)
        monkeypatch.setitem(sys.modules, name, module)
    for name, expected in (
        ("database-manager", DatabaseManager),
        ("swap-credentials", SwapCredentialService),
    ):
        service = apps.service(name)
        module, _, entry = service.entry.partition(":")
        getattr(importlib.import_module(module), entry)()
        assert len(collected) == 1
        assert isinstance(collected.pop(), expected)
        group = next(group for group in apps.MERGED if name in group.services)
        assert group.services == (name,)
        assert service.database
        assert name in apps.NEEDED_TO_OPEN


def test_upgraded_profile_keeps_ports_and_wires_private_services(tmp_path: Path):
    data = DataDir(tmp_path / "profile")
    data.prepare()
    old_names = tuple(name for name in ports.PORT_NAMES if name != "swap_credential")
    before = ports.allocate(data.ports_file, old_names)
    after = ports.allocate(data.ports_file)
    assert {name: after[name] for name in before} == before
    assert after["swap_credential"] not in before.values()
    assert json.loads(data.ports_file.read_text()) == after
    bundle = Bundle(tmp_path / "runtime")
    secret = settings.ensure_secrets(bundle, data)
    backend = settings.backend_environment(
        bundle,
        data,
        after,
        secret,
        {"AUTOGPT_DESKTOP": "false", "TRUSTED_PROXY_COUNT": "9"},
    )
    assert backend["AUTOGPT_DESKTOP"] == "1"
    assert backend["TRUSTED_PROXY_COUNT"] == "1"
    assert backend["SWAP_CREDENTIAL_SERVICE_PORT"] == str(after["swap_credential"])
    assert backend["SWAPCREDENTIALSERVICE_HOST"] == "127.0.0.1"
    frontend = settings.frontend_environment(backend, after, secret, data)
    assert frontend["TRUST_X_FORWARDED_FOR"] == "true"
    assert "INTERNAL_API_KEY" not in frontend
