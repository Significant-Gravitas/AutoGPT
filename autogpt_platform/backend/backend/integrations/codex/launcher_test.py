import os

from backend.integrations.codex.launcher import sanitized_environment


def test_launcher_drops_ambient_backend_secrets(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://secret")
    monkeypatch.setenv("ENCRYPTION_KEY", "encryption-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "provider-secret")
    monkeypatch.setenv("CODEX_HOME", "/tmp/codex-user")
    monkeypatch.setenv("APPDATA", "/tmp/codex-user")
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))

    child_env = sanitized_environment()

    assert child_env["CODEX_HOME"] == "/tmp/codex-user"
    assert child_env["APPDATA"] == "/tmp/codex-user"
    assert "PATH" in child_env
    assert "DATABASE_URL" not in child_env
    assert "ENCRYPTION_KEY" not in child_env
    assert child_env["OPENAI_API_KEY"] == ""


def test_launcher_respects_an_explicit_empty_environment():
    child_env = sanitized_environment({})

    assert child_env == {
        "CODEX_ACCESS_TOKEN": "",
        "CODEX_API_KEY": "",
        "OPENAI_API_KEY": "",
    }
