"""Desktop integration with the shared runtime configuration generator."""

from pathlib import Path

from autogpt_desktop import settings
from autogpt_desktop.layout import Bundle, DataDir


def test_upstreams_runtime_config_runs_unmodified_on_this_system(tmp_path: Path):
    """The real module, the real OS: Windows, macOS and Linux each run this."""
    data = DataDir(tmp_path / "data")
    data.prepare()
    bundle = Bundle(tmp_path / "no-bundle")
    first = settings.ensure_secrets(bundle, data)
    assert first == settings.ensure_secrets(bundle, data)
    assert data.runtime_env.read_text(encoding="ascii").count("\n") > 5
    leftovers = [
        path.name for path in data.config.iterdir() if path.name.endswith(".tmp")
    ]
    assert not leftovers
