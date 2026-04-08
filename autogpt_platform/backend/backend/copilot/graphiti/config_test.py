import pytest

from .config import GraphitiConfig

_ENV_VARS_TO_CLEAR = (
    "GRAPHITI_FALKORDB_HOST",
    "GRAPHITI_FALKORDB_PORT",
    "GRAPHITI_FALKORDB_PASSWORD",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _ENV_VARS_TO_CLEAR:
        monkeypatch.delenv(var, raising=False)


def test_graphiti_config_reads_backend_env_defaults() -> None:
    cfg = GraphitiConfig()

    assert cfg.falkordb_host == "localhost"
    assert cfg.falkordb_port == 6380
