import pytest

from backend.api import rest_api


@pytest.mark.parametrize("enabled", [True, False])
def test_metrics_get_a_listener_of_their_own_only_when_enabled(
    enabled: bool, mocker, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ENABLE_METRICS", "true" if enabled else "false")
    # run() turns asyncio debug on under a local app_env; restored afterwards.
    monkeypatch.setenv("PYTHONASYNCIODEBUG", "")
    serve_api = mocker.patch("backend.api.rest_api.uvicorn.run")
    serve_metrics = mocker.patch("backend.monitoring.instrumentation.start_http_server")

    rest_api.AgentServer().run()

    config = rest_api.settings.config
    assert serve_api.call_args.kwargs["port"] == config.agent_api_port
    if not enabled:
        serve_metrics.assert_not_called()
        return
    serve_metrics.assert_called_once_with(
        config.agent_api_metrics_port, addr=config.pyro_host
    )
    assert config.agent_api_metrics_port != config.agent_api_port
