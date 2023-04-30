import json
import urllib
from unittest import mock
from unittest.mock import MagicMock

import pytest

from autogpt.event_dispatcher import fire, fire_and_forget


@pytest.fixture
def mock_cfg():
    cfg_mock = MagicMock()
    cfg_mock.event_dispatcher_protocol = "http"
    cfg_mock.event_dispatcher_host = "localhost"
    cfg_mock.event_dispatcher_port = 8080
    cfg_mock.event_dispatcher_endpoint = "/endpoint"
    cfg_mock.debug_mode = True
    return cfg_mock


def test_fire_and_forget(mock_cfg, monkeypatch, mocker):
    mock_thread_class = mocker.MagicMock()
    monkeypatch.setattr("autogpt.event_dispatcher.CFG", mock_cfg)
    monkeypatch.setattr("urllib.request.urlopen", MagicMock())
    monkeypatch.setattr("time.time_ns", lambda: 1234567890)
    monkeypatch.setattr("threading.Thread", mock_thread_class)

    data = {"key": "value"}
    headers = {
        "Content-type": "application/json",
        "Event-time": "1234567890",
        "Event-origin": "AutoGPT",
    }
    expected_url = "http://localhost:8080/endpoint"
    fire_and_forget(expected_url, data, headers)
    mock_thread_class.assert_called_once_with(target=mock.ANY)


def test_fire_and_forget_with_exception(mock_cfg, monkeypatch, mocker):
    mock_thread_class = mocker.MagicMock()
    monkeypatch.setattr("autogpt.event_dispatcher.CFG", mock_cfg)
    monkeypatch.setattr("time.time_ns", lambda: 1234567890)
    monkeypatch.setattr("threading.Thread", mock_thread_class)

    data = {"key": "value"}
    headers = {
        "Content-type": "application/json",
        "Event-time": "1234567890",
        "Event-origin": "AutoGPT",
    }
    expected_url = "http://localhost:8080/endpoint"
    with mock.patch("urllib.request.urlopen", side_effect=Exception()):
        fire_and_forget(expected_url, data, headers)
        mock_thread_class.assert_called_once_with(target=mock.ANY)


def test_fire(monkeypatch):
    mock_fire_and_forget = MagicMock()
    monkeypatch.setattr(
        "autogpt.event_dispatcher.fire_and_forget", mock_fire_and_forget
    )

    data = {"key": "value"}
    fire(data)

    mock_fire_and_forget.assert_called_once()
