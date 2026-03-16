"""Tests for format_detector.py."""

from .format_detector import detect_format
from .models import SourcePlatform


class TestDetectFormat:
    def test_n8n_workflow(self):
        data = {
            "name": "My n8n Workflow",
            "nodes": [
                {
                    "name": "Webhook",
                    "type": "n8n-nodes-base.webhook",
                    "parameters": {"path": "/hook"},
                },
                {
                    "name": "HTTP Request",
                    "type": "n8n-nodes-base.httpRequest",
                    "parameters": {"url": "https://api.example.com"},
                },
            ],
            "connections": {
                "Webhook": {
                    "main": [[{"node": "HTTP Request", "type": "main", "index": 0}]]
                }
            },
        }
        assert detect_format(data) == SourcePlatform.N8N

    def test_n8n_langchain_nodes(self):
        data = {
            "nodes": [
                {
                    "name": "Agent",
                    "type": "@n8n/n8n-nodes-langchain.agent",
                    "parameters": {},
                },
            ],
            "connections": {},
        }
        assert detect_format(data) == SourcePlatform.N8N

    def test_make_scenario(self):
        data = {
            "name": "My Make Scenario",
            "flow": [
                {
                    "module": "google-sheets:watchUpdatedCells",
                    "mapper": {"spreadsheetId": "123"},
                },
                {
                    "module": "google-calendar:createAnEvent",
                    "mapper": {"title": "Test"},
                },
            ],
        }
        assert detect_format(data) == SourcePlatform.MAKE

    def test_zapier_zap(self):
        data = {
            "name": "My Zap",
            "steps": [
                {"app": "gmail", "action": "new_email"},
                {
                    "app": "slack",
                    "action": "send_message",
                    "params": {"channel": "#general"},
                },
            ],
        }
        assert detect_format(data) == SourcePlatform.ZAPIER

    def test_unknown_format(self):
        data = {"foo": "bar", "nodes": []}
        assert detect_format(data) == SourcePlatform.UNKNOWN

    def test_empty_dict(self):
        assert detect_format({}) == SourcePlatform.UNKNOWN

    def test_autogpt_graph_not_detected_as_n8n(self):
        """AutoGPT graphs have nodes but not n8n-style types."""
        data = {
            "nodes": [
                {"id": "abc", "block_id": "some-uuid", "input_default": {}},
            ],
            "connections": {},
        }
        assert detect_format(data) == SourcePlatform.UNKNOWN

    def test_make_without_colon_not_detected(self):
        data = {
            "flow": [{"module": "simplemodule", "mapper": {}}],
        }
        assert detect_format(data) == SourcePlatform.UNKNOWN

    def test_zapier_without_action_not_detected(self):
        data = {
            "steps": [{"app": "gmail"}],
        }
        assert detect_format(data) == SourcePlatform.UNKNOWN
