"""Tests for describers.py."""

import pytest

from .describers import (
    describe_make_workflow,
    describe_n8n_workflow,
    describe_workflow,
    describe_zapier_workflow,
)
from .models import CompetitorFormat


class TestDescribeN8nWorkflow:
    def test_basic_workflow(self):
        data = {
            "name": "Email on Webhook",
            "nodes": [
                {
                    "name": "Webhook",
                    "type": "n8n-nodes-base.webhookTrigger",
                    "parameters": {"path": "/incoming"},
                },
                {
                    "name": "Send Email",
                    "type": "n8n-nodes-base.gmail",
                    "parameters": {"resource": "message", "operation": "send"},
                },
            ],
            "connections": {
                "Webhook": {
                    "main": [[{"node": "Send Email", "type": "main", "index": 0}]]
                }
            },
        }
        desc = describe_n8n_workflow(data)
        assert desc.name == "Email on Webhook"
        assert desc.source_format == CompetitorFormat.N8N
        assert len(desc.steps) == 2
        assert desc.steps[0].connections_to == [1]
        assert desc.steps[1].connections_to == []
        assert desc.trigger_type is not None

    def test_step_extraction(self):
        data = {
            "name": "Test",
            "nodes": [
                {
                    "name": "HTTP",
                    "type": "n8n-nodes-base.httpRequest",
                    "parameters": {"url": "https://example.com", "method": "GET"},
                },
            ],
            "connections": {},
        }
        desc = describe_n8n_workflow(data)
        step = desc.steps[0]
        assert step.service == "Http Request"
        assert step.order == 0
        assert "url" in step.parameters

    def test_empty_nodes(self):
        data = {"name": "Empty", "nodes": [], "connections": {}}
        desc = describe_n8n_workflow(data)
        assert len(desc.steps) == 0
        assert desc.trigger_type is None


class TestDescribeMakeWorkflow:
    def test_basic_scenario(self):
        data = {
            "name": "Sheets to Calendar",
            "flow": [
                {
                    "module": "google-sheets:watchUpdatedCells",
                    "mapper": {"spreadsheetId": "abc"},
                },
                {
                    "module": "google-calendar:createAnEvent",
                    "mapper": {"title": "Meeting"},
                },
            ],
        }
        desc = describe_make_workflow(data)
        assert desc.name == "Sheets to Calendar"
        assert desc.source_format == CompetitorFormat.MAKE
        assert len(desc.steps) == 2
        # Sequential: step 0 connects to step 1
        assert desc.steps[0].connections_to == [1]
        assert desc.steps[1].connections_to == []
        assert desc.trigger_type is not None  # "watch" in module name

    def test_service_extraction(self):
        data = {
            "flow": [{"module": "slack:sendMessage", "mapper": {"text": "hello"}}],
        }
        desc = describe_make_workflow(data)
        assert desc.steps[0].service == "Slack"


class TestDescribeZapierWorkflow:
    def test_basic_zap(self):
        data = {
            "name": "Gmail to Slack",
            "steps": [
                {"app": "Gmail", "action": "new_email"},
                {
                    "app": "Slack",
                    "action": "send_message",
                    "params": {"channel": "#alerts"},
                },
            ],
        }
        desc = describe_zapier_workflow(data)
        assert desc.name == "Gmail to Slack"
        assert desc.source_format == CompetitorFormat.ZAPIER
        assert len(desc.steps) == 2
        assert desc.steps[0].connections_to == [1]
        assert desc.trigger_type == "Gmail"


class TestDescribeWorkflowRouter:
    def test_routes_to_n8n(self):
        data = {
            "nodes": [
                {"name": "N", "type": "n8n-nodes-base.webhook", "parameters": {}}
            ],
            "connections": {},
        }
        desc = describe_workflow(data, CompetitorFormat.N8N)
        assert desc.source_format == CompetitorFormat.N8N

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="No describer"):
            describe_workflow({}, CompetitorFormat.UNKNOWN)
