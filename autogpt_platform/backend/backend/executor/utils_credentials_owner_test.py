"""Tests for OWNER-mode credential resolution in the execution pipeline.

Covers the two enqueue-time pieces that decide WHOSE credential store the
graph's stored references resolve against:

- ``owner_referenced_credential_ids``: the per-node allowlist of ids the graph
  itself references (regular + auto), so OWNER mode can never look up an
  arbitrary id in the owner's store.
- ``_validate_node_input_credentials``: mirrors execution-time resolution —
  in OWNER mode the graph's own references validate against the owner's store,
  in CONSUMER mode against the executing user (regression).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.executor.utils import (
    _validate_node_input_credentials,
    owner_referenced_credential_ids,
)


def _input_schema(*, cred_fields=None, auto_fields=None, required=None):
    def _validate(value):
        meta = MagicMock()
        meta.id = value["id"]
        meta.provider = value.get("provider", "prov")
        meta.type = value.get("type", "api_key")
        return meta

    schema = MagicMock()
    fields = {}
    for name in cred_fields or []:
        cred_type = MagicMock()
        cred_type.model_validate.side_effect = _validate
        fields[name] = cred_type
    schema.get_credentials_fields.return_value = fields
    schema.get_auto_credentials_fields.return_value = auto_fields or {}
    schema.get_required_fields.return_value = set(required or [])
    return schema


def _node(*, node_id="n1", input_default=None, schema=None, optional=False):
    block = MagicMock()
    block.input_schema = schema
    node = MagicMock()
    node.id = node_id
    node.block = block
    node.input_default = input_default or {}
    node.credentials_optional = optional
    return node


class TestOwnerReferencedCredentialIds:
    def test_collects_regular_and_auto_ids(self):
        schema = _input_schema(
            cred_fields=["credentials"],
            auto_fields={
                "auto": {"field_name": "spreadsheet", "config": {"provider": "google"}}
            },
        )
        input_default = {
            "credentials": {"id": "regular-1", "provider": "p", "type": "api_key"},
            "spreadsheet": {"_credentials_id": "auto-1", "id": "file-1"},
            "unrelated": {"id": "not-a-cred"},
        }

        assert owner_referenced_credential_ids(input_default, schema) == {
            "regular-1",
            "auto-1",
        }

    def test_ignores_missing_or_malformed_references(self):
        schema = _input_schema(cred_fields=["credentials"])
        # No credential value baked in -> nothing to allow.
        assert owner_referenced_credential_ids({}, schema) == set()
        assert owner_referenced_credential_ids({"credentials": {}}, schema) == set()


class TestValidateNodeInputCredentialsOwnerMode:
    @pytest.fixture
    def mock_store(self, mocker):
        creds = MagicMock()
        creds.id = "owner-cred-1"
        creds.provider = "prov"
        creds.type = "api_key"
        store = MagicMock()
        store.get_creds_by_id = AsyncMock(return_value=creds)
        mocker.patch(
            "backend.executor.utils.get_integration_credentials_store",
            return_value=store,
        )
        return store

    def _graph_with_owner_cred(self):
        schema = _input_schema(cred_fields=["credentials"], required=["credentials"])
        node = _node(
            input_default={
                "credentials": {
                    "id": "owner-cred-1",
                    "provider": "prov",
                    "type": "api_key",
                }
            },
            schema=schema,
        )
        graph = MagicMock()
        graph.nodes = [node]
        return graph

    @pytest.mark.asyncio
    async def test_consumer_mode_uses_executing_user_store(self, mock_store):
        graph = self._graph_with_owner_cred()

        errors, skip = await _validate_node_input_credentials(
            graph, "consumer-1", None, None
        )

        assert errors == {}
        mock_store.get_creds_by_id.assert_awaited_once_with(
            "consumer-1", "owner-cred-1"
        )

    @pytest.mark.asyncio
    async def test_owner_mode_resolves_graph_reference_against_owner_store(
        self, mock_store
    ):
        graph = self._graph_with_owner_cred()

        errors, skip = await _validate_node_input_credentials(
            graph, "consumer-1", None, "owner-1"
        )

        assert errors == {}
        # The graph's baked reference resolves against the OWNER, not consumer.
        mock_store.get_creds_by_id.assert_awaited_once_with("owner-1", "owner-cred-1")

    @pytest.mark.asyncio
    async def test_owner_mode_ignores_consumer_supplied_mask(self, mock_store):
        graph = self._graph_with_owner_cred()
        # A consumer tries to substitute their own id via a mask; OWNER mode
        # must ignore it and use the owner's baked reference + owner store.
        masks = {
            "n1": {
                "credentials": {
                    "id": "consumer-injected",
                    "provider": "prov",
                    "type": "api_key",
                }
            }
        }

        await _validate_node_input_credentials(graph, "consumer-1", masks, "owner-1")

        mock_store.get_creds_by_id.assert_awaited_once_with("owner-1", "owner-cred-1")

    @pytest.mark.asyncio
    async def test_owner_credential_missing_records_error(self, mocker):
        store = MagicMock()
        store.get_creds_by_id = AsyncMock(return_value=None)  # owner cred gone
        mocker.patch(
            "backend.executor.utils.get_integration_credentials_store",
            return_value=store,
        )
        graph = self._graph_with_owner_cred()

        errors, skip = await _validate_node_input_credentials(
            graph, "consumer-1", None, "owner-1"
        )

        # Failing closed: the missing owner credential surfaces as an error,
        # never a silent pass that would fall back to the consumer at runtime.
        assert "n1" in errors
        assert "credentials" in errors["n1"]
        store.get_creds_by_id.assert_awaited_once_with("owner-1", "owner-cred-1")
