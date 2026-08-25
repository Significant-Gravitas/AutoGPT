"""Tests for OWNER-mode credential resolution in the execution pipeline.

Covers the two enqueue-time pieces that decide WHOSE credential store the
graph's stored references resolve against:

- ``owner_referenced_credentials``: the per-node field-bound map the graph
  itself references (regular + auto), so OWNER mode can never look up an
  arbitrary id in the owner's store.
- ``_validate_node_input_credentials``: mirrors execution-time resolution —
  in OWNER mode the graph's own references validate against the owner's store,
  in CONSUMER mode against the executing user (regression).
"""

from unittest.mock import AsyncMock, MagicMock, call

import pytest

from backend.executor.utils import (
    _validate_node_input_credentials,
    owner_referenced_credentials,
)


def _input_schema(
    *,
    cred_fields=None,
    auto_fields=None,
    required=None,
    reference_only=None,
    discriminators=None,
):
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
    field_infos = {}
    for name in fields:
        field_info = MagicMock()
        field_info.credential_reference_only = name in set(reference_only or [])
        discriminator = (discriminators or {}).get(name)
        if discriminator:
            discriminator_name, credential_values = discriminator
            field_info.discriminator = discriminator_name
            field_info.requires_credentials.side_effect = (
                lambda value, values=set(credential_values): value in values
            )
        else:
            field_info.discriminator = None
            field_info.requires_credentials.return_value = True
        field_infos[name] = field_info
    schema.get_credentials_fields_info.return_value = field_infos
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


class TestOwnerReferencedCredentials:
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

        assert owner_referenced_credentials(input_default, schema) == {
            "credentials": "regular-1",
            "spreadsheet": "auto-1",
        }

    def test_ignores_missing_or_malformed_references(self):
        schema = _input_schema(cred_fields=["credentials"])
        # No credential value baked in -> nothing to allow.
        assert owner_referenced_credentials({}, schema) == {}
        assert owner_referenced_credentials({"credentials": {}}, schema) == {}
        assert (
            owner_referenced_credentials({"credentials": {"id": "   "}}, schema) == {}
        )


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

    @pytest.mark.asyncio
    async def test_optional_configured_owner_credential_is_validated(self, mock_store):
        schema = _input_schema(cred_fields=["credentials"], required=[])
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
        graph = MagicMock(nodes=[node])

        errors, _ = await _validate_node_input_credentials(
            graph, "consumer-1", None, "owner-1"
        )

        assert errors == {}
        mock_store.get_creds_by_id.assert_awaited_once_with("owner-1", "owner-cred-1")

    @pytest.mark.asyncio
    async def test_reference_only_owner_credential_fails_closed(self, mock_store):
        schema = _input_schema(
            cred_fields=["credentials"],
            required=["credentials"],
            reference_only=["credentials"],
        )
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
        graph = MagicMock(nodes=[node])

        errors, _ = await _validate_node_input_credentials(
            graph, "consumer-1", None, "owner-1"
        )

        assert "runtime-managed credential references" in errors["n1"]["credentials"]
        mock_store.get_creds_by_id.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_inactive_discriminated_owner_credential_stays_inactive(
        self, mock_store
    ):
        schema = _input_schema(
            cred_fields=["credentials"],
            required=[],
            reference_only=["credentials"],
            discriminators={"credentials": ("transport", {"codex"})},
        )
        node = _node(
            input_default={
                "transport": "platform",
                "credentials": {
                    "id": "stale-owner-cred",
                    "provider": "codex",
                    "type": "oauth2",
                },
            },
            schema=schema,
        )

        errors, _ = await _validate_node_input_credentials(
            MagicMock(nodes=[node]), "consumer-1", None, "owner-1"
        )

        assert errors == {}
        mock_store.get_creds_by_id.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_consumer_activating_discriminated_owner_reference_fails_closed(
        self, mock_store
    ):
        schema = _input_schema(
            cred_fields=["credentials"],
            required=[],
            reference_only=["credentials"],
            discriminators={"credentials": ("transport", {"codex"})},
        )
        node = _node(
            input_default={
                "transport": "platform",
                "credentials": {
                    "id": "stale-owner-cred",
                    "provider": "codex",
                    "type": "oauth2",
                },
            },
            schema=schema,
        )

        errors, _ = await _validate_node_input_credentials(
            MagicMock(nodes=[node]),
            "consumer-1",
            {"n1": {"transport": "codex"}},
            "owner-1",
        )

        assert "runtime-managed credential references" in errors["n1"]["credentials"]
        mock_store.get_creds_by_id.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_auto_credentials_are_bound_to_their_picker_field(self, mocker):
        schema = _input_schema(
            auto_fields={
                "first_creds": {
                    "field_name": "first_file",
                    "config": {"provider": "google"},
                },
                "second_creds": {
                    "field_name": "second_file",
                    "config": {"provider": "google"},
                },
            },
            required=["first_file", "second_file"],
        )
        node = _node(
            input_default={
                "first_file": {
                    "id": "owner-resource",
                    "_credentials_id": "shared-id",
                }
            },
            schema=schema,
        )
        graph = MagicMock(nodes=[node])
        owner_creds = MagicMock(provider="google")
        consumer_creds = MagicMock(provider="google")
        store = MagicMock()
        store.get_creds_by_id = AsyncMock(side_effect=[owner_creds, consumer_creds])
        mocker.patch(
            "backend.executor.utils.get_integration_credentials_store",
            return_value=store,
        )
        masks = {
            "n1": {
                "second_file": {
                    "id": "consumer-chosen-resource",
                    "_credentials_id": "shared-id",
                }
            }
        }

        errors, _ = await _validate_node_input_credentials(
            graph, "consumer-1", masks, "owner-1"
        )

        assert errors == {}
        assert store.get_creds_by_id.await_args_list == [
            call("owner-1", "shared-id"),
            call("consumer-1", "shared-id"),
        ]
