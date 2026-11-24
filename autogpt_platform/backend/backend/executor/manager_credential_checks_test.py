"""A node's credentials must be of a provider and type its field takes, and a
required field whose credentials are gone fails as not found."""

from typing import Literal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.blocks._base import AnyBlockSchema, BlockSchemaInput
from backend.blocks.airtable.bases import AirtableListBasesBlock
from backend.blocks.github.issues import GithubCommentBlock
from backend.data.execution import ExecutionContext, ExecutionStatus, NodeExecutionEntry
from backend.data.graph import Node
from backend.data.model import (
    APIKeyCredentials,
    Credentials,
    CredentialsField,
    CredentialsMetaInput,
    NodeExecutionStats,
    UserPasswordCredentials,
)
from backend.executor.manager import ExecutionProcessor
from backend.executor.utils import NodeExecutionProgress
from backend.integrations.providers import ProviderName

GITHUB_INPUTS = {"issue_url": "https://github.com/o/r/issues/1", "comment": "hi"}


@pytest.mark.parametrize(
    "named, stored, runs",
    [
        (("airtable", "api_key"), APIKeyCredentials, True),
        (("github", "api_key"), APIKeyCredentials, False),
        (("airtable", "user_password"), UserPasswordCredentials, False),
    ],
    ids=["its-own", "another-provider", "another-type"],
)
@pytest.mark.asyncio
async def test_an_sdk_blocks_credentials_must_be_of_its_provider_and_type(
    named: tuple[str, str], stored: type[Credentials], runs: bool
):
    """AirtableListBasesBlock's field is an untyped CredentialsMetaInput, which
    takes any provider and type."""
    provider, cred_type = named
    if stored is APIKeyCredentials:
        key: Credentials = APIKeyCredentials(
            id="cred-1", provider=provider, api_key=SecretStr("k"), title="Key"
        )
    else:
        key = UserPasswordCredentials(
            id="cred-1",
            provider=provider,
            username=SecretStr("u"),
            password=SecretStr("p"),
            title="Login",
        )
    inputs = {"credentials": {"id": "cred-1", "provider": provider, "type": cred_type}}

    run, stats, status = await _execute(AirtableListBasesBlock(), inputs, key)

    if runs:
        run.assert_called_once()
        assert run.call_args.kwargs["credentials"] is key
    else:
        assert status == ExecutionStatus.FAILED
        assert "in field 'credentials' are not of a provider and type" in str(
            stats.error
        )
        run.assert_not_called()


@pytest.mark.parametrize(
    "block, runs",
    [
        (GithubCommentBlock(), False),
        # A field with a default runs without credentials it can't load.
        (None, True),
    ],
    ids=["required-field", "field-with-default"],
)
@pytest.mark.asyncio
async def test_a_required_field_whose_credentials_are_gone_fails_as_not_found(
    block: AnyBlockSchema | None, runs: bool
):
    block = block or _defaulted_key_block()
    inputs = {
        "credentials": {"id": "gone", "provider": "github", "type": "api_key"},
        **GITHUB_INPUTS,
    }

    run, stats, status = await _execute(block, inputs, None)

    if runs:
        run.assert_called_once()
        assert "credentials" not in run.call_args.kwargs
    else:
        assert status == ExecutionStatus.FAILED
        assert "Credentials #gone for user #user-1 not found" in str(stats.error)
        run.assert_not_called()


class _DefaultedKeyInput(BlockSchemaInput):
    credentials: CredentialsMetaInput[
        Literal[ProviderName.GITHUB], Literal["api_key"]
    ] = CredentialsField(description="GitHub key", default={})
    issue_url: str
    comment: str


def _defaulted_key_block() -> GithubCommentBlock:
    """One instance whose key field has a default. (A Block subclass would join
    every block list in the process.)"""
    block = GithubCommentBlock()
    block.input_schema = _DefaultedKeyInput
    return block


async def _execute(
    block: AnyBlockSchema, inputs: dict, stored: Credentials | None
) -> tuple[AsyncMock, NodeExecutionStats, ExecutionStatus]:
    """Run one node through the executor, with the stored credential and the
    block's ``run`` stubbed."""

    async def _run(*_args, **_kwargs):
        yield "done", True

    run = MagicMock(side_effect=_run)
    processor = ExecutionProcessor()
    processor.creds_manager = MagicMock(get=AsyncMock(return_value=stored))
    stats = NodeExecutionStats()
    db_client = MagicMock(
        upsert_execution_output=AsyncMock(),
        get_node_execution=AsyncMock(return_value=None),
    )
    with (
        patch("backend.data.graph.get_block", return_value=block),
        patch("backend.executor.utils.get_block", return_value=block),
        patch.object(type(block), "run", run),
        patch(
            "backend.executor.manager.async_update_node_execution_status",
            new=AsyncMock(),
        ),
    ):
        _, status = await processor._on_node_execution(
            node=Node(id="node-1", block_id=block.id, input_default=inputs),
            node_exec=NodeExecutionEntry(
                user_id="user-1",
                graph_exec_id="graph-exec-1",
                graph_id="graph-1",
                graph_version=1,
                node_exec_id="node-exec-1",
                node_id="node-1",
                block_id=block.id,
                inputs=dict(inputs),
                execution_context=ExecutionContext(root_execution_id="graph-exec-1"),
            ),
            node_exec_progress=NodeExecutionProgress(),
            stats=stats,
            db_client=db_client,
            log_metadata=MagicMock(),
        )
    return run, stats, status
