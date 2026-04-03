from typing import Any, Type

import pytest

from backend.blocks import get_blocks
from backend.blocks._base import Block, BlockSchemaInput
from backend.blocks.io import AgentDropdownInputBlock, AgentInputBlock
from backend.data.graph import BaseGraph
from backend.data.model import SchemaField
from backend.util.test import execute_block_test

SKIP_BLOCK_TESTS = {
    "HumanInTheLoopBlock",
}


@pytest.mark.parametrize("block", get_blocks().values(), ids=lambda b: b().name)
async def test_available_blocks(block: Type[Block]):
    block_instance = block()
    if block_instance.__class__.__name__ in SKIP_BLOCK_TESTS:
        pytest.skip(
            f"Skipping {block_instance.__class__.__name__} - requires external service"
        )
    await execute_block_test(block_instance)


@pytest.mark.parametrize("block", get_blocks().values(), ids=lambda b: b().name)
async def test_block_ids_valid(block: Type[Block]):
    # add the tests here to check they are uuid4
    import uuid

    # Skip list for blocks with known invalid UUIDs
    skip_blocks = {
        "GetWeatherInformationBlock",
        "ExecuteCodeBlock",
        "CountdownTimerBlock",
        "TwitterGetListTweetsBlock",
        "TwitterRemoveListMemberBlock",
        "TwitterAddListMemberBlock",
        "TwitterGetListMembersBlock",
        "TwitterGetListMembershipsBlock",
        "TwitterUnfollowListBlock",
        "TwitterFollowListBlock",
        "TwitterUnpinListBlock",
        "TwitterPinListBlock",
        "TwitterGetPinnedListsBlock",
        "TwitterDeleteListBlock",
        "TwitterUpdateListBlock",
        "TwitterCreateListBlock",
        "TwitterGetListBlock",
        "TwitterGetOwnedListsBlock",
        "TwitterGetSpacesBlock",
        "TwitterGetSpaceByIdBlock",
        "TwitterGetSpaceBuyersBlock",
        "TwitterGetSpaceTweetsBlock",
        "TwitterSearchSpacesBlock",
        "TwitterGetUserMentionsBlock",
        "TwitterGetHomeTimelineBlock",
        "TwitterGetUserTweetsBlock",
        "TwitterGetTweetBlock",
        "TwitterGetTweetsBlock",
        "TwitterGetQuoteTweetsBlock",
        "TwitterLikeTweetBlock",
        "TwitterGetLikingUsersBlock",
        "TwitterGetLikedTweetsBlock",
        "TwitterUnlikeTweetBlock",
        "TwitterBookmarkTweetBlock",
        "TwitterGetBookmarkedTweetsBlock",
        "TwitterRemoveBookmarkTweetBlock",
        "TwitterRetweetBlock",
        "TwitterRemoveRetweetBlock",
        "TwitterGetRetweetersBlock",
        "TwitterHideReplyBlock",
        "TwitterUnhideReplyBlock",
        "TwitterPostTweetBlock",
        "TwitterDeleteTweetBlock",
        "TwitterSearchRecentTweetsBlock",
        "TwitterUnfollowUserBlock",
        "TwitterFollowUserBlock",
        "TwitterGetFollowersBlock",
        "TwitterGetFollowingBlock",
        "TwitterUnmuteUserBlock",
        "TwitterGetMutedUsersBlock",
        "TwitterMuteUserBlock",
        "TwitterGetBlockedUsersBlock",
        "TwitterGetUserBlock",
        "TwitterGetUsersBlock",
        "TodoistCreateLabelBlock",
        "TodoistListLabelsBlock",
        "TodoistGetLabelBlock",
        "TodoistUpdateLabelBlock",
        "TodoistDeleteLabelBlock",
        "TodoistGetSharedLabelsBlock",
        "TodoistRenameSharedLabelsBlock",
        "TodoistRemoveSharedLabelsBlock",
        "TodoistCreateTaskBlock",
        "TodoistGetTasksBlock",
        "TodoistGetTaskBlock",
        "TodoistUpdateTaskBlock",
        "TodoistCloseTaskBlock",
        "TodoistReopenTaskBlock",
        "TodoistDeleteTaskBlock",
        "TodoistListSectionsBlock",
        "TodoistGetSectionBlock",
        "TodoistDeleteSectionBlock",
        "TodoistCreateProjectBlock",
        "TodoistGetProjectBlock",
        "TodoistUpdateProjectBlock",
        "TodoistDeleteProjectBlock",
        "TodoistListCollaboratorsBlock",
        "TodoistGetCommentsBlock",
        "TodoistGetCommentBlock",
        "TodoistUpdateCommentBlock",
        "TodoistDeleteCommentBlock",
        "GithubListStargazersBlock",
        "Slant3DSlicerBlock",
    }

    block_instance = block()

    # Skip blocks with known invalid UUIDs
    if block_instance.__class__.__name__ in skip_blocks:
        pytest.skip(
            f"Skipping UUID check for {block_instance.__class__.__name__} - known invalid UUID"
        )

    # Check that the ID is not empty
    assert block_instance.id, f"Block {block.name} has empty ID"

    # Check that the ID is a valid UUID4
    try:
        parsed_uuid = uuid.UUID(block_instance.id)
        # Verify it's specifically UUID version 4
        assert (
            parsed_uuid.version == 4
        ), f"Block {block.name} ID is UUID version {parsed_uuid.version}, expected version 4"
    except ValueError:
        pytest.fail(f"Block {block.name} has invalid UUID format: {block_instance.id}")


class TestAutoCredentialsFieldsValidation:
    """Tests for auto_credentials field validation in BlockSchema."""

    def test_duplicate_auto_credentials_kwarg_name_raises_error(self):
        """Test that duplicate kwarg_name in auto_credentials raises ValueError."""

        class DuplicateKwargSchema(BlockSchemaInput):
            """Schema with duplicate auto_credentials kwarg_name."""

            # Both fields explicitly use the same kwarg_name "credentials"
            file1: dict[str, Any] | None = SchemaField(
                description="First file input",
                default=None,
                json_schema_extra={
                    "auto_credentials": {
                        "provider": "google",
                        "type": "oauth2",
                        "scopes": ["https://www.googleapis.com/auth/drive.file"],
                        "kwarg_name": "credentials",
                    }
                },
            )
            file2: dict[str, Any] | None = SchemaField(
                description="Second file input",
                default=None,
                json_schema_extra={
                    "auto_credentials": {
                        "provider": "google",
                        "type": "oauth2",
                        "scopes": ["https://www.googleapis.com/auth/drive.file"],
                        "kwarg_name": "credentials",  # Duplicate kwarg_name!
                    }
                },
            )

        with pytest.raises(ValueError) as exc_info:
            DuplicateKwargSchema.get_auto_credentials_fields()

        error_message = str(exc_info.value)
        assert "Duplicate auto_credentials kwarg_name 'credentials'" in error_message
        assert "file1" in error_message
        assert "file2" in error_message

    def test_unique_auto_credentials_kwarg_names_succeed(self):
        """Test that unique kwarg_name values work correctly."""

        class UniqueKwargSchema(BlockSchemaInput):
            """Schema with unique auto_credentials kwarg_name values."""

            file1: dict[str, Any] | None = SchemaField(
                description="First file input",
                default=None,
                json_schema_extra={
                    "auto_credentials": {
                        "provider": "google",
                        "type": "oauth2",
                        "scopes": ["https://www.googleapis.com/auth/drive.file"],
                        "kwarg_name": "file1_credentials",
                    }
                },
            )
            file2: dict[str, Any] | None = SchemaField(
                description="Second file input",
                default=None,
                json_schema_extra={
                    "auto_credentials": {
                        "provider": "google",
                        "type": "oauth2",
                        "scopes": ["https://www.googleapis.com/auth/drive.file"],
                        "kwarg_name": "file2_credentials",  # Different kwarg_name
                    }
                },
            )

        # Should not raise
        result = UniqueKwargSchema.get_auto_credentials_fields()

        assert "file1_credentials" in result
        assert "file2_credentials" in result
        assert result["file1_credentials"]["field_name"] == "file1"
        assert result["file2_credentials"]["field_name"] == "file2"

    def test_default_kwarg_name_is_credentials(self):
        """Test that missing kwarg_name defaults to 'credentials'."""

        class DefaultKwargSchema(BlockSchemaInput):
            """Schema with auto_credentials missing kwarg_name."""

            file: dict[str, Any] | None = SchemaField(
                description="File input",
                default=None,
                json_schema_extra={
                    "auto_credentials": {
                        "provider": "google",
                        "type": "oauth2",
                        "scopes": ["https://www.googleapis.com/auth/drive.file"],
                        # No kwarg_name specified - should default to "credentials"
                    }
                },
            )

        result = DefaultKwargSchema.get_auto_credentials_fields()

        assert "credentials" in result
        assert result["credentials"]["field_name"] == "file"

    def test_duplicate_default_kwarg_name_raises_error(self):
        """Test that two fields with default kwarg_name raises ValueError."""

        class DefaultDuplicateSchema(BlockSchemaInput):
            """Schema where both fields omit kwarg_name, defaulting to 'credentials'."""

            file1: dict[str, Any] | None = SchemaField(
                description="First file input",
                default=None,
                json_schema_extra={
                    "auto_credentials": {
                        "provider": "google",
                        "type": "oauth2",
                        "scopes": ["https://www.googleapis.com/auth/drive.file"],
                        # No kwarg_name - defaults to "credentials"
                    }
                },
            )
            file2: dict[str, Any] | None = SchemaField(
                description="Second file input",
                default=None,
                json_schema_extra={
                    "auto_credentials": {
                        "provider": "google",
                        "type": "oauth2",
                        "scopes": ["https://www.googleapis.com/auth/drive.file"],
                        # No kwarg_name - also defaults to "credentials"
                    }
                },
            )

        with pytest.raises(ValueError) as exc_info:
            DefaultDuplicateSchema.get_auto_credentials_fields()

        assert "Duplicate auto_credentials kwarg_name 'credentials'" in str(
            exc_info.value
        )


def test_agent_input_block_ignores_legacy_placeholder_values():
    """Verify AgentInputBlock.Input.model_construct tolerates extra placeholder_values
    for backward compatibility with existing agent JSON."""
    legacy_data = {
        "name": "url",
        "value": "",
        "description": "Enter a URL",
        "placeholder_values": ["https://example.com"],
    }
    instance = AgentInputBlock.Input.model_construct(**legacy_data)
    schema = instance.generate_schema()
    assert (
        "enum" not in schema
    ), "AgentInputBlock should not produce enum from legacy placeholder_values"


def test_dropdown_input_block_produces_enum():
    """Verify AgentDropdownInputBlock.Input.generate_schema() produces enum
    using the canonical 'options' field name."""
    opts = ["Option A", "Option B"]
    instance = AgentDropdownInputBlock.Input.model_construct(
        name="choice", value=None, options=opts
    )
    schema = instance.generate_schema()
    assert schema.get("enum") == opts


def test_dropdown_input_block_legacy_placeholder_values_produces_enum():
    """Verify backward compat: passing legacy 'placeholder_values' to
    AgentDropdownInputBlock still produces enum via model_construct remap."""
    opts = ["Option A", "Option B"]
    instance = AgentDropdownInputBlock.Input.model_construct(
        name="choice", value=None, placeholder_values=opts
    )
    schema = instance.generate_schema()
    assert (
        schema.get("enum") == opts
    ), "Legacy placeholder_values should be remapped to options"


def test_generate_schema_integration_legacy_placeholder_values():
    """Test the full Graph._generate_schema path with legacy placeholder_values
    on AgentInputBlock — verifies no enum leaks through the graph loading path."""
    legacy_input_default = {
        "name": "url",
        "value": "",
        "description": "Enter a URL",
        "placeholder_values": ["https://example.com"],
    }
    result = BaseGraph._generate_schema(
        (AgentInputBlock.Input, legacy_input_default),
    )
    url_props = result["properties"]["url"]
    assert (
        "enum" not in url_props
    ), "Graph schema should not contain enum from AgentInputBlock placeholder_values"


def test_generate_schema_integration_dropdown_produces_enum():
    """Test the full Graph._generate_schema path with AgentDropdownInputBlock
    — verifies enum IS produced for dropdown blocks using canonical field name."""
    dropdown_input_default = {
        "name": "color",
        "value": None,
        "options": ["Red", "Green", "Blue"],
    }
    result = BaseGraph._generate_schema(
        (AgentDropdownInputBlock.Input, dropdown_input_default),
    )
    color_props = result["properties"]["color"]
    assert color_props.get("enum") == [
        "Red",
        "Green",
        "Blue",
    ], "Graph schema should contain enum from AgentDropdownInputBlock"


def test_generate_schema_integration_dropdown_legacy_placeholder_values():
    """Test the full Graph._generate_schema path with AgentDropdownInputBlock
    using legacy 'placeholder_values' — verifies backward compat produces enum."""
    legacy_dropdown_input_default = {
        "name": "color",
        "value": None,
        "placeholder_values": ["Red", "Green", "Blue"],
    }
    result = BaseGraph._generate_schema(
        (AgentDropdownInputBlock.Input, legacy_dropdown_input_default),
    )
    color_props = result["properties"]["color"]
    assert color_props.get("enum") == [
        "Red",
        "Green",
        "Blue",
    ], "Legacy placeholder_values should still produce enum via model_construct remap"


def test_dropdown_input_block_init_legacy_placeholder_values():
    """Verify backward compat: constructing AgentDropdownInputBlock.Input via
    model_validate with legacy 'placeholder_values' correctly maps to 'options'."""
    opts = ["Option A", "Option B"]
    instance = AgentDropdownInputBlock.Input.model_validate(
        {"name": "choice", "value": None, "placeholder_values": opts}
    )
    assert (
        instance.options == opts
    ), "Legacy placeholder_values should be remapped to options via model_validate"
    schema = instance.generate_schema()
    assert schema.get("enum") == opts
