from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from backend.copilot.learning.owner_actions import apply_owner_edit, restore_version
from backend.copilot.learning.publish import PublishRequest, publish_learned_version
from backend.copilot.tools.skills import read_user_skill_markdown, store_user_skill
from backend.data import skill_versions as versions
from backend.data.skill_learning_test import _cleanup, _create_user


@pytest.mark.asyncio(loop_scope="session")
async def test_owner_approved_occurrence_survives_restore_and_safe_improvement(server):
    user = str(uuid4())
    name = "approved-example"
    token = "Bearer " + "x" * 30
    body = f"## Steps\n1. Use the public example {token}\n## Verification\nChecked example\n"
    await _create_user(user)
    try:
        with patch("backend.util.workspace.scan_content_safe", AsyncMock()):
            await store_user_skill(
                user, name=name, description="Example", body="Original"
            )
            head = await versions.get_head(user, "personal", name)
            edited = await apply_owner_edit(
                user_id=user,
                expert_id=None,
                skill_name=name,
                description="Example",
                body=body,
                triggers=[],
                keep_auto_improve=True,
                allowed_pattern_classes=["bearer_token"],
                expected_version_id=head.current_version_id,
            )
            assert edited.status == "applied" and edited.version is not None
            approved = edited.version
            await store_user_skill(
                user, name=name, description="Example", body="Different"
            )
            restored = await restore_version(
                user_id=user,
                expert_id=None,
                skill_name=name,
                version_id=approved.id,
                actor_user_id=user,
            )
            assert restored.status == "applied", restored.reason
            assert token in await read_user_skill_markdown(user, name)
            await versions.update_head_policy(user, "personal", name, auto_improve=True)
            request = PublishRequest(
                user_id=user,
                expert_id=None,
                skill_name=name,
                description="Example",
                body=body + "\n2. Recheck the public example.\n",
                summary="Add recheck",
                origin="saved_overnight",
            )
            improved = await publish_learned_version(request)
            assert improved.status == "applied", improved.reason
            current = (
                await versions.get_head(user, "personal", name)
            ).current_version_id
            for changed in (body.replace("x" * 30, "y" * 30), body + "\n" + token):
                rejected = await publish_learned_version(
                    request.model_copy(update={"body": changed})
                )
                assert rejected.status == "blocked_content"
                assert (
                    await versions.get_head(user, "personal", name)
                ).current_version_id == current
    finally:
        await _cleanup(user)
