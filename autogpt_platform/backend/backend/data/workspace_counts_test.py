from uuid import uuid4

import pytest
from prisma.models import User

from backend.data.workspace import (
    count_generated_files_by_session,
    create_workspace_file,
    get_or_create_workspace,
    soft_delete_workspace_file,
)


@pytest.mark.asyncio(loop_scope="session")
async def test_session_file_counts_match_visible_generated_files(setup_test_user):
    workspace = await get_or_create_workspace(setup_test_user)
    other_user = await User.prisma().create(
        data={"id": str(uuid4()), "email": f"file-counts-{uuid4()}@example.com"}
    )
    other = await get_or_create_workspace(other_user.id)
    session_id = str(uuid4())
    for path, metadata, deleted, workspace_id in [
        ("report.txt", None, False, workspace.id),
        ("nested/tool-results/report.txt", None, False, workspace.id),
        ("tool-results/kept.txt", {"purpose": "artifact"}, False, workspace.id),
        ("upload.txt", {"origin": "user-upload"}, False, workspace.id),
        ("AGENT.JSON", None, False, workspace.id),
        ("tool-results/raw.txt", None, False, workspace.id),
        ("tool-outputs/raw.txt", None, False, workspace.id),
        ("raw.txt", {"purpose": "tool-output"}, False, workspace.id),
        ("deleted.txt", None, True, workspace.id),
        ("other-user.txt", None, False, other.id),
    ]:
        stored = await create_workspace_file(
            workspace_id,
            str(uuid4()),
            path.rsplit("/", 1)[-1],
            f"/sessions/{session_id}/{path}",
            f"{workspace_id}/{path}",
            "text/plain",
            3,
            metadata=metadata,
        )
        if deleted:
            await soft_delete_workspace_file(stored.id, workspace_id)

    assert await count_generated_files_by_session(setup_test_user, []) == {}
    assert await count_generated_files_by_session(setup_test_user, [session_id]) == {
        session_id: 3
    }
    assert await count_generated_files_by_session(other_user.id, [session_id]) == {
        session_id: 1
    }
