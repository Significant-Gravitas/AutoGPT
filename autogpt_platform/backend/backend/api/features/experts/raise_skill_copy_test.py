"""Raise reconciliation with real skill IO and isolated database/service boundaries."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import prisma.enums
import prisma.models
import pytest

from backend.api.features.experts import experts_db, raise_attachments
from backend.api.features.experts.models import RaiseAttachment
from backend.api.features.workspace import embeddings
from backend.copilot.learning import history
from backend.copilot.learning._fake_store import FakeLearningStore
from backend.copilot.tools import skills
from backend.data.skill_learning import owner_key_for
from backend.data.workspace import WorkspaceFile
from backend.data.workspace_skill import RetiredSkillFile, WorkspaceSkillPublication
from backend.util import workspace
from backend.util.exceptions import NotFoundError
from backend.util.workspace_storage import LocalWorkspaceStorage


class _WorkspaceRows:
    def __init__(self):
        self.rows: dict[tuple[str, str], WorkspaceFile] = {}
        self.learning = FakeLearningStore()

    async def get_or_create_workspace(self, user_id):
        return SimpleNamespace(id=f"ws-{user_id}")

    async def get_workspace_file_by_path(self, workspace_id, path):
        return next(
            (
                r
                for r in self.rows.values()
                if r.workspace_id == workspace_id and r.path == path
            ),
            None,
        )

    async def get_workspace_file(self, file_id, workspace_id):
        return self.rows.get((workspace_id, file_id))

    async def get_workspace_total_size(self, workspace_id):
        return sum(
            r.size_bytes for r in self.rows.values() if r.workspace_id == workspace_id
        )

    async def list_workspace_files(self, workspace_id, path_prefix=None, **kwargs):
        rows = [
            r
            for r in reversed(self.rows.values())
            if r.workspace_id == workspace_id
            and (path_prefix is None or r.path.startswith(path_prefix))
            and (kwargs.get("name_contains") or "").lower() in r.name.lower()
        ]
        offset, limit = kwargs.get("offset", 0), kwargs.get("limit")
        return rows[offset:] if limit is None else rows[offset : offset + limit]

    async def create_workspace_file(self, file_id, **data):
        now = datetime.now(timezone.utc)
        row = WorkspaceFile(
            id=file_id,
            created_at=now,
            updated_at=now,
            **{**data, "metadata": data.get("metadata") or {}},
        )
        self.rows[(row.workspace_id, row.id)] = row
        return row

    async def publish_workspace_skill_file(self, write):
        previous = dict(self.rows)
        retired = []
        try:
            for path in [
                write.path,
                *(f.path for f in write.files),
                *write.removed_paths,
            ]:
                existing = await self.get_workspace_file_by_path(
                    write.workspace_id, path
                )
                if existing is not None:
                    await self.soft_delete_workspace_file(
                        existing.id, write.workspace_id
                    )
                    retired.append(
                        RetiredSkillFile(
                            file_id=existing.id, storage_path=existing.storage_path
                        )
                    )
            for file in write.files:
                await self.create_workspace_file(
                    workspace_id=write.workspace_id, **file.model_dump()
                )
            row = await self.create_workspace_file(
                **write.model_dump(exclude={"overwrite", "files", "removed_paths"})
            )
        except BaseException:
            self.rows = previous
            raise
        return WorkspaceSkillPublication(
            status="stored", file=row, retired_files=retired
        )

    async def soft_delete_workspace_file(self, file_id, workspace_id):
        return self.rows.pop((workspace_id, file_id), None)


@pytest.fixture
def skill_storage(monkeypatch, tmp_path):
    rows = _WorkspaceRows()
    storage = LocalWorkspaceStorage(str(tmp_path / "blobs"))
    monkeypatch.setattr(workspace, "workspace_db", lambda: rows)
    monkeypatch.setattr(skills, "workspace_db", lambda: rows)
    monkeypatch.setattr(workspace, "workspace_skill_db", lambda: rows)
    monkeypatch.setattr(
        workspace, "get_workspace_storage", AsyncMock(return_value=storage)
    )
    monkeypatch.setattr(
        workspace, "get_workspace_storage_limit_bytes", AsyncMock(return_value=0)
    )
    monkeypatch.setattr(workspace, "scan_content_safe", AsyncMock())
    monkeypatch.setattr(embeddings, "schedule_workspace_file_embedding", MagicMock())
    monkeypatch.setattr(embeddings, "delete_workspace_file_embedding", AsyncMock())
    redis = AsyncMock()
    redis.get.return_value = None
    monkeypatch.setattr(skills, "get_redis_async", AsyncMock(return_value=redis))
    lock = MagicMock(owner_id="test-owner")
    lock.try_acquire = AsyncMock(return_value="test-owner")
    lock.refresh = AsyncMock(return_value=True)
    lock.release = AsyncMock()
    monkeypatch.setattr(skills, "AsyncClusterLock", lambda *args, **kwargs: lock)
    monkeypatch.setattr(skills, "experts_db", lambda: experts_db)
    monkeypatch.setattr(skills, "skill_versions_db", lambda: rows.learning)
    monkeypatch.setattr(skills, "skill_deletion_db", lambda: rows.learning)
    monkeypatch.setattr(history, "skill_versions_db", lambda: rows.learning)
    monkeypatch.setattr(history, "skill_publication_db", lambda: rows.learning)
    return rows


@pytest.fixture
def expert_rows(monkeypatch):
    saved = {}

    async def create(data, **kwargs):
        now = datetime.now(timezone.utc)
        row = prisma.models.Expert(
            id="raised",
            createdAt=now,
            updatedAt=now,
            categories=[],
            isTemplate=False,
            isArchived=False,
            setupStatus=prisma.enums.ExpertSetupStatus.READY,
            setupFailures=[],
            visibility=prisma.enums.ResourceVisibility.PRIVATE,
            **data,
        )
        saved[row.id] = row
        return row.model_copy(deep=True)

    async def update(where, data, **kwargs):
        row = saved[where["id"]].model_copy(update=data)
        saved[row.id] = row
        return row.model_copy(deep=True)

    async def find_first(where, **kwargs):
        row = saved.get(where["id"])
        assert row is not None and row.ownerUserId == where["ownerUserId"]
        return row.model_copy(deep=True)

    client = SimpleNamespace(
        count=AsyncMock(return_value=0),
        create=create,
        update=update,
        find_first=find_first,
    )

    @asynccontextmanager
    async def transaction():
        yield SimpleNamespace(expert=client, execute_raw=AsyncMock())

    monkeypatch.setattr(experts_db, "transaction", transaction)
    monkeypatch.setattr(prisma.models.Expert, "prisma", lambda: client)
    return saved


@pytest.mark.parametrize("delete_source", [False, True])
@pytest.mark.parametrize("library_first", [False, True])
async def test_raise_reconciles_a_library_source_deleted_after_resolution(
    monkeypatch, skill_storage, expert_rows, delete_source, library_first
):
    await skills.store_user_skill(
        "owner",
        name="shared",
        description="Private instructions",
        body="PRIVATE BODY",
        files=[
            skills.SkillFile(
                relative_path="scripts/run.py",
                content=b"print('private')\n",
                is_executable=True,
            )
        ],
    )
    monkeypatch.setattr(
        raise_attachments.skill_db, "get_marketplace_skill", AsyncMock()
    )

    async def unavailable(user_id, slug, *, expert_id):
        assert (user_id, slug, expert_id) == ("owner", "shared", "raised")
        if delete_source:
            await skills.delete_user_skill(user_id, slug)
        raise NotFoundError("Hub listing withdrawn after resolution")

    monkeypatch.setattr(
        raise_attachments.skill_db, "install_marketplace_skill", unavailable
    )
    sources = (
        ["library", "marketplace"] if library_first else ["marketplace", "library"]
    )
    attachments = [
        RaiseAttachment(kind="skill", source=source, id=" SHARED ")
        for source in sources
    ]
    result = await experts_db.create_raised_expert(
        "owner", "Nova", None, None, attachments=attachments * 2
    )
    package = await skills.read_user_skill_package(
        "owner", "shared", expert_id="raised"
    )
    indexed = await skills._list_user_skills_from_workspace("owner", "raised")
    expected_names = [] if delete_source else ["shared"]
    assert result.expert.skills == expert_rows["raised"].skills == expected_names
    assert [entry.name for entry in indexed] == expected_names
    assert [(f.source, f.reason) for f in result.failed_attachments] == [
        ("marketplace", "unavailable")
    ]
    if delete_source:
        assert package is None
        assert skill_storage.rows == {}
        head = await skill_storage.learning.get_head(
            "owner", owner_key_for(None), "shared"
        )
        assert head is not None and head.deleted
    else:
        assert package is not None and "PRIVATE BODY" in package.skill_md
        assert package.files == [
            skills.SkillFile(
                relative_path="scripts/run.py",
                content=b"print('private')\n",
                is_executable=True,
            )
        ]
        assert indexed[0].origin == skills.SKILL_ORIGIN_USER
        head = await skill_storage.learning.get_head(
            "owner", owner_key_for("raised"), "shared"
        )
        assert (
            head is not None
            and not head.auto_improve
            and head.current_version_id is not None
        )
        version = await skill_storage.learning.get_version(
            "owner", head.current_version_id
        )
        assert version is not None and version.origin == "imported"
        assert version.actor_user_id == "owner" and version.files is not None
        assert [
            (f.relative_path, f.content, f.is_executable) for f in version.files
        ] == [("scripts/run.py", b"print('private')\n", True)]
        assert (
            await skills.read_user_skill_package(
                "another-owner", "shared", expert_id="raised"
            )
            is None
        )


async def test_raise_keeps_a_copy_when_library_attachments_share_a_name(
    monkeypatch, skill_storage, expert_rows
):
    await skills.store_user_skill(
        "owner", name="shared", description="Private skill", body="PRIVATE BODY"
    )
    manager = await skills._get_user_skill_manager("owner")
    await manager.write_file(
        b"---\nname: shared\ndescription: Legacy skill\n---\nALTERNATE BODY\n",
        "SKILL.md",
        path="/skills/alternate/SKILL.md",
    )
    copy = experts_db.copy_skill_to_expert
    attempts = []

    async def copy_with_a_repeated_lookup_failure(user_id, expert_id, slug):
        if attempts:
            raise OSError("Repeated storage lookup failed")
        attempts.append(slug)
        return await copy(user_id, expert_id, slug)

    monkeypatch.setattr(
        experts_db, "copy_skill_to_expert", copy_with_a_repeated_lookup_failure
    )
    result = await experts_db.create_raised_expert(
        "owner",
        "Nova",
        None,
        None,
        attachments=[
            RaiseAttachment(kind="skill", source="library", id="shared"),
            RaiseAttachment(kind="skill", source="library", id="alternate"),
        ],
    )
    package = await skills.read_user_skill_package(
        "owner", "shared", expert_id="raised"
    )

    assert package is not None and "PRIVATE BODY" in package.skill_md
    assert result.expert.skills == expert_rows["raised"].skills == ["shared"]


@pytest.mark.parametrize(
    ("builtin_slug", "delete_custom"),
    [
        ("shared", False),
        ("builtin-with-a-different-slug", False),
        ("builtin-with-a-different-slug", True),
    ],
)
async def test_raise_keeps_a_confirmed_builtin_without_a_workspace_copy(
    monkeypatch, skill_storage, expert_rows, builtin_slug, delete_custom
):
    if delete_custom:
        await skills.store_user_skill(
            "owner", name="shared", description="Custom skill", body="Private steps"
        )
    original_builtin = skills.DEFAULT_SKILLS[0]
    builtin = skills._DefaultSkill(
        name="shared",
        description=original_builtin.description,
        body_path=original_builtin.body_path,
        triggers=original_builtin.triggers,
    )
    monkeypatch.setattr(skills, "DEFAULT_SKILLS", (builtin,))
    monkeypatch.setattr(skills, "_DEFAULT_SKILLS_BY_NAME", {builtin_slug: builtin})
    monkeypatch.setattr(
        raise_attachments.skill_db, "get_marketplace_skill", AsyncMock()
    )

    async def unavailable(user_id, slug, *, expert_id):
        if delete_custom:
            await skills.delete_user_skill(user_id, slug)
        raise NotFoundError("withdrawn")

    monkeypatch.setattr(
        raise_attachments.skill_db, "install_marketplace_skill", unavailable
    )
    attachments = [
        RaiseAttachment(kind="skill", source="marketplace", id="shared"),
        RaiseAttachment(kind="skill", source="library", id=builtin_slug),
    ]
    if delete_custom:
        attachments.append(RaiseAttachment(kind="skill", source="library", id="shared"))
    result = await experts_db.create_raised_expert(
        "owner",
        "Nova",
        None,
        None,
        attachments=attachments,
    )

    assert result.expert.skills == expert_rows["raised"].skills == ["shared"]
    assert skill_storage.rows == {}
    assert (
        await skills.read_user_skill_package("owner", "shared", expert_id="raised")
        is None
    )
    indexed = (
        skills.get_default_skills_for_index()
        + await skills._list_user_skills_from_workspace("owner", "raised")
    )
    assert [entry.name for entry in indexed] == ["shared"]
    default = skills.get_default_skill_with_body(builtin_slug)
    assert default is not None and default.body
