from unittest.mock import AsyncMock

import pytest

from backend.copilot.tools import skills
from backend.copilot.tools.skill_updates_test import _install, _marketplace, _package
from backend.copilot.tools.skills_test import _FakeWorkspaceManager, _patch_skills_path


@pytest.mark.asyncio
@pytest.mark.parametrize("change_asset", [False, True])
async def test_a_failed_marketplace_update_preserves_baseline_and_retries(
    monkeypatch, fake_learning_store, change_asset
):
    manager = _FakeWorkspaceManager()
    v1 = _package(
        "checks",
        "v1",
        "Run scripts/check.py\n",
        {"scripts/check.py": b"print('old')", "scripts/removed.py": b"print('gone')"},
    )
    v2 = _package(
        "checks",
        "v2",
        "Run scripts/check.py with the updated checks\n",
        {"scripts/check.py": b"print('new')" if change_asset else b"print('old')"},
    )
    with _patch_skills_path(manager) as patched:
        await _install(v1)
        _marketplace(patched, v1, v2)
        invalidate = AsyncMock()
        remember = AsyncMock()
        monkeypatch.setattr(skills, "invalidate_skills_index_cache", invalidate)
        monkeypatch.setattr(skills, "_remember_updated", remember)
        write = manager.write_file

        async def fail_the_root(**kwargs):
            # The last step of the package swap, after the stale file is gone.
            if kwargs["path"].endswith("/SKILL.md"):
                raise OSError("unavailable")
            return await write(**kwargs)

        monkeypatch.setattr(manager, "write_file", fail_the_root)

        [failed] = await skills.list_user_skills("user-1")
        assert failed.update == skills.UPDATE_AVAILABLE
        assert failed.baseline is not None
        assert failed.baseline.version_id == v1.version_id
        assert manager.files["/skills/checks/SKILL.md"] == v1.skill_markdown.encode()
        assert (
            manager.metadata["/skills/checks/SKILL.md"]["listing_version_id"]
            == v1.version_id
        )
        assert "/skills/checks/scripts/removed.py" in manager.files
        assert (
            len(await fake_learning_store.list_versions("user-1", "personal", "checks"))
            == 1
        )
        invalidate.assert_not_awaited()
        remember.assert_not_awaited()

        monkeypatch.setattr(manager, "write_file", write)
        [recovered] = await skills.list_user_skills("user-1")
        assert recovered.baseline is not None
        assert recovered.baseline.version_id == v2.version_id
        assert manager.files["/skills/checks/SKILL.md"] == v2.skill_markdown.encode()
        assert manager.files["/skills/checks/scripts/check.py"] == v2.files[0].content
        assert "/skills/checks/scripts/removed.py" not in manager.files
        assert (
            len(await fake_learning_store.list_versions("user-1", "personal", "checks"))
            == 2
        )
        invalidate.assert_awaited_once_with("user-1", None)
        assert remember.await_count > 0
        assert patched.skill_db.get_version_packages.await_count == 2
