"""The per-user skill file store and version retention, against real Postgres."""

import asyncio
import json
from uuid import uuid4

import pytest

from backend.data import skill_version_store as store
from backend.data import skill_versions as versions
from backend.data.db import execute_raw_with_schema, query_raw_with_schema, transaction
from backend.data.skill_learning_test import _cleanup, _create_user
from backend.data.skill_package import file_sha256
from backend.data.skill_version_files import SkillVersionFile

SCRIPT = SkillVersionFile.from_content("scripts/check.py", b"print('same')")


async def _save(user: str, files, *, name: str = "store-check", **kwargs):
    head = await versions.ensure_head(user, None, name)
    return await versions.create_version(
        user,
        head=head,
        content=f"Run {uuid4()}",
        description="Store check",
        triggers=[],
        origin="saved_overnight",
        files=files,
        **kwargs,
    )


async def _held(user: str) -> set[str]:
    rows = await query_raw_with_schema(
        'SELECT "sha256" FROM {schema_prefix}"SkillFileContent" WHERE "userId" = $1',
        user,
    )
    return {row["sha256"] for row in rows}


def _file(path: str, text: str) -> SkillVersionFile:
    return SkillVersionFile.from_content(path, text.encode())


def _sha(file: SkillVersionFile) -> str:
    return file_sha256(file.content)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_file_unchanged_across_versions_skills_and_paths_is_stored_once(
    server,
):
    user = str(uuid4())
    await _create_user(user)
    try:
        first = await _save(user, [SCRIPT])
        second = await _save(user, [SCRIPT, _file("notes.md", "new")])
        moved = SkillVersionFile.from_content(
            "bin/check.py", SCRIPT.content, is_executable=True
        )
        other = await _save(user, [moved], name="other-skill")
        empty = await _save(user, [])
        plain = await _save(user, None)

        assert await _held(user) == {_sha(SCRIPT), _sha(_file("notes.md", "new"))}
        for version in (first, second, other, empty, plain):
            stored = await versions.get_version(user, version.id)
            assert stored is not None and stored.files == version.files
    finally:
        await _cleanup(user)


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("other_user_holds_it", [False, True])
async def test_another_users_copy_of_a_file_changes_nothing_for_this_user(
    server, other_user_holds_it
):
    other, user = str(uuid4()), str(uuid4())
    await _create_user(other)
    await _create_user(user)
    try:
        if other_user_holds_it:
            await _save(other, [SCRIPT])
        async with transaction() as tx:
            package = await store.store_version_files(tx, user, [SCRIPT])

        assert package.stored == 1
        assert await _held(user) == {_sha(SCRIPT)}
        assert await _held(other) == ({_sha(SCRIPT)} if other_user_holds_it else set())
    finally:
        await _cleanup(other, user)


@pytest.mark.asyncio(loop_scope="session")
async def test_history_keeps_the_newest_versions_and_what_open_work_points_at(
    server, mocker
):
    mocker.patch.object(store, "MAX_VERSIONS_PER_SKILL", 3)
    user, other = str(uuid4()), str(uuid4())
    await _create_user(user)
    await _create_user(other)
    a, b, c, d = (_file(f"{n}.md", n) for n in "abcd")
    try:
        others = await _save(other, [d])
        v1 = await _save(user, [a])
        await _save(user, [], state="needs_decision", base_version_id=v1.id)
        v3 = await _save(user, [b])
        head = await versions.ensure_head(user, None, "store-check")
        assert await versions.publish_version_if_current(
            user,
            head.id,
            expected_current_version=0,
            version_id=v3.id,
            version_number=v3.version,
            new_content_hash=v3.content_hash,
        )
        v4 = await _save(user, [c])
        await _save(user, [d])
        await _save(user, [a])
        await _save(user, [a], restored_from_version_id=v4.id)
        await _save(user, [a])

        kept = await versions.list_versions(user, "personal", "store-check")
        assert sorted(v.version for v in kept) == [1, 2, 3, 4, 6, 7, 8]
        assert await _held(user) == {_sha(a), _sha(b), _sha(c)}
        assert await _held(other) == {_sha(d)}
        stored = await versions.get_version(other, others.id)
        assert stored is not None and stored.files == [d]
    finally:
        await _cleanup(user, other)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_version_written_before_the_store_reads_the_same_once_moved(server):
    user = str(uuid4())
    await _create_user(user)
    try:
        packaged = await _save(user, None)
        markdown_only = await _save(user, None, name="plain-skill")
        for version, legacy in (
            (packaged, [SCRIPT.model_dump()]),
            (markdown_only, None),
        ):
            await execute_raw_with_schema(
                'UPDATE {schema_prefix}"SkillVersion" '
                'SET "packageFiles" = $2::jsonb WHERE "id" = $1',
                version.id,
                json.dumps(legacy),
            )
        before = await versions.get_version(user, packaged.id)
        assert before is not None and before.files == [SCRIPT]

        assert await store.move_inline_package_files() >= 2
        assert await store.move_inline_package_files() == 0

        rows = await query_raw_with_schema(
            'SELECT "id", "packageFiles", "packageManifest" '
            'FROM {schema_prefix}"SkillVersion" WHERE "userId" = $1',
            user,
        )
        shapes = {
            row["id"]: (row["packageFiles"], row["packageManifest"]) for row in rows
        }
        assert shapes[markdown_only.id] == (None, None)
        assert shapes[packaged.id][0] is None and shapes[packaged.id][1]
        after = await versions.get_version(user, packaged.id)
        assert after is not None and after.files == [SCRIPT]
        plain = await versions.get_version(user, markdown_only.id)
        assert plain is not None and plain.files is None
        assert await _held(user) == {_sha(SCRIPT)}
    finally:
        await _cleanup(user)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_cleanup_waits_for_a_version_being_written_with_a_file_it_frees(
    server, mocker
):
    """A save finds the user already holds the file; before it lists it, the
    retention of another skill drops the only version listing it."""
    mocker.patch.object(store, "MAX_VERSIONS_PER_SKILL", 1)
    user = str(uuid4())
    await _create_user(user)
    try:
        await _save(user, [SCRIPT], name="old-skill")
        checked, resume = asyncio.Event(), asyncio.Event()
        real_query = store.query_raw_with_schema

        async def held_check_then_wait(query, *args, **kwargs):
            rows = await real_query(query, *args, **kwargs)
            if '"SkillFileContent"' in query:
                checked.set()
                await asyncio.wait_for(resume.wait(), timeout=30)
            return rows

        mocker.patch.object(store, "query_raw_with_schema", held_check_then_wait)
        write = asyncio.create_task(_save(user, [SCRIPT], name="new-skill"))
        await asyncio.wait_for(checked.wait(), timeout=30)
        mocker.patch.object(store, "query_raw_with_schema", real_query)

        cleanup = asyncio.create_task(_save(user, [], name="old-skill"))
        await _until_blocked_or_done(cleanup)
        resume.set()
        written = await asyncio.wait_for(write, timeout=30)
        await asyncio.wait_for(cleanup, timeout=30)

        stored = await versions.get_version(user, written.id)
        assert stored is not None and stored.files == [SCRIPT]
        assert await _held(user) == {_sha(SCRIPT)}
    finally:
        await _cleanup(user)


async def _until_blocked_or_done(task: asyncio.Task) -> None:
    for _ in range(300):
        if task.done():
            return
        waiting = await query_raw_with_schema(
            "SELECT count(*) AS n FROM pg_locks "
            "WHERE locktype = 'advisory' AND NOT granted"
        )
        if waiting[0]["n"]:
            return
        await asyncio.sleep(0.1)
    raise AssertionError("the cleanup neither finished nor waited")
