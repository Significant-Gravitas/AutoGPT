"""Learn, run, update and restore an executable package against real storage."""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from backend.copilot import db as chats
from backend.copilot.dream.llm import CompletionUsage, StructuredCompletion
from backend.copilot.learning import nightly
from backend.copilot.learning.chat_source import record_chat_turn
from backend.copilot.learning.owner_actions import restore_version
from backend.copilot.learning.packages import LearningFile
from backend.copilot.learning.prompts import LearningProposal
from backend.copilot.model import ChatSession
from backend.copilot.tools import skills
from backend.copilot.tools.skills import ReadSkillResponse, ReadSkillTool
from backend.data import db, skill_publication, skill_versions
from backend.data.skill_learning_flow_test import _persist_turn

SCRIPT = """import csv, sys
with open(sys.argv[1], newline='', encoding='utf-8') as source:
    rows = list(csv.reader(source))
print(f'{len(rows) - 1} rows validated')
"""
FIXTURE = "name,count\napples,1\npears,2\npeaches,3\n"


def proposal(revision: int):
    path = f"references/sample{revision}.csv"
    script = (
        SCRIPT
        if revision == 1
        else SCRIPT.replace(
            "print(", "assert len(set(rows[0])) == len(rows[0])\nprint("
        )
    )
    ref = f"msg:{2 if revision == 1 else 6}"
    return StructuredCompletion(
        value=LearningProposal(
            decision="create" if revision == 1 else "update",
            skill_name="csv-package",
            description="Validate CSV inputs using the saved validator and fixtures",
            triggers=[
                "Validate recurring CSV imports with encoding, column checks, and repeatable sample fixtures"
            ],
            body=f"## Steps\n1. Run python scripts/validate_csv.py {path}.\n## Verification\nThe fixture prints 3 rows validated.\n## Limits\nOnly the recorded fixture was checked.",
            files=[
                LearningFile(
                    relative_path="scripts/validate_csv.py",
                    content=script,
                    is_executable=True,
                    supported_by=[ref],
                ),
                LearningFile(relative_path=path, content=FIXTURE, supported_by=[ref]),
            ],
            supported_by=[ref],
            verification="The saved script exited 0 and printed 3 rows validated",
        ),
        usage=CompletionUsage(
            model="fixture-model", input_tokens=10, output_tokens=10, cost_usd=0
        ),
    )


def execute_script(directory: Path, fixture: str) -> str:
    result = subprocess.run(
        [sys.executable, "-I", "scripts/validate_csv.py", fixture],
        cwd=directory,
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    return result.stdout.strip()


@pytest.mark.asyncio(loop_scope="session")
async def test_nightly_package_can_run_update_and_restore(
    server, tmp_path, monkeypatch
):
    user, expert = str(uuid4()), str(uuid4())
    await db.prisma.user.create(data={"id": user, "email": f"{user}@example.invalid"})
    await db.prisma.expert.create(
        data={
            "id": expert,
            "ownerUserId": user,
            "name": "CSV",
            "role": "Data",
            "identity": "Validate CSV inputs",
        }
    )
    session = ChatSession.new(user, dry_run=False, expert_id=expert)
    await chats.create_chat_session(session.session_id, user, expert_id=expert)

    async def materialize(path, content, session_id):
        target = Path(path)
        assert target.is_relative_to(tmp_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return str(target)

    async def remove(paths, session_id):
        for path in paths:
            target = Path(path)
            assert target.is_relative_to(tmp_path)
            target.unlink(missing_ok=True)
        return []

    monkeypatch.setattr(skills, "workdir_root", lambda _: str(tmp_path / "sandbox"))
    monkeypatch.setattr(skills, "save_to_workdir", materialize)
    monkeypatch.setattr(skills, "read_workdir_bytes", AsyncMock(return_value=None))
    monkeypatch.setattr(skills, "remove_from_workdir", remove)
    monkeypatch.setattr(skills, "set_executable", AsyncMock(return_value=[]))
    monkeypatch.setattr(nightly, "is_feature_enabled", AsyncMock(return_value=True))
    monkeypatch.setattr(
        nightly, "check_dream_budget", AsyncMock(return_value=(True, None))
    )
    monkeypatch.setattr(nightly, "record_review_cost", AsyncMock(return_value=0))
    monkeypatch.setattr(
        "backend.copilot.learning.dispositions.link_version_to_memory",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr("backend.util.workspace.scan_content_safe", AsyncMock())
    monkeypatch.setattr(
        skills, "is_skills_feature_enabled", AsyncMock(return_value=True)
    )
    reviewer = AsyncMock(side_effect=[proposal(1), proposal(2)])
    monkeypatch.setattr(nightly, "review_evidence", reviewer)
    try:
        originals = []
        duplicate_columns = tmp_path / "duplicate-columns.csv"
        duplicate_columns.write_text(FIXTURE.replace("name,count", "name,name"))
        for revision in (1, 2):
            candidate = proposal(revision).value
            fixture = f"references/sample{revision}.csv"
            training = tmp_path / f"training{revision}"
            for file in candidate.files:
                target = training / file.relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(file.content)
            assert execute_script(training, fixture) == "3 rows validated"
            arguments = json.dumps(
                {
                    "command": f"python scripts/validate_csv.py {fixture}",
                    "verified_files": {
                        f.relative_path: f.content for f in candidate.files
                    },
                }
            )
            messages = await _persist_turn(
                session,
                (revision - 1) * 4,
                "Save the verified CSV procedure",
                arguments,
            )
            await record_chat_turn(
                user, session, messages, "Save the verified CSV procedure"
            )
            result = await nightly.run_skill_learning_pass(user)
            assert result.applied == 1, result.model_dump()
            loaded = await ReadSkillTool()._execute(user, session, name="csv-package")
            assert isinstance(loaded, ReadSkillResponse) and loaded.version == revision
            assert len(loaded.files) == 2 and loaded.package_dir
            assert (
                execute_script(Path(loaded.package_dir), fixture) == "3 rows validated"
            )
            if revision == 2:
                with pytest.raises(subprocess.CalledProcessError):
                    execute_script(Path(loaded.package_dir), str(duplicate_columns))
            record = await skill_versions.get_version(user, loaded.version_id)
            assert record.files and len(record.files) == 2
            originals.append(record)
        reviewed = reviewer.call_args.args[3]["csv-package"]
        assert reviewed.complete and len(reviewed.files) == 2
        assert await skill_versions.get_version(str(uuid4()), originals[0].id) is None
        restored = await restore_version(
            user_id=user,
            expert_id=expert,
            skill_name="csv-package",
            version_id=originals[0].id,
            actor_user_id=user,
        )
        assert restored.status == "applied", restored.reason
        loaded = await ReadSkillTool()._execute(user, session, name="csv-package")
        assert isinstance(loaded, ReadSkillResponse) and loaded.version == 3
        assert (
            execute_script(Path(loaded.package_dir), "references/sample1.csv")
            == "3 rows validated"
        )
        assert (
            execute_script(Path(loaded.package_dir), str(duplicate_columns))
            == "3 rows validated"
        )
        restored_package = await skills.read_user_skill_package(
            user, "csv-package", expert_id=expert
        )
        assert {f.relative_path for f in restored_package.files} == {
            "scripts/validate_csv.py",
            "references/sample1.csv",
        }
        assert (
            await skill_versions.get_version(user, originals[0].id)
        ).files == originals[0].files
        assert reviewer.await_count == 2
        assert await skill_publication.list_pending_publications(user) == []
    finally:
        await db.prisma.user.delete_many(where={"id": user})
