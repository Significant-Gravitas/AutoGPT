from unittest.mock import AsyncMock

import pytest

from backend.copilot.learning import publish
from backend.copilot.learning.contract import (
    EvidenceBundle,
    EvidenceSpan,
    LearningScope,
    OutcomeSignal,
    SourceRevision,
)
from backend.copilot.learning.packages import ReviewedPackage, _reviewed_package
from backend.copilot.learning.prompts import LearningProposal
from backend.copilot.learning.proposal import validate_proposal
from backend.copilot.tools.skills import (
    ParsedSkill,
    SkillFile,
    render_skills_index,
    validate_skill_content,
)

BODY = "## Steps\n1. Run scripts/normalize.py using references/cases.json.\n## Verification\nFixtures passed."
SCRIPT = "import json, sys\nprint(json.dumps([x.strip().lower() for x in json.load(sys.stdin)]))\n"


def bundle():
    return EvidenceBundle(
        source=SourceRevision(
            source_id="source",
            source_kind="chat",
            source_ref="session",
            scope=LearningScope(user_id="user-1", owner_key="personal"),
            revision="1",
            epoch=0,
            outcome_signals=[OutcomeSignal(kind="tool_result", ref="msg:3")],
        ),
        spans=[
            EvidenceSpan(
                ref="msg:3",
                role="tool",
                text="2 fixture cases passed",
                outcome="tool_result",
            )
        ],
    )


def script_proposal(**changes):
    return LearningProposal.model_validate(
        {
            "decision": "create",
            "skill_name": "normalize-input",
            "description": "Normalize input",
            "body": BODY,
            "supported_by": ["msg:3"],
            "verification": "2 fixture cases passed",
            "files": [
                {
                    "relative_path": "scripts/normalize.py",
                    "content": SCRIPT,
                    "supported_by": ["msg:3"],
                }
            ],
            **changes,
        }
    )


def request(files):
    return publish.PublishRequest(
        user_id="user-1",
        expert_id=None,
        skill_name="normalize-input",
        description="Normalize input",
        body=BODY,
        summary="Save reusable normalization",
        origin="saved_overnight",
        files=files,
    )


def test_natural_language_trigger_is_valid():
    validate_skill_content(
        "Replay paginated responses",
        "## Steps\n1. Replay responses.\n## Verification\nFixtures pass.",
        [
            "Replay cursor pagination with rate-limit retries and deduplication by item version"
        ],
    )


def test_reviewer_preserves_verified_script_proposal():
    proposal = LearningProposal.model_validate(
        {
            "decision": "create",
            "skill_name": "normalize-input",
            "description": "Normalize input",
            "body": "## Steps\n1. Run scripts/normalize.py.\n## Verification\nFixture passed.",
            "files": [
                {
                    "relative_path": "scripts/normalize.py",
                    "content": "print('verified')\n",
                    "supported_by": ["msg:3"],
                }
            ],
        }
    )
    assert proposal.model_dump()["files"][0]["content"] == "print('verified')\n"


@pytest.mark.asyncio
async def test_invalid_metadata_never_creates_pending_version(fake_store, monkeypatch):
    monkeypatch.setattr(
        publish, "read_user_skill_markdown", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(publish, "read_user_skill_files", AsyncMock(return_value=[]))
    write = AsyncMock(side_effect=ValueError("trigger too long"))
    monkeypatch.setattr(publish, "store_user_skill", write)
    outcome = await publish.publish_learned_version(
        publish.PublishRequest(
            user_id="user-1",
            expert_id=None,
            skill_name="normalize-input",
            description="Normalize input",
            triggers=["x" * 513],
            body="## Steps\n1. Normalize.\n## Verification\nFixture passed.",
            summary="Normalize input",
            origin="saved_overnight",
        )
    )
    assert outcome.status == "invalid_proposal"
    assert await fake_store.list_versions("user-1", "personal", "normalize-input") == []
    write.assert_not_awaited()


def test_trigger_index_is_bounded_without_truncating_stored_phrases():
    triggers = tuple(str(i) + "x" * 511 for i in range(10))
    skill = ParsedSkill(
        name="normalize-input",
        description="Normalize input",
        body=BODY,
        triggers=triggers,
    )
    index = render_skills_index([skill])
    assert len(index) < 750 and index.endswith("…")
    assert skill.triggers == triggers
    validate_skill_content(skill.description, skill.body, skill.triggers)


def test_file_evidence_and_existing_package_completeness():
    proposed = script_proposal()
    assert validate_proposal(proposed, bundle(), []) is None
    proposed.files[0].supported_by = ["missing"]
    assert "checked outcome" in validate_proposal(proposed, bundle(), [])
    proposed = script_proposal(decision="update")
    existing = [
        ParsedSkill(name="normalize-input", description="Normalize input", body=BODY)
    ]
    partial = {
        "normalize-input": ReviewedPackage(
            package_hash="previous", files=[], complete=False
        )
    }
    assert "complete existing skill package" in validate_proposal(
        proposed, bundle(), existing, partial
    )
    proposed.files = None
    assert validate_proposal(proposed, bundle(), existing, partial) is None


@pytest.mark.parametrize(
    "paths", [["../outside.py"], ["SKILL.md"], ["scripts/run.py", "scripts/run.py"]]
)
def test_package_paths_are_validated_before_publication(paths):
    proposal = script_proposal(
        files=[
            {"relative_path": p, "content": SCRIPT, "supported_by": ["msg:3"]}
            for p in paths
        ]
    )
    assert validate_proposal(proposal, bundle(), []) is not None


def test_binary_or_omitted_files_do_not_appear_as_an_empty_reviewed_package():
    files = [
        SkillFile(relative_path="assets/template.bin", content=b"\xff"),
        SkillFile(relative_path="references/long.md", content=b"x" * 30),
    ]
    reviewed = _reviewed_package("root", files, 10)
    assert reviewed.files == [] and not reviewed.complete
    assert _reviewed_package("root", [], 10).complete
