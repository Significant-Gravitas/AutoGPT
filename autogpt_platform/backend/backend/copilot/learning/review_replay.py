"""Bounded recovery of validated, already-paid review output."""

import json
from datetime import datetime, timedelta, timezone
from typing import Literal

from pydantic import BaseModel, ValidationError

from backend.copilot.tools.skills import (
    ParsedSkill,
    canonicalize_skill,
    render_skill_markdown,
)
from backend.data.db_accessors import skill_reviews_db
from backend.data.skill_reviews import LearningReviewRecord

from .content_checks import check_metadata, check_skill_bundle
from .contract import SourceRevision
from .packages import ReviewedPackage
from .publication_models import PublishRequest
from .publication_validation import _validate_request

MAX_MODEL_ATTEMPTS = 3
MAX_REPLAY_BYTES = 524_288
REPLAY_TTL = timedelta(days=7)


class ReviewReplay(BaseModel):
    version: Literal[1] = 1
    created_at: datetime
    request: PublishRequest


async def load_review_replay(
    user_id: str, source: SourceRevision, policy_version: int
) -> PublishRequest | None:
    row = await skill_reviews_db().get_review_for_revision(
        user_id, source.source_id, source.revision, policy_version
    )
    if row is None or row.replay is None:
        return None
    try:
        if len(json.dumps(row.replay).encode("utf-8")) > MAX_REPLAY_BYTES:
            raise ValueError("oversized replay")
        replay = ReviewReplay.model_validate(row.replay)
        age = datetime.now(timezone.utc) - replay.created_at
        if not timedelta(0) <= age <= REPLAY_TTL:
            raise ValueError("expired replay")
        request = replay.request
        stamp = request.review
        if (
            stamp is None
            or stamp.policy_version != policy_version
            or request.user_id != user_id
            or request.expert_id != source.scope.expert_id
            or stamp.source.user_id != user_id
            or stamp.source.owner_key != source.scope.owner_key
            or stamp.source.id != source.source_id
            or stamp.source_revision != source.revision
            or stamp.source.epoch != source.epoch
            or len(request.sources) != 1
        ):
            raise ValueError("replay scope changed")
        snapshot = request.sources[0]
        if (
            snapshot.source_id != source.source_id
            or snapshot.source_ref != source.source_ref
            or snapshot.source_kind != source.source_kind
            or snapshot.revision != source.revision
            or snapshot.epoch != source.epoch
            or snapshot.approval != source.approval
            or request.expected_package_hash is None
        ):
            raise ValueError("replay source changed")
        return request
    except (ValidationError, ValueError, TypeError):
        await skill_reviews_db().set_review_replay(user_id, row.id, row.run_id, None)
        return None


async def save_review_replay(
    user_id: str,
    row: LearningReviewRecord,
    request: PublishRequest,
    packages: dict[str, ReviewedPackage],
) -> None:
    if not _safe_to_store(request, packages):
        return
    replay = ReviewReplay(created_at=datetime.now(timezone.utc), request=request)
    payload = replay.model_dump(mode="json")
    if len(json.dumps(payload).encode("utf-8")) > MAX_REPLAY_BYTES:
        return
    saved = await skill_reviews_db().set_review_replay(
        user_id, row.id, row.run_id, payload
    )
    if not saved:
        raise RuntimeError("review ownership changed before checkpoint")


def _safe_to_store(
    request: PublishRequest, packages: dict[str, ReviewedPackage]
) -> bool:
    if request.expected_package_hash is None or _validate_request(request) is not None:
        return False
    fields = {
        "name": request.skill_name,
        "description": request.description,
        "triggers": " ".join(request.triggers),
        "summary": request.summary,
        "limits": " ".join(request.limits),
        "refs": " ".join(request.supported_refs),
        "evidence": json.dumps(request.evidence),
        "files": " ".join(f.relative_path for f in request.files or []),
    }
    if check_metadata(fields) is not None:
        return False
    rendered = render_skill_markdown(
        canonicalize_skill(
            ParsedSkill(
                name=request.skill_name,
                description=request.description,
                body=request.body,
                triggers=tuple(request.triggers),
            )
        )
    )
    bundle = {
        "SKILL.md": rendered,
        **{
            f.relative_path: f.content.decode("utf-8", errors="replace")
            for f in request.files or []
        },
    }
    current = packages.get(request.skill_name)
    return (
        check_skill_bundle(
            bundle, approved_matches=current.content_approvals if current else []
        )
        is None
    )
