from __future__ import annotations

import re
from collections import Counter
from enum import StrEnum
from pathlib import PurePosixPath
from urllib.parse import unquote, urlsplit

from prisma import Prisma
from pydantic import BaseModel, ConfigDict

if __package__:
    from scripts.media_url_backfill_queries import UPDATE_QUERIES
else:
    from media_url_backfill_queries import UPDATE_QUERIES

PRIVATE_MEDIA_PREFIX = "/api/store/submissions/media"
_SAFE_PATH_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")
_CONTENT_TYPE_BY_SUFFIX = {
    ".gif": "image",
    ".jpeg": "image",
    ".jpg": "image",
    ".png": "image",
    ".webp": "image",
    ".mp4": "video",
    ".webm": "video",
}


class MalformedPrivateMediaUrl(ValueError):
    pass


class Target(StrEnum):
    PROFILE_AVATAR = "Profile.avatarUrl"
    EXPERT_AVATAR = "Expert.avatarUrl"
    LIBRARY_IMAGE = "LibraryAgent.imageUrl"
    LISTING_IMAGES = "StoreListingVersion.imageUrls"
    LISTING_VIDEO = "StoreListingVersion.videoUrl"
    LISTING_DEMO = "StoreListingVersion.agentOutputDemoUrl"
    ORGANIZATION_AVATAR = "Organization.avatarUrl"
    ORGANIZATION_PROFILE_AVATAR = "OrganizationProfile.avatarUrl"
    OAUTH_LOGO = "OAuthApplication.logoUrl"
    SKILL_SOURCE = "SkillListingVersion.sourceUrl"


class HoldReason(StrEnum):
    ACTIVE_PUBLIC = "active_public"
    MARKETPLACE = "marketplace"
    AMBIGUOUS = "ambiguous"


class Outcome(StrEnum):
    REWRITE = "rewrite"
    HOLD_ACTIVE_PUBLIC = "hold_active_public"
    HOLD_MARKETPLACE = "hold_marketplace"
    HOLD_AMBIGUOUS = "hold_ambiguous"
    HOLD_CROSS_USER = "hold_cross_user"
    HOLD_MALFORMED = "hold_malformed"


class ParsedPrivateMedia(BaseModel):
    model_config = ConfigDict(frozen=True)

    owner_user_id: str
    media_type: str
    filename: str
    private_url: str


class Candidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    target: Target
    record_id: str
    owner_user_id: str | None
    values: list[str]
    is_array: bool
    hold_reason: HoldReason | None = None


class Mutation(BaseModel):
    model_config = ConfigDict(frozen=True)

    target: Target
    record_id: str
    owner_user_id: str
    old_values: list[str]
    new_values: list[str]
    is_array: bool


class BackfillPlan(BaseModel):
    mutations: list[Mutation]
    counts: Counter[Outcome]
    target_counts: dict[Target, Counter[Outcome]]


class BackfillReport(BaseModel):
    counts: Counter[Outcome]
    target_counts: dict[Target, Counter[Outcome]]
    planned_rows: int
    applied_rows: int = 0
    cas_conflicts: int = 0

    @classmethod
    def from_plan(cls, plan: BackfillPlan) -> BackfillReport:
        return cls(
            counts=plan.counts,
            target_counts=plan.target_counts,
            planned_rows=len(plan.mutations),
        )


def parse_private_media_url(url: str, private_bucket: str) -> ParsedPrivateMedia | None:
    parsed = urlsplit(url)
    object_path = _managed_object_path(parsed, private_bucket)
    if object_path is None:
        return None
    if parsed.query or parsed.fragment:
        raise MalformedPrivateMediaUrl("Managed URL has query or fragment")

    parts = object_path.split("/")
    if len(parts) != 4 or parts[0] != "users":
        raise MalformedPrivateMediaUrl("Managed object path is malformed")
    _, owner_user_id, media_type, filename = parts
    if not _safe_path_component(owner_user_id) or not _safe_path_component(filename):
        raise MalformedPrivateMediaUrl("Managed object path is malformed")
    if media_type not in {"images", "videos"}:
        raise MalformedPrivateMediaUrl("Managed object has an invalid media type")

    content_kind = _CONTENT_TYPE_BY_SUFFIX.get(PurePosixPath(filename).suffix.lower())
    if content_kind != media_type.removesuffix("s"):
        raise MalformedPrivateMediaUrl(
            "Managed object filename does not match its type"
        )
    private_url = f"{PRIVATE_MEDIA_PREFIX}/{owner_user_id}/{media_type}/{filename}"
    return ParsedPrivateMedia(
        owner_user_id=owner_user_id,
        media_type=media_type,
        filename=filename,
        private_url=private_url,
    )


def build_plan(candidates: list[Candidate], private_bucket: str) -> BackfillPlan:
    counts: Counter[Outcome] = Counter()
    target_counts: dict[Target, Counter[Outcome]] = {}
    mutations: list[Mutation] = []
    for candidate in candidates:
        if not candidate.is_array and len(candidate.values) != 1:
            raise ValueError("Scalar candidate must have exactly one value")
        new_values = list(candidate.values)
        changed = False
        for index, value in enumerate(candidate.values):
            outcome, replacement = _classify_reference(candidate, value, private_bucket)
            if outcome is None:
                continue
            counts[outcome] += 1
            target_counts.setdefault(candidate.target, Counter())[outcome] += 1
            if replacement is not None:
                new_values[index] = replacement
                changed = True
        if changed:
            if candidate.owner_user_id is None:
                raise RuntimeError("A rewrite candidate has no proven owner")
            mutations.append(
                Mutation(
                    target=candidate.target,
                    record_id=candidate.record_id,
                    owner_user_id=candidate.owner_user_id,
                    old_values=candidate.values,
                    new_values=new_values,
                    is_array=candidate.is_array,
                )
            )
    return BackfillPlan(
        mutations=mutations,
        counts=counts,
        target_counts=target_counts,
    )


async def process_candidates(
    client: Prisma,
    candidates: list[Candidate],
    private_bucket: str,
    *,
    apply: bool,
) -> BackfillReport:
    plan = build_plan(candidates, private_bucket)
    report = BackfillReport.from_plan(plan)
    if not apply:
        return report

    for mutation in plan.mutations:
        changed = await client.execute_raw(
            UPDATE_QUERIES[mutation.target.value],
            mutation.record_id,
            mutation.new_values if mutation.is_array else mutation.new_values[0],
            mutation.old_values if mutation.is_array else mutation.old_values[0],
            mutation.owner_user_id,
        )
        if changed == 1:
            report.applied_rows += 1
        elif changed == 0:
            report.cas_conflicts += 1
        else:
            raise RuntimeError("A compare-and-swap update affected multiple rows")
    return report


def print_report(report: BackfillReport, *, apply: bool) -> None:
    print("Private media URL backfill summary (aggregate counts only):")
    for outcome in Outcome:
        print(f"  {outcome.value}: {report.counts[outcome]}")
    for target in Target:
        target_counts = report.target_counts.get(target)
        if not target_counts:
            continue
        summary = ", ".join(
            f"{outcome.value}={target_counts[outcome]}"
            for outcome in Outcome
            if target_counts[outcome]
        )
        print(f"  {target.value}: {summary}")
    print(f"  planned_row_updates: {report.planned_rows}")
    if apply:
        print(f"  applied_row_updates: {report.applied_rows}")
        print(f"  compare_and_swap_conflicts: {report.cas_conflicts}")
    else:
        print("Dry run: no database writes were attempted. Re-run with --apply.")


def _managed_object_path(parsed, private_bucket: str) -> str | None:
    scheme = parsed.scheme.lower()
    host = parsed.netloc.lower()
    bucket = private_bucket.lower()
    if scheme == "gs" and host == bucket:
        return unquote(parsed.path.lstrip("/"))
    if scheme not in {"http", "https"}:
        return None
    if host == "storage.googleapis.com":
        path_bucket, separator, object_path = unquote(
            parsed.path.lstrip("/")
        ).partition("/")
        return object_path if separator and path_bucket == private_bucket else None
    if host == f"{bucket}.storage.googleapis.com":
        return unquote(parsed.path.lstrip("/"))
    return None


def _safe_path_component(value: str) -> bool:
    return (
        value not in {".", ".."} and _SAFE_PATH_COMPONENT.fullmatch(value) is not None
    )


def _classify_reference(
    candidate: Candidate, value: str, private_bucket: str
) -> tuple[Outcome | None, str | None]:
    try:
        parsed = parse_private_media_url(value, private_bucket)
    except MalformedPrivateMediaUrl:
        return Outcome.HOLD_MALFORMED, None
    if parsed is None:
        return None, None
    if (
        candidate.owner_user_id is not None
        and parsed.owner_user_id != candidate.owner_user_id
    ):
        return Outcome.HOLD_CROSS_USER, None
    if not _target_accepts_media_type(candidate.target, parsed.media_type):
        return Outcome.HOLD_MALFORMED, None
    if candidate.hold_reason is not None:
        return _HOLD_OUTCOMES[candidate.hold_reason], None
    if candidate.owner_user_id is None:
        return Outcome.HOLD_AMBIGUOUS, None
    return Outcome.REWRITE, parsed.private_url


def _target_accepts_media_type(target: Target, media_type: str) -> bool:
    if target in {Target.LISTING_VIDEO}:
        return media_type == "videos"
    if target in {Target.LISTING_DEMO, Target.SKILL_SOURCE}:
        return True
    return media_type == "images"


_HOLD_OUTCOMES = {
    HoldReason.ACTIVE_PUBLIC: Outcome.HOLD_ACTIVE_PUBLIC,
    HoldReason.MARKETPLACE: Outcome.HOLD_MARKETPLACE,
    HoldReason.AMBIGUOUS: Outcome.HOLD_AMBIGUOUS,
}
