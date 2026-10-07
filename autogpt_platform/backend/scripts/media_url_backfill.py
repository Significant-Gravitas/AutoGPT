from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from enum import StrEnum
from pathlib import PurePosixPath
from typing import LiteralString
from urllib.parse import parse_qs, unquote, urlsplit

from prisma import Prisma
from pydantic import BaseModel, ConfigDict

if __package__:
    from scripts.media_url_backfill_queries import PATH_OWNER_TARGETS, UPDATE_QUERIES
else:
    from media_url_backfill_queries import PATH_OWNER_TARGETS, UPDATE_QUERIES

PRIVATE_MEDIA_PREFIX = "/api/store/submissions/media"
APPLY_BATCH_SIZE = 200

Transaction = Callable[[], AbstractAsyncContextManager[Prisma]]

# Same URL forms as backend.api.features.store.public_media (parity-tested).
_GCS_URL_FORMS = (
    re.compile(
        r"https?://(?:www|storage)\.googleapis\.com/(?:download/)?storage/v1"
        r"/b/(?P<bucket>[^/?#]+)/o/(?P<path>[^?#]+)(?:[?#].*)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"https?://(?:storage\.googleapis\.com|storage\.cloud\.google\.com"
        r"|commondatastorage\.googleapis\.com)/(?P<bucket>[^/?#]+)/(?P<path>[^?#]+)"
        r"(?:[?#].*)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"https?://(?P<bucket>[a-z0-9._-]+)\.storage\.googleapis\.com"
        r"/(?P<path>[^?#]+)(?:[?#].*)?",
        re.IGNORECASE,
    ),
    re.compile(r"gs://(?P<bucket>[^/?#]+)/(?P<path>[^?#]+)", re.IGNORECASE),
)
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


class HoldReason(StrEnum):
    PUBLIC = "public"
    AMBIGUOUS = "ambiguous"


class Outcome(StrEnum):
    REWRITE = "rewrite"
    HOLD_PUBLIC = "hold_public"
    HOLD_AMBIGUOUS = "hold_ambiguous"
    HOLD_CROSS_USER = "hold_cross_user"
    HOLD_MALFORMED = "hold_malformed"
    UNRECOGNIZED = "unrecognized"
    ALREADY_PUBLIC = "already_public"


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
    # Users whose uploads the row's readers can open through the private media
    # endpoint: active members of a listing's or avatar's org, or the org
    # colleagues of a library owner.
    co_owner_ids: list[str] = []


class Mutation(BaseModel):
    model_config = ConfigDict(frozen=True)

    target: Target
    record_id: str
    owner_user_id: str | None
    old_values: list[str]
    new_values: list[str]
    is_array: bool


class BackfillPlan(BaseModel):
    mutations: list[Mutation]
    counts: Counter[Outcome]
    target_counts: dict[Target, Counter[Outcome]]


class ApplyProgress(BaseModel):
    applied_rows: int = 0
    cas_conflicts: int = 0


class BackfillReport(ApplyProgress):
    counts: Counter[Outcome]
    target_counts: dict[Target, Counter[Outcome]]
    planned_rows: int

    @classmethod
    def from_plan(cls, plan: BackfillPlan) -> BackfillReport:
        return cls(
            counts=plan.counts,
            target_counts=plan.target_counts,
            planned_rows=len(plan.mutations),
        )


def resolve_source_bucket(
    *, private_bucket: str, legacy_bucket: str, override: str | None
) -> str:
    """The bucket whose stored URLs are processed: the legacy bucket."""
    if override:
        return override
    if legacy_bucket and private_bucket and legacy_bucket != private_bucket:
        raise SystemExit(
            "PRIVATE_USER_DATA_BUCKET differs from MEDIA_GCS_BUCKET_NAME, so stored "
            "URLs may point at either bucket. Re-run with --bucket set to the "
            "bucket the stored URLs point at."
        )
    if not private_bucket:
        raise SystemExit(
            "PRIVATE_USER_DATA_BUCKET or legacy MEDIA_GCS_BUCKET_NAME must be set"
        )
    return private_bucket


def gcs_object_path(url: str, bucket: str) -> str | None:
    """Object path of a GCS URL in ``bucket``, in any form the backend accepts."""
    url = unquote(_unwrap(url))
    for form in _GCS_URL_FORMS:
        match = form.fullmatch(url)
        if match:
            return match["path"] if match["bucket"] == bucket else None
    return None


def private_api_object_path(url: str) -> str | None:
    """Object path of a private media API URL, validated like the endpoint."""
    parsed = urlsplit(_unwrap(url))
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        return None
    path = unquote(parsed.path)
    if not path.startswith(f"{PRIVATE_MEDIA_PREFIX}/"):
        return None
    parts = path.removeprefix(f"{PRIVATE_MEDIA_PREFIX}/").split("/")
    if len(parts) != 3:
        return None
    owner_user_id, media_type, filename = parts
    if (
        media_type not in {"images", "videos"}
        or not _safe_path_component(owner_user_id)
        or not _safe_path_component(filename)
        or PurePosixPath(filename).suffix.lower() not in _CONTENT_TYPE_BY_SUFFIX
    ):
        return None
    return f"users/{owner_user_id}/{media_type}/{filename}"


def parse_private_media_url(url: str, private_bucket: str) -> ParsedPrivateMedia | None:
    object_path = gcs_object_path(url, private_bucket)
    if object_path is None:
        return None
    return parse_private_media_path(object_path)


def parse_private_media_path(object_path: str) -> ParsedPrivateMedia:
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


def build_plan(
    candidates: list[Candidate],
    private_bucket: str,
    public_bucket: str | None = None,
) -> BackfillPlan:
    counts: Counter[Outcome] = Counter()
    target_counts: dict[Target, Counter[Outcome]] = {}
    mutations: list[Mutation] = []
    for candidate in candidates:
        if not candidate.is_array and len(candidate.values) != 1:
            raise ValueError("Scalar candidate must have exactly one value")
        new_values = list(candidate.values)
        changed = False
        for index, value in enumerate(candidate.values):
            outcome, replacement = _classify_reference(
                candidate, value, private_bucket, public_bucket
            )
            if outcome is None:
                continue
            counts[outcome] += 1
            target_counts.setdefault(candidate.target, Counter())[outcome] += 1
            if replacement is not None and replacement != value:
                new_values[index] = replacement
                changed = True
        if changed:
            owner_user_id = None
            if candidate.target.value not in PATH_OWNER_TARGETS:
                if candidate.owner_user_id is None:
                    raise RuntimeError("A rewrite candidate has no proven owner")
                owner_user_id = candidate.owner_user_id
            mutations.append(
                Mutation(
                    target=candidate.target,
                    record_id=candidate.record_id,
                    owner_user_id=owner_user_id,
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
    candidates: list[Candidate],
    private_bucket: str,
    *,
    public_bucket: str | None = None,
    apply: bool,
    transaction: Transaction | None = None,
    progress: ApplyProgress | None = None,
) -> BackfillReport:
    plan = build_plan(candidates, private_bucket, public_bucket)
    report = BackfillReport.from_plan(plan)
    if not apply:
        return report
    if transaction is None:
        raise ValueError("Applying a backfill needs a transaction factory")

    await apply_mutations(
        plan.mutations, UPDATE_QUERIES, transaction, progress or report
    )
    if progress is not None:
        report.applied_rows = progress.applied_rows
        report.cas_conflicts = progress.cas_conflicts
    return report


async def apply_mutations(
    mutations: Sequence[Mutation],
    queries: Mapping[str, LiteralString],
    transaction: Transaction,
    progress: ApplyProgress,
    *,
    batch_size: int = APPLY_BATCH_SIZE,
) -> None:
    """Apply compare-and-swap updates in short transactions of ``batch_size``.

    Committed batches stay committed if a later one fails; re-running is safe
    because every update matches only the value it was planned from.
    """
    for start in range(0, len(mutations), batch_size):
        applied = conflicts = 0
        async with transaction() as client:
            for mutation in mutations[start : start + batch_size]:
                changed = await client.execute_raw(
                    queries[mutation.target.value], *_update_args(mutation)
                )
                if changed == 1:
                    applied += 1
                elif changed == 0:
                    conflicts += 1
                else:
                    raise RuntimeError(
                        "A compare-and-swap update affected multiple rows"
                    )
        progress.applied_rows += applied
        progress.cas_conflicts += conflicts


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


def _update_args(mutation: Mutation) -> list[str | list[str]]:
    if mutation.is_array:
        args: list[str | list[str]] = [
            mutation.record_id,
            mutation.new_values,
            mutation.old_values,
        ]
    else:
        args = [mutation.record_id, mutation.new_values[0], mutation.old_values[0]]
    if mutation.owner_user_id is not None:
        args.append(mutation.owner_user_id)
    return args


def _unwrap(url: str) -> str:
    url = url.strip()
    wrapped = urlsplit(url)
    if wrapped.path == "/_next/image":
        url = parse_qs(wrapped.query).get("url", [""])[0].strip()
    return url


def _safe_path_component(value: str) -> bool:
    return (
        value not in {".", ".."} and _SAFE_PATH_COMPONENT.fullmatch(value) is not None
    )


def _classify_reference(
    candidate: Candidate,
    value: str,
    private_bucket: str,
    public_bucket: str | None,
) -> tuple[Outcome | None, str | None]:
    if private_bucket not in value:
        return None, None
    if public_bucket and gcs_object_path(value, public_bucket) is not None:
        return Outcome.ALREADY_PUBLIC, None
    object_path = gcs_object_path(value, private_bucket)
    if object_path is None:
        if private_api_object_path(value) is not None:
            return None, None
        return Outcome.UNRECOGNIZED, None
    if candidate.hold_reason is not None:
        return _HOLD_OUTCOMES[candidate.hold_reason], None
    try:
        parsed = parse_private_media_path(object_path)
    except MalformedPrivateMediaUrl:
        return Outcome.HOLD_MALFORMED, None
    if not _target_accepts_media_type(candidate.target, parsed.media_type):
        return Outcome.HOLD_MALFORMED, None
    if candidate.target.value in PATH_OWNER_TARGETS:
        if parsed.owner_user_id not in candidate.co_owner_ids:
            return Outcome.HOLD_CROSS_USER, None
    else:
        if candidate.owner_user_id is None:
            return Outcome.HOLD_AMBIGUOUS, None
        if parsed.owner_user_id != candidate.owner_user_id and (
            parsed.owner_user_id not in candidate.co_owner_ids
        ):
            return Outcome.HOLD_CROSS_USER, None
    return Outcome.REWRITE, parsed.private_url


def _target_accepts_media_type(target: Target, media_type: str) -> bool:
    if target == Target.LISTING_VIDEO:
        return media_type == "videos"
    if target == Target.LISTING_DEMO:
        return True
    return media_type == "images"


_HOLD_OUTCOMES = {
    HoldReason.PUBLIC: Outcome.HOLD_PUBLIC,
    HoldReason.AMBIGUOUS: Outcome.HOLD_AMBIGUOUS,
}
