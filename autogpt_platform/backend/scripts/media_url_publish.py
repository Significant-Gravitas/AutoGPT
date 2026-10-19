from __future__ import annotations

import asyncio
import re
from collections import Counter
from collections.abc import Iterable
from enum import StrEnum
from typing import NamedTuple, Protocol
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict

if __package__:
    from scripts.media_url_backfill import (
        PRIVATE_MEDIA_PREFIX,
        ApplyProgress,
        Mutation,
        Target,
        gcs_object_path,
        private_api_object_path,
    )
else:
    from media_url_backfill import (
        PRIVATE_MEDIA_PREFIX,
        ApplyProgress,
        Mutation,
        Target,
        gcs_object_path,
        private_api_object_path,
    )

COPY_CONCURRENCY = 8

_MEDIA_PATH = re.compile(
    r"users/(?P<owner>[^/]+)/(?:images|videos)/(?P<filename>[^/]+)"
)
_LOGO_PATH = re.compile(r"oauth-apps/(?P<app_id>[^/]+)/logo/(?P<filename>[^/]+)")


class PublishOutcome(StrEnum):
    PUBLISH = "publish"
    ALREADY_PUBLIC = "already_public"
    COPY_FAILED = "copy_failed"
    SKIP_FOREIGN_OWNER = "skip_foreign_owner"
    SKIP_MALFORMED = "skip_malformed"
    SKIP_UNRECOGNIZED = "skip_unrecognized"


class Buckets(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: str
    private: str
    public: str


class SourceObject(NamedTuple):
    bucket: str
    path: str


class PublishCandidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    target: Target
    record_id: str
    values: list[str]
    is_array: bool
    owner_ids: list[str] = []


class PlannedReference(BaseModel):
    model_config = ConfigDict(frozen=True)

    candidate: PublishCandidate
    index: int
    source: SourceObject


class PublishPlan(BaseModel):
    references: list[PlannedReference]
    counts: dict[Target, Counter[PublishOutcome]]
    already_public_image_paths: set[str]

    @property
    def objects(self) -> set[SourceObject]:
        return {reference.source for reference in self.references}


class PublishReport(ApplyProgress):
    counts: dict[Target, Counter[PublishOutcome]]
    planned_objects: int
    copied_objects: int = 0
    planned_rows: int = 0


class ObjectCopier(Protocol):
    async def copy(
        self,
        bucket: str,
        object_name: str,
        destination_bucket: str,
        *,
        new_name: str | None = None,
    ) -> object: ...


def public_url(public_bucket: str, path: str) -> str:
    return f"https://storage.googleapis.com/{public_bucket}/{quote(path, safe='/')}"


def locate_object(value: str, buckets: Buckets) -> tuple[str, str] | None:
    """The bucket and object path a reference to private data points at."""
    for bucket in dict.fromkeys((buckets.source, buckets.private)):
        if (path := gcs_object_path(value, bucket)) is not None:
            return bucket, path
    if (path := private_api_object_path(value)) is not None:
        return buckets.private, path
    return None


def plan_publication(
    candidates: Iterable[PublishCandidate], buckets: Buckets
) -> PublishPlan:
    references: list[PlannedReference] = []
    counts: dict[Target, Counter[PublishOutcome]] = {}
    already_public_image_paths: set[str] = set()
    for candidate in candidates:
        if not candidate.is_array and len(candidate.values) != 1:
            raise ValueError("Scalar candidate must have exactly one value")
        for index, value in enumerate(candidate.values):
            if (public_path := gcs_object_path(value, buckets.public)) is not None:
                _count(counts, candidate.target, PublishOutcome.ALREADY_PUBLIC)
                if candidate.target == Target.LISTING_IMAGES:
                    already_public_image_paths.add(public_path)
                continue
            located = locate_object(value, buckets)
            if located is None:
                if _mentions_private_data(value, buckets):
                    _count(counts, candidate.target, PublishOutcome.SKIP_UNRECOGNIZED)
                continue
            bucket, path = located
            if skip := _check_path(candidate, path):
                _count(counts, candidate.target, skip)
                continue
            references.append(
                PlannedReference(
                    candidate=candidate,
                    index=index,
                    source=SourceObject(bucket=bucket, path=path),
                )
            )
    return PublishPlan(
        references=references,
        counts=counts,
        already_public_image_paths=already_public_image_paths,
    )


async def copy_objects(
    objects: Iterable[SourceObject], public_bucket: str, copier: ObjectCopier
) -> set[SourceObject]:
    """Copy objects to the public bucket under the same path; return successes."""
    semaphore = asyncio.Semaphore(COPY_CONCURRENCY)

    async def copy_one(source: SourceObject) -> SourceObject | None:
        async with semaphore:
            try:
                await copier.copy(
                    source.bucket, source.path, public_bucket, new_name=source.path
                )
            except Exception:
                return None
            return source

    results = await asyncio.gather(*(copy_one(source) for source in set(objects)))
    return {source for source in results if source is not None}


def build_mutations(
    plan: PublishPlan, copied: set[SourceObject], public_bucket: str
) -> tuple[list[Mutation], dict[Target, Counter[PublishOutcome]]]:
    counts = {target: Counter(counter) for target, counter in plan.counts.items()}
    new_values: dict[tuple[Target, str], list[str]] = {}
    candidates: dict[tuple[Target, str], PublishCandidate] = {}
    for reference in plan.references:
        candidate = reference.candidate
        if reference.source not in copied:
            _count(counts, candidate.target, PublishOutcome.COPY_FAILED)
            continue
        _count(counts, candidate.target, PublishOutcome.PUBLISH)
        key = (candidate.target, candidate.record_id)
        candidates[key] = candidate
        values = new_values.setdefault(key, list(candidate.values))
        values[reference.index] = public_url(public_bucket, reference.source.path)
    mutations = [
        Mutation(
            target=candidate.target,
            record_id=candidate.record_id,
            owner_user_id=None,
            old_values=candidate.values,
            new_values=new_values[key],
            is_array=candidate.is_array,
        )
        for key, candidate in candidates.items()
    ]
    return mutations, counts


def published_image_paths(plan: PublishPlan, copied: set[SourceObject]) -> set[str]:
    """Listing image objects that are public once this run's copies are done."""
    return plan.already_public_image_paths | {
        reference.source.path
        for reference in plan.references
        if reference.candidate.target == Target.LISTING_IMAGES
        and reference.source in copied
    }


def library_mutations(
    rows: Iterable[tuple[str, str]],
    image_paths: set[str],
    buckets: Buckets,
    counts: dict[Target, Counter[PublishOutcome]],
) -> list[Mutation]:
    """Point library copies of a published listing image at its public copy."""
    mutations: list[Mutation] = []
    for record_id, value in rows:
        located = locate_object(value, buckets)
        if located is None or located[1] not in image_paths:
            continue
        _count(counts, Target.LIBRARY_IMAGE, PublishOutcome.PUBLISH)
        mutations.append(
            Mutation(
                target=Target.LIBRARY_IMAGE,
                record_id=record_id,
                owner_user_id=None,
                old_values=[value],
                new_values=[public_url(buckets.public, located[1])],
                is_array=False,
            )
        )
    return mutations


def print_report(report: PublishReport, *, apply: bool) -> None:
    print("Live media publication summary (aggregate counts only):")
    totals: Counter[PublishOutcome] = Counter()
    for target_counts in report.counts.values():
        totals.update(target_counts)
    for outcome in PublishOutcome:
        print(f"  {outcome.value}: {totals[outcome]}")
    for target in Target:
        target_counts = report.counts.get(target)
        if not target_counts:
            continue
        summary = ", ".join(
            f"{outcome.value}={target_counts[outcome]}"
            for outcome in PublishOutcome
            if target_counts[outcome]
        )
        print(f"  {target.value}: {summary}")
    print(f"  planned_object_copies: {report.planned_objects}")
    print(f"  planned_row_updates: {report.planned_rows}")
    if apply:
        print(f"  copied_objects: {report.copied_objects}")
        print(f"  applied_row_updates: {report.applied_rows}")
        print(f"  compare_and_swap_conflicts: {report.cas_conflicts}")
    else:
        print(
            "Dry run: no objects were copied and no database writes were "
            "attempted. Re-run with --apply."
        )


def _check_path(candidate: PublishCandidate, path: str) -> PublishOutcome | None:
    if candidate.target == Target.OAUTH_LOGO:
        match = _LOGO_PATH.fullmatch(path)
        if not match or not _is_segment(match["filename"]):
            return PublishOutcome.SKIP_MALFORMED
        if match["app_id"] != candidate.record_id:
            return PublishOutcome.SKIP_FOREIGN_OWNER
        return None
    match = _MEDIA_PATH.fullmatch(path)
    if (
        not match
        or not _is_segment(match["owner"])
        or not _is_segment(match["filename"])
    ):
        return PublishOutcome.SKIP_MALFORMED
    if match["owner"] not in candidate.owner_ids:
        return PublishOutcome.SKIP_FOREIGN_OWNER
    return None


def _is_segment(value: str) -> bool:
    return value not in {".", ".."}


def _mentions_private_data(value: str, buckets: Buckets) -> bool:
    return (
        buckets.source in value
        or buckets.private in value
        or PRIVATE_MEDIA_PREFIX in value
    )


def _count(
    counts: dict[Target, Counter[PublishOutcome]],
    target: Target,
    outcome: PublishOutcome,
) -> None:
    counts.setdefault(target, Counter())[outcome] += 1
