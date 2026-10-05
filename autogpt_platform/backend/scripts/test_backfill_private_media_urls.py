from collections import Counter
from unittest.mock import AsyncMock

import pytest

import scripts.backfill_private_media_urls as backfill_cli
import scripts.media_url_backfill as backfill

PRIVATE_BUCKET = "private-media"


@pytest.mark.parametrize(
    "url",
    [
        "https://storage.googleapis.com/private-media/users/owner/images/photo.jpeg",
        "https://private-media.storage.googleapis.com/users/owner/images/photo.jpeg",
        "gs://private-media/users/owner/images/photo.jpeg",
    ],
)
def test_parse_accepts_exact_managed_gcs_urls(url: str):
    parsed = backfill.parse_private_media_url(url, PRIVATE_BUCKET)

    assert parsed is not None
    assert parsed.owner_user_id == "owner"
    assert parsed.media_type == "images"
    assert parsed.filename == "photo.jpeg"
    assert parsed.private_url == (
        "/api/store/submissions/media/owner/images/photo.jpeg"
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://storage.googleapis.com/other/users/owner/images/photo.jpeg",
        "/api/store/submissions/media/owner/images/photo.jpeg",
        "https://example.test/private-media/users/owner/images/photo.jpeg",
    ],
)
def test_parse_ignores_urls_not_raw_private_bucket_urls(url: str):
    assert backfill.parse_private_media_url(url, PRIVATE_BUCKET) is None


@pytest.mark.parametrize(
    "url",
    [
        "https://storage.googleapis.com/private-media/users/owner/images/photo.jpeg?x=1",
        "https://storage.googleapis.com/private-media/users/owner/images/a/b.jpeg",
        "https://storage.googleapis.com/private-media/users/owner/images/%2e%2e",
        "https://storage.googleapis.com/private-media/users/owner/videos/photo.jpeg",
    ],
)
def test_parse_holds_malformed_managed_urls(url: str):
    with pytest.raises(backfill.MalformedPrivateMediaUrl):
        backfill.parse_private_media_url(url, PRIVATE_BUCKET)


def test_scalar_rewrite_requires_owner_proof():
    owned = _candidate(
        target=backfill.Target.PROFILE_AVATAR,
        owner="owner",
        values=[_raw_url("owner", "images", "photo.jpeg")],
    )
    cross_user = _candidate(
        target=backfill.Target.EXPERT_AVATAR,
        owner="owner",
        values=[_raw_url("someone-else", "images", "photo.jpeg")],
    )

    plan = backfill.build_plan([owned, cross_user], PRIVATE_BUCKET)

    assert plan.counts == Counter(
        {
            backfill.Outcome.REWRITE: 1,
            backfill.Outcome.HOLD_CROSS_USER: 1,
        }
    )
    assert len(plan.mutations) == 1
    assert plan.mutations[0].new_values == [
        "/api/store/submissions/media/owner/images/photo.jpeg"
    ]


def test_array_rewrites_only_owned_well_formed_references():
    owned = _raw_url("owner", "images", "owned.png")
    cross_user = _raw_url("other", "images", "cross.png")
    malformed = (
        "https://storage.googleapis.com/private-media/"
        "users/owner/images/nested/file.png"
    )
    external = "https://example.test/external.png"
    candidate = _candidate(
        target=backfill.Target.LISTING_IMAGES,
        owner="owner",
        values=[owned, cross_user, malformed, external],
        is_array=True,
    )

    plan = backfill.build_plan([candidate], PRIVATE_BUCKET)

    assert plan.counts == Counter(
        {
            backfill.Outcome.REWRITE: 1,
            backfill.Outcome.HOLD_CROSS_USER: 1,
            backfill.Outcome.HOLD_MALFORMED: 1,
        }
    )
    assert plan.mutations[0].new_values == [
        "/api/store/submissions/media/owner/images/owned.png",
        cross_user,
        malformed,
        external,
    ]


@pytest.mark.parametrize(
    ("reason", "outcome"),
    [
        (backfill.HoldReason.ACTIVE_PUBLIC, backfill.Outcome.HOLD_ACTIVE_PUBLIC),
        (backfill.HoldReason.MARKETPLACE, backfill.Outcome.HOLD_MARKETPLACE),
        (backfill.HoldReason.AMBIGUOUS, backfill.Outcome.HOLD_AMBIGUOUS),
    ],
)
def test_classification_holds_unsafe_reference_classes(reason, outcome):
    candidate = _candidate(
        target=backfill.Target.PROFILE_AVATAR,
        owner="owner",
        values=[_raw_url("owner", "images", "photo.jpeg")],
        hold_reason=reason,
    )

    plan = backfill.build_plan([candidate], PRIVATE_BUCKET)

    assert plan.counts == Counter({outcome: 1})
    assert plan.mutations == []


def test_candidate_query_allows_owner_proven_shared_rows():
    assert "e.visibility" not in backfill_cli.CANDIDATE_QUERY
    assert "la.visibility" not in backfill_cli.CANDIDATE_QUERY


async def test_dry_run_never_executes_updates():
    client = AsyncMock()
    candidate = _candidate(
        target=backfill.Target.PROFILE_AVATAR,
        owner="owner",
        values=[_raw_url("owner", "images", "photo.jpeg")],
    )

    report = await backfill.process_candidates(
        client, [candidate], PRIVATE_BUCKET, apply=False
    )

    assert report.planned_rows == 1
    assert report.applied_rows == 0
    client.execute_raw.assert_not_awaited()


async def test_apply_uses_compare_and_swap_and_reports_conflict():
    client = AsyncMock()
    client.execute_raw.return_value = 0
    old_url = _raw_url("owner", "images", "photo.jpeg")
    candidate = _candidate(
        target=backfill.Target.PROFILE_AVATAR,
        owner="owner",
        values=[old_url],
    )

    report = await backfill.process_candidates(
        client, [candidate], PRIVATE_BUCKET, apply=True
    )

    assert report.applied_rows == 0
    assert report.cas_conflicts == 1
    query, row_id, replacement, expected, owner = client.execute_raw.await_args.args
    assert 'AND p."avatarUrl" = $3' in query
    assert 'AND p."userId" = $4' in query
    assert "AND NOT EXISTS (" in query
    assert row_id == "record-secret"
    assert replacement == "/api/store/submissions/media/owner/images/photo.jpeg"
    assert expected == old_url
    assert owner == "owner"


async def test_apply_updates_owned_array_with_whole_row_compare_and_swap():
    client = AsyncMock()
    client.execute_raw.return_value = 1
    old_urls = [
        _raw_url("owner", "images", "first.png"),
        "https://example.test/unchanged.png",
    ]
    candidate = _candidate(
        target=backfill.Target.LISTING_IMAGES,
        owner="owner",
        values=old_urls,
        is_array=True,
    )

    report = await backfill.process_candidates(
        client, [candidate], PRIVATE_BUCKET, apply=True
    )

    assert report.applied_rows == 1
    assert report.cas_conflicts == 0
    query, _row_id, replacement, expected, owner = client.execute_raw.await_args.args
    assert 'AND slv."imageUrls" = $3::text[]' in query
    assert 'AND sl."owningUserId" = $4' in query
    assert "AND NOT (" in query
    assert replacement == [
        "/api/store/submissions/media/owner/images/first.png",
        "https://example.test/unchanged.png",
    ]
    assert expected == old_urls
    assert owner == "owner"


@pytest.mark.parametrize(
    "target",
    [backfill.Target.EXPERT_AVATAR, backfill.Target.LIBRARY_IMAGE],
)
async def test_apply_allows_owner_proven_shared_media(target):
    client = AsyncMock()
    client.execute_raw.return_value = 1
    old_url = _raw_url("owner", "images", "shared.png")
    candidate = _candidate(target=target, owner="owner", values=[old_url])

    report = await backfill.process_candidates(
        client, [candidate], PRIVATE_BUCKET, apply=True
    )

    assert report.applied_rows == 1
    query = client.execute_raw.await_args.args[0]
    assert "visibility" not in query


async def test_apply_is_idempotent_for_already_rewritten_url():
    client = AsyncMock()
    candidate = _candidate(
        target=backfill.Target.PROFILE_AVATAR,
        owner="owner",
        values=["/api/store/submissions/media/owner/images/photo.jpeg"],
    )

    report = await backfill.process_candidates(
        client, [candidate], PRIVATE_BUCKET, apply=True
    )

    assert report.planned_rows == 0
    client.execute_raw.assert_not_awaited()


def test_summary_never_prints_identifiers_or_urls(capsys: pytest.CaptureFixture[str]):
    candidate = _candidate(
        target=backfill.Target.PROFILE_AVATAR,
        owner="owner-secret",
        values=[_raw_url("owner-secret", "images", "filename-secret.jpeg")],
    )
    plan = backfill.build_plan([candidate], PRIVATE_BUCKET)
    report = backfill.BackfillReport.from_plan(plan)

    backfill.print_report(report, apply=False)

    output = capsys.readouterr().out
    assert "owner-secret" not in output
    assert "filename-secret" not in output
    assert "record-secret" not in output
    assert "storage.googleapis.com" not in output
    assert "rewrite" in output
    assert "Profile.avatarUrl" in output


def test_organization_reference_is_reported_as_ambiguous_without_a_mutation():
    candidate = _candidate(
        target=backfill.Target.ORGANIZATION_AVATAR,
        owner=None,
        values=[_raw_url("member", "images", "avatar.jpeg")],
        hold_reason=backfill.HoldReason.AMBIGUOUS,
    )

    plan = backfill.build_plan([candidate], PRIVATE_BUCKET)

    assert plan.counts == Counter({backfill.Outcome.HOLD_AMBIGUOUS: 1})
    assert plan.target_counts == {
        backfill.Target.ORGANIZATION_AVATAR: Counter(
            {backfill.Outcome.HOLD_AMBIGUOUS: 1}
        )
    }
    assert plan.mutations == []


def _candidate(
    *,
    target: "backfill.Target",
    owner: str | None,
    values: list[str],
    is_array: bool = False,
    hold_reason: "backfill.HoldReason | None" = None,
):
    return backfill.Candidate(
        target=target,
        record_id="record-secret",
        owner_user_id=owner,
        values=values,
        is_array=is_array,
        hold_reason=hold_reason,
    )


def _raw_url(owner: str, media_type: str, filename: str) -> str:
    return (
        f"https://storage.googleapis.com/{PRIVATE_BUCKET}/"
        f"users/{owner}/{media_type}/{filename}"
    )
