from collections import Counter
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest

import scripts.backfill_private_media_urls as backfill_cli
import scripts.media_url_backfill as backfill
import scripts.media_url_backfill_queries as queries

PRIVATE_BUCKET = "private-media"


@pytest.mark.parametrize(
    "url",
    [
        "https://storage.googleapis.com/private-media/users/owner/images/photo.jpeg",
        "https://private-media.storage.googleapis.com/users/owner/images/photo.jpeg",
        "gs://private-media/users/owner/images/photo.jpeg",
        "https://storage.cloud.google.com/private-media/users/owner/images/photo.jpeg",
        "https://commondatastorage.googleapis.com/private-media/"
        + "users/owner/images/photo.jpeg",
        "https://storage.googleapis.com/storage/v1/b/private-media/o/"
        + "users%2Fowner%2Fimages%2Fphoto.jpeg?alt=media",
        "https://www.googleapis.com/download/storage/v1/b/private-media/o/"
        + "users%2Fowner%2Fimages%2Fphoto.jpeg",
        "  https://storage.googleapis.com/private-media/users/owner/images/photo.jpeg ",
        "https://storage.googleapis.com/private-media/users/owner/images/photo.jpeg?v=2",
        "/_next/image?url=https%3A%2F%2Fstorage.googleapis.com%2Fprivate-media"
        + "%2Fusers%2Fowner%2Fimages%2Fphoto.jpeg&w=640&q=75",
    ],
)
def test_parse_accepts_every_backend_gcs_url_form(url: str):
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
        "https://storage.googleapis.com/private-media-public/users/o/images/a.png",
    ],
)
def test_parse_ignores_urls_not_raw_private_bucket_urls(url: str):
    assert backfill.parse_private_media_url(url, PRIVATE_BUCKET) is None


@pytest.mark.parametrize(
    "url",
    [
        "https://storage.googleapis.com/private-media/users/owner/images/a/b.jpeg",
        "https://storage.googleapis.com/private-media/users/owner/images/%2e%2e",
        "https://storage.googleapis.com/private-media/users/owner/videos/photo.jpeg",
        "https://storage.googleapis.com/private-media/users/owner/images/my photo.png",
        "https://storage.googleapis.com/private-media/oauth-apps/app/logo/logo.png",
    ],
)
def test_parse_holds_malformed_managed_urls(url: str):
    with pytest.raises(backfill.MalformedPrivateMediaUrl):
        backfill.parse_private_media_url(url, PRIVATE_BUCKET)


@pytest.mark.parametrize(
    "url",
    [
        "https://storage.googleapis.com/private-media/users/owner/images/photo.jpeg",
        "https://storage.googleapis.com/other/users/owner/images/photo.jpeg",
        "https://private-media.storage.googleapis.com/users/owner/videos/clip.mp4",
        "https://storage.cloud.google.com/private-media/users/o/images/a.png#x",
        "https://commondatastorage.googleapis.com/private-media/users/o/images/a.png",
        "https://storage.googleapis.com/storage/v1/b/private-media/o/"
        + "users%2Fo%2Fimages%2Fa.png?alt=media",
        "https://www.googleapis.com/download/storage/v1/b/private-media/o/"
        + "users%2Fo%2Fimages%2Fa.png",
        "gs://private-media/users/owner/images/photo.jpeg",
        " gs://private-media/users/owner/images/photo.jpeg\n",
        "/_next/image?url=https%3A%2F%2Fstorage.googleapis.com%2Fprivate-media"
        + "%2Fusers%2Fo%2Fimages%2Fa.png&w=640",
        "https://platform.example/_next/image?url=gs%3A%2F%2Fprivate-media"
        + "%2Fusers%2Fo%2Fimages%2Fa.png",
        "/api/store/submissions/media/owner/images/photo.jpeg",
        "/api/store/submissions/media/owner/images/photo.txt",
        "/api/store/submissions/media/owner/images/a/b.png",
        "/_next/image?url=%2Fapi%2Fstore%2Fsubmissions%2Fmedia%2Fo%2Fimages%2Fa.png",
        "https://storage.googleapis.com/private-media/users/o/images/legacy name.png",
        "https://example.test/private-media/users/owner/images/photo.jpeg",
        "not a url",
    ],
)
def test_url_parser_matches_backend_parser(url: str):
    from backend.api.features.store import public_media

    ours = backfill.private_api_object_path(url) or backfill.gcs_object_path(
        url, PRIVATE_BUCKET
    )

    assert ours == public_media.object_path_from_url(url, PRIVATE_BUCKET)


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
        (backfill.HoldReason.PUBLIC, backfill.Outcome.HOLD_PUBLIC),
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


@pytest.mark.parametrize(
    "value",
    [
        "https://storage.googleapis.com/private-media/users/other/images/a.png",
        "https://storage.googleapis.com/private-media/users/owner/images/a/b.png",
        "https://storage.googleapis.com/private-media/users/owner/videos/a.png",
    ],
)
def test_hold_reason_is_reported_before_owner_and_shape_checks(value: str):
    candidate = _candidate(
        target=backfill.Target.LISTING_IMAGES,
        owner="owner",
        values=[value],
        is_array=True,
        hold_reason=backfill.HoldReason.PUBLIC,
    )

    plan = backfill.build_plan([candidate], PRIVATE_BUCKET)

    assert plan.counts == Counter({backfill.Outcome.HOLD_PUBLIC: 1})
    assert plan.mutations == []


def test_selected_but_unparsed_references_are_counted_as_unrecognized():
    candidate = _candidate(
        target=backfill.Target.LISTING_IMAGES,
        owner="owner",
        values=[
            "https://example.test/private-media/users/owner/images/photo.jpeg",
            "https://cdn.example/private-media.png",
            "https://example.test/unrelated.png",
        ],
        is_array=True,
    )

    plan = backfill.build_plan([candidate], PRIVATE_BUCKET)

    assert plan.counts == Counter({backfill.Outcome.UNRECOGNIZED: 2})
    assert plan.mutations == []


def test_public_bucket_urls_are_not_unrecognized_when_names_overlap():
    candidate = _candidate(
        target=backfill.Target.PROFILE_AVATAR,
        owner="owner",
        values=[
            "https://storage.googleapis.com/private-media-public/"
            "users/owner/images/photo.jpeg"
        ],
    )

    plan = backfill.build_plan(
        [candidate], PRIVATE_BUCKET, public_bucket="private-media-public"
    )

    assert plan.counts == Counter({backfill.Outcome.ALREADY_PUBLIC: 1})
    assert plan.mutations == []


def test_rewrite_accepts_query_and_wrapper_forms():
    candidate = _candidate(
        target=backfill.Target.PROFILE_AVATAR,
        owner="owner",
        values=[
            "/_next/image?url=https%3A%2F%2Fstorage.googleapis.com%2Fprivate-media"
            "%2Fusers%2Fowner%2Fimages%2Fphoto.jpeg%3Fv%3D3&w=640"
        ],
    )

    plan = backfill.build_plan([candidate], PRIVATE_BUCKET)

    assert plan.counts == Counter({backfill.Outcome.REWRITE: 1})
    assert plan.mutations[0].new_values == [
        "/api/store/submissions/media/owner/images/photo.jpeg"
    ]


@pytest.mark.parametrize(
    "target",
    [
        backfill.Target.ORGANIZATION_AVATAR,
        backfill.Target.ORGANIZATION_PROFILE_AVATAR,
    ],
)
def test_org_avatars_are_rewritten_only_while_the_uploader_is_a_member(target):
    member_upload = _raw_url("member", "images", "avatar.jpeg")
    former_member_upload = _raw_url("former", "images", "avatar.jpeg")
    candidates = [
        _candidate(target=target, owner=None, values=[url], co_owners=["member"])
        for url in (member_upload, former_member_upload)
    ]

    plan = backfill.build_plan(candidates, PRIVATE_BUCKET)

    assert plan.counts == Counter(
        {backfill.Outcome.REWRITE: 1, backfill.Outcome.HOLD_CROSS_USER: 1}
    )
    assert plan.mutations[0].owner_user_id is None
    assert plan.mutations[0].new_values == [
        "/api/store/submissions/media/member/images/avatar.jpeg"
    ]


def test_library_rows_are_rewritten_only_to_media_their_owner_can_open():
    own = _raw_url("owner", "images", "own.png")
    colleague = _raw_url("colleague", "images", "shared.png")
    creator = _raw_url("creator", "images", "listing.png")
    candidates = [
        _candidate(
            target=backfill.Target.LIBRARY_IMAGE,
            owner="owner",
            values=[url],
            co_owners=["colleague"],
        )
        for url in (own, colleague, creator)
    ]

    plan = backfill.build_plan(candidates, PRIVATE_BUCKET)

    assert plan.counts == Counter(
        {backfill.Outcome.REWRITE: 2, backfill.Outcome.HOLD_CROSS_USER: 1}
    )
    assert [mutation.owner_user_id for mutation in plan.mutations] == [
        "owner",
        "owner",
    ]


def test_candidate_query_allows_owner_proven_shared_rows():
    assert "e.visibility" not in backfill_cli.CANDIDATE_QUERY
    assert "la.visibility" not in backfill_cli.CANDIDATE_QUERY


def test_candidate_query_leaves_logos_and_attribution_urls_alone():
    assert "OAuthApplication" not in backfill_cli.CANDIDATE_QUERY
    assert "SkillListingVersion.sourceUrl" not in backfill_cli.CANDIDATE_QUERY
    assert "marketplace" not in backfill_cli.CANDIDATE_QUERY


def test_candidate_query_covers_every_update_query():
    for target in queries.UPDATE_QUERIES:
        assert f"'{target}'" in backfill_cli.CANDIDATE_QUERY


def test_select_and_update_share_null_safe_public_predicates():
    assert "activeVersionId" not in queries.VERSION_IS_PUBLIC
    assert backfill_cli.CANDIDATE_QUERY.count(queries.VERSION_IS_PUBLIC) == 3
    assert queries.CREATOR_IS_PUBLIC in backfill_cli.CANDIDATE_QUERY
    assert queries.CREATOR_IS_PUBLIC in queries.UPDATE_QUERIES["Profile.avatarUrl"]
    for column in ("imageUrls", "videoUrl", "agentOutputDemoUrl"):
        query = queries.UPDATE_QUERIES[f"StoreListingVersion.{column}"]
        assert f"AND NOT {queries.VERSION_IS_PUBLIC}" in query
    assert "SkillListing" in queries.CREATOR_IS_PUBLIC


def test_updates_never_touch_updated_at():
    for query in queries.UPDATE_QUERIES.values():
        assert "updatedAt" not in query


def test_only_owner_checked_updates_take_an_owner_parameter():
    for target, query in queries.UPDATE_QUERIES.items():
        assert ("$4" in query) == (target not in queries.PATH_OWNER_TARGETS)


async def test_dry_run_never_executes_updates():
    client = AsyncMock()
    transaction, opened = _transactions(client)
    candidate = _candidate(
        target=backfill.Target.PROFILE_AVATAR,
        owner="owner",
        values=[_raw_url("owner", "images", "photo.jpeg")],
    )

    report = await backfill.process_candidates(
        [candidate], PRIVATE_BUCKET, apply=False, transaction=transaction
    )

    assert report.planned_rows == 1
    assert report.applied_rows == 0
    assert opened == []
    client.execute_raw.assert_not_awaited()


async def test_apply_uses_compare_and_swap_and_reports_conflict():
    client = AsyncMock()
    client.execute_raw.return_value = 0
    transaction, _ = _transactions(client)
    old_url = _raw_url("owner", "images", "photo.jpeg")
    candidate = _candidate(
        target=backfill.Target.PROFILE_AVATAR,
        owner="owner",
        values=[old_url],
    )

    report = await backfill.process_candidates(
        [candidate], PRIVATE_BUCKET, apply=True, transaction=transaction
    )

    assert report.applied_rows == 0
    assert report.cas_conflicts == 1
    query, row_id, replacement, expected, owner = client.execute_raw.await_args.args
    assert 'AND p."avatarUrl" = $3' in query
    assert 'AND p."userId" = $4' in query
    assert "AND NOT (" in query
    assert row_id == "record-secret"
    assert replacement == "/api/store/submissions/media/owner/images/photo.jpeg"
    assert expected == old_url
    assert owner == "owner"


async def test_apply_updates_owned_array_with_whole_row_compare_and_swap():
    client = AsyncMock()
    client.execute_raw.return_value = 1
    transaction, _ = _transactions(client)
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
        [candidate], PRIVATE_BUCKET, apply=True, transaction=transaction
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


async def test_apply_allows_owner_proven_shared_expert_media():
    client = AsyncMock()
    client.execute_raw.return_value = 1
    transaction, _ = _transactions(client)
    old_url = _raw_url("owner", "images", "shared.png")
    candidate = _candidate(
        target=backfill.Target.EXPERT_AVATAR, owner="owner", values=[old_url]
    )

    report = await backfill.process_candidates(
        [candidate], PRIVATE_BUCKET, apply=True, transaction=transaction
    )

    assert report.applied_rows == 1
    query = client.execute_raw.await_args.args[0]
    assert "visibility" not in query


async def test_apply_rewrites_a_library_image_only_for_its_owner():
    client = AsyncMock()
    client.execute_raw.return_value = 1
    transaction, _ = _transactions(client)
    old_url = _raw_url("owner", "images", "agent.png")
    candidate = _candidate(
        target=backfill.Target.LIBRARY_IMAGE, owner="owner", values=[old_url]
    )

    report = await backfill.process_candidates(
        [candidate], PRIVATE_BUCKET, apply=True, transaction=transaction
    )

    assert report.applied_rows == 1
    query, row_id, replacement, expected, owner = client.execute_raw.await_args.args
    assert 'la."userId" = $4' in query
    assert row_id == "record-secret"
    assert replacement == "/api/store/submissions/media/owner/images/agent.png"
    assert expected == old_url
    assert owner == "owner"


async def test_apply_commits_in_short_batches_and_keeps_progress_on_failure():
    client = AsyncMock()
    client.execute_raw.side_effect = [1] * 400 + [RuntimeError("db down")]
    transaction, opened = _transactions(client)
    candidates = [
        _candidate(
            target=backfill.Target.LIBRARY_IMAGE,
            owner="creator",
            values=[_raw_url("creator", "images", f"{index}.png")],
        )
        for index in range(450)
    ]
    progress = backfill.ApplyProgress()

    with pytest.raises(RuntimeError):
        await backfill.process_candidates(
            candidates,
            PRIVATE_BUCKET,
            apply=True,
            transaction=transaction,
            progress=progress,
        )

    assert len(opened) == 3
    assert progress.applied_rows == 400


async def test_apply_splits_mutations_into_batches_of_the_configured_size():
    client = AsyncMock()
    client.execute_raw.return_value = 1
    transaction, opened = _transactions(client)
    candidates = [
        _candidate(
            target=backfill.Target.LIBRARY_IMAGE,
            owner="creator",
            values=[_raw_url("creator", "images", f"{index}.png")],
        )
        for index in range(backfill.APPLY_BATCH_SIZE * 2 + 1)
    ]

    report = await backfill.process_candidates(
        candidates, PRIVATE_BUCKET, apply=True, transaction=transaction
    )

    assert len(opened) == 3
    assert report.applied_rows == backfill.APPLY_BATCH_SIZE * 2 + 1


async def test_apply_is_idempotent_for_already_rewritten_url():
    client = AsyncMock()
    transaction, opened = _transactions(client)
    candidate = _candidate(
        target=backfill.Target.PROFILE_AVATAR,
        owner="owner",
        values=["/api/store/submissions/media/owner/images/photo.jpeg"],
    )

    report = await backfill.process_candidates(
        [candidate], PRIVATE_BUCKET, apply=True, transaction=transaction
    )

    assert report.planned_rows == 0
    assert opened == []
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
    assert "unrecognized" in output
    assert "Profile.avatarUrl" in output


def test_source_bucket_defaults_to_the_private_bucket():
    assert (
        backfill.resolve_source_bucket(
            private_bucket="legacy", legacy_bucket="", override=None
        )
        == "legacy"
    )
    assert (
        backfill.resolve_source_bucket(
            private_bucket="legacy", legacy_bucket="legacy", override=None
        )
        == "legacy"
    )


def test_source_bucket_refuses_to_guess_when_buckets_diverge():
    with pytest.raises(SystemExit, match="--bucket"):
        backfill.resolve_source_bucket(
            private_bucket="new-private", legacy_bucket="legacy", override=None
        )

    assert (
        backfill.resolve_source_bucket(
            private_bucket="new-private", legacy_bucket="legacy", override="legacy"
        )
        == "legacy"
    )


def test_source_bucket_requires_a_bucket():
    with pytest.raises(SystemExit):
        backfill.resolve_source_bucket(
            private_bucket="", legacy_bucket="", override=None
        )


def _transactions(client):
    opened: list[int] = []

    @asynccontextmanager
    async def transaction():
        opened.append(len(opened))
        yield client

    return transaction, opened


def _candidate(
    *,
    target: "backfill.Target",
    owner: str | None,
    values: list[str],
    is_array: bool = False,
    hold_reason: "backfill.HoldReason | None" = None,
    co_owners: list[str] | None = None,
):
    return backfill.Candidate(
        target=target,
        record_id="record-secret",
        owner_user_id=owner,
        values=values,
        is_array=is_array,
        hold_reason=hold_reason,
        co_owner_ids=co_owners or [],
    )


def _raw_url(owner: str, media_type: str, filename: str) -> str:
    return (
        f"https://storage.googleapis.com/{PRIVATE_BUCKET}/"
        f"users/{owner}/{media_type}/{filename}"
    )


def test_org_listing_rewrites_media_uploaded_by_an_active_member():
    member_image = _raw_url("member", "images", "shot.png")
    stranger_image = _raw_url("stranger", "images", "shot.png")
    candidate = _candidate(
        target=backfill.Target.LISTING_IMAGES,
        owner="owner",
        values=[member_image, stranger_image],
        is_array=True,
        co_owners=["owner", "member"],
    )

    plan = backfill.build_plan([candidate], PRIVATE_BUCKET)

    assert plan.mutations[0].owner_user_id == "owner"
    assert plan.mutations[0].new_values == [
        "/api/store/submissions/media/member/images/shot.png",
        stranger_image,
    ]
    assert plan.counts[backfill.Outcome.HOLD_CROSS_USER] == 1


@pytest.mark.parametrize(
    "outcome, blocks",
    [
        (backfill.Outcome.HOLD_CROSS_USER, True),
        (backfill.Outcome.HOLD_MALFORMED, True),
        (backfill.Outcome.HOLD_AMBIGUOUS, True),
        (backfill.Outcome.HOLD_PUBLIC, True),
        (backfill.Outcome.UNRECOGNIZED, True),
        (backfill.Outcome.REWRITE, False),
        (backfill.Outcome.ALREADY_PUBLIC, False),
    ],
)
def test_references_left_on_the_legacy_bucket_fail_the_run(outcome, blocks):
    report = backfill.BackfillReport(
        counts=Counter({outcome: 1}), target_counts={}, planned_rows=0
    )

    assert bool(backfill_cli.stranded_references(report)) is blocks
