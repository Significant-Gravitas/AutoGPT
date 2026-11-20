import re
from collections import Counter
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest

import scripts.media_url_backfill_queries as queries
import scripts.media_url_publish as publish
import scripts.publish_live_media as publish_cli
from scripts.media_url_backfill import Target

PRIVATE = "legacy-media"
PUBLIC = "site-media"
BUCKETS = publish.Buckets(source=PRIVATE, private=PRIVATE, public=PUBLIC)
Outcome = publish.PublishOutcome


def test_plan_publishes_only_media_owned_by_the_listing_or_its_org_members():
    owned = _private_url("owner", "images", "a.png")
    member = _private_url("member", "videos", "clip.mp4")
    foreign = _private_url("stranger", "images", "b.png")
    nested = f"https://storage.googleapis.com/{PRIVATE}/users/owner/images/x/y.png"
    traversal = f"https://storage.googleapis.com/{PRIVATE}/users/owner/images/%2e%2e"
    external = "https://example.test/c.png"
    candidate = _candidate(
        Target.LISTING_IMAGES,
        [owned, member, foreign, nested, traversal, external],
        owner_ids=["owner", "member"],
        is_array=True,
    )

    plan = publish.plan_publication([candidate], BUCKETS)

    assert [reference.source.path for reference in plan.references] == [
        "users/owner/images/a.png",
        "users/member/videos/clip.mp4",
    ]
    assert plan.counts == {
        Target.LISTING_IMAGES: Counter(
            {Outcome.SKIP_FOREIGN_OWNER: 1, Outcome.SKIP_MALFORMED: 2}
        )
    }


def test_plan_accepts_legacy_filenames_and_url_forms_the_new_validator_rejects():
    candidate = _candidate(
        Target.PROFILE_AVATAR,
        [
            f"/_next/image?url=gs%3A%2F%2F{PRIVATE}%2Fusers%2Fowner%2Fimages"
            "%2FMy%20Photo%20(1).PNG&w=64"
        ],
        owner_ids=["owner"],
    )

    plan = publish.plan_publication([candidate], BUCKETS)

    assert plan.objects == {
        publish.SourceObject(bucket=PRIVATE, path="users/owner/images/My Photo (1).PNG")
    }


def test_plan_skips_already_public_and_counts_unrecognized_private_references():
    candidate = _candidate(
        Target.LISTING_IMAGES,
        [
            _public_url("users/owner/images/a.png"),
            f"https://cdn.example/{PRIVATE}/users/owner/images/a.png",
            "/api/store/submissions/media/owner/images/legacy name.png",
        ],
        owner_ids=["owner"],
        is_array=True,
    )

    plan = publish.plan_publication([candidate], BUCKETS)

    assert plan.references == []
    assert plan.counts == {
        Target.LISTING_IMAGES: Counter(
            {Outcome.ALREADY_PUBLIC: 1, Outcome.SKIP_UNRECOGNIZED: 2}
        )
    }
    assert plan.already_public_image_paths == {"users/owner/images/a.png"}


def test_plan_repairs_private_api_references_from_the_private_bucket():
    buckets = publish.Buckets(source="legacy", private="new-private", public=PUBLIC)
    candidate = _candidate(
        Target.LISTING_VIDEO,
        ["/api/store/submissions/media/owner/videos/clip.mp4"],
        owner_ids=["owner"],
    )

    plan = publish.plan_publication([candidate], buckets)

    assert plan.objects == {
        publish.SourceObject(bucket="new-private", path="users/owner/videos/clip.mp4")
    }


def test_plan_publishes_oauth_logos_only_under_their_own_app_prefix():
    own = _candidate(
        Target.OAUTH_LOGO,
        [f"https://storage.googleapis.com/{PRIVATE}/oauth-apps/app-1/logo/l.png"],
        record_id="app-1",
    )
    other = _candidate(
        Target.OAUTH_LOGO,
        [f"https://storage.googleapis.com/{PRIVATE}/oauth-apps/app-1/logo/l.png"],
        record_id="app-2",
    )
    user_media = _candidate(
        Target.OAUTH_LOGO, [_private_url("owner", "images", "a.png")], record_id="app-3"
    )

    plan = publish.plan_publication([own, other, user_media], BUCKETS)

    assert plan.objects == {
        publish.SourceObject(bucket=PRIVATE, path="oauth-apps/app-1/logo/l.png")
    }
    assert plan.counts == {
        Target.OAUTH_LOGO: Counter(
            {Outcome.SKIP_FOREIGN_OWNER: 1, Outcome.SKIP_MALFORMED: 1}
        )
    }


async def test_copy_objects_copies_each_object_once_and_reports_failures():
    copier = _Copier(fail={"users/owner/images/missing.png"})
    objects = [
        publish.SourceObject(bucket=PRIVATE, path="users/owner/images/a.png"),
        publish.SourceObject(bucket=PRIVATE, path="users/owner/images/a.png"),
        publish.SourceObject(bucket=PRIVATE, path="users/owner/images/missing.png"),
    ]

    copied = await publish.copy_objects(objects, PUBLIC, copier)

    assert copied == {
        publish.SourceObject(bucket=PRIVATE, path="users/owner/images/a.png")
    }
    assert sorted(copier.calls) == [
        (PRIVATE, "users/owner/images/a.png", PUBLIC, "users/owner/images/a.png"),
        (
            PRIVATE,
            "users/owner/images/missing.png",
            PUBLIC,
            "users/owner/images/missing.png",
        ),
    ]


def test_build_mutations_rewrites_only_copied_references():
    copied_url = _private_url("owner", "images", "a b.png")
    failed_url = _private_url("owner", "images", "missing.png")
    candidate = _candidate(
        Target.LISTING_IMAGES,
        [copied_url, failed_url, "https://example.test/x.png"],
        owner_ids=["owner"],
        is_array=True,
    )
    plan = publish.plan_publication([candidate], BUCKETS)
    copied = {publish.SourceObject(bucket=PRIVATE, path="users/owner/images/a b.png")}

    mutations, counts = publish.build_mutations(plan, copied, PUBLIC)

    assert len(mutations) == 1
    assert mutations[0].owner_user_id is None
    assert mutations[0].old_values == candidate.values
    assert mutations[0].new_values == [
        f"https://storage.googleapis.com/{PUBLIC}/users/owner/images/a%20b.png",
        failed_url,
        "https://example.test/x.png",
    ]
    assert counts == {
        Target.LISTING_IMAGES: Counter({Outcome.PUBLISH: 1, Outcome.COPY_FAILED: 1})
    }


async def test_dry_run_copies_nothing_and_writes_nothing():
    client = _client(
        listings=[_listing_row([_private_url("owner", "images", "a.png")])],
        library=[("lib-1", _private_url("owner", "images", "a.png"))],
    )
    copier_factory, copier = _copier_factory()
    transaction, opened = _transactions(client)

    report = await publish_cli._run(
        client,
        BUCKETS,
        apply=False,
        transaction=transaction,
        copier_factory=copier_factory,
        progress=publish.ApplyProgress(),
    )

    assert copier.calls == []
    assert opened == []
    client.execute_raw.assert_not_awaited()
    assert report.planned_objects == 1
    assert report.planned_rows == 2
    assert report.counts[Target.LIBRARY_IMAGE] == Counter({Outcome.PUBLISH: 1})


async def test_apply_copies_then_rewrites_listing_profile_logo_and_library_rows():
    image = _private_url("owner", "images", "a.png")
    avatar = _private_url("owner", "images", "avatar.png")
    logo = f"https://storage.googleapis.com/{PRIVATE}/oauth-apps/app-1/logo/l.png"
    client = _client(
        listings=[
            _listing_row([image], video_url=_private_url("owner", "videos", "v.mp4"))
        ],
        profiles=[{"record_id": "profile-1", "owner_id": "owner", "value": avatar}],
        logos=[{"record_id": "app-1", "value": logo}],
        library=[
            ("lib-1", image),
            ("lib-2", "/api/store/submissions/media/owner/images/a.png"),
            ("lib-3", _private_url("owner", "images", "not-listed.png")),
        ],
    )
    client.execute_raw.return_value = 1
    copier_factory, copier = _copier_factory()
    transaction, opened = _transactions(client)

    report = await publish_cli._run(
        client,
        BUCKETS,
        apply=True,
        transaction=transaction,
        copier_factory=copier_factory,
        progress=publish.ApplyProgress(),
    )

    assert {call[1] for call in copier.calls} == {
        "users/owner/images/a.png",
        "users/owner/videos/v.mp4",
        "users/owner/images/avatar.png",
        "oauth-apps/app-1/logo/l.png",
    }
    assert all(call[0] == PRIVATE and call[2] == PUBLIC for call in copier.calls)
    assert len(opened) == 1
    updates = {
        (args[1], _set_column(args[0])): args[2:]
        for args in (call.args for call in client.execute_raw.await_args_list)
    }
    assert updates == {
        ("version-1", "imageUrls"): (
            [_public_url("users/owner/images/a.png")],
            [image],
        ),
        ("version-1", "videoUrl"): (
            _public_url("users/owner/videos/v.mp4"),
            _private_url("owner", "videos", "v.mp4"),
        ),
        ("profile-1", "avatarUrl"): (
            _public_url("users/owner/images/avatar.png"),
            avatar,
        ),
        ("app-1", "logoUrl"): (_public_url("oauth-apps/app-1/logo/l.png"), logo),
        ("lib-1", "imageUrl"): (_public_url("users/owner/images/a.png"), image),
        ("lib-2", "imageUrl"): (
            _public_url("users/owner/images/a.png"),
            "/api/store/submissions/media/owner/images/a.png",
        ),
    }
    assert client.execute_raw.await_count == 6
    assert report.applied_rows == 6
    assert report.copied_objects == 4


async def test_apply_skips_rows_whose_copy_failed():
    image = _private_url("owner", "images", "missing.png")
    client = _client(listings=[_listing_row([image])], library=[("lib-1", image)])
    copier_factory, _ = _copier_factory(fail={"users/owner/images/missing.png"})
    transaction, opened = _transactions(client)

    report = await publish_cli._run(
        client,
        BUCKETS,
        apply=True,
        transaction=transaction,
        copier_factory=copier_factory,
        progress=publish.ApplyProgress(),
    )

    client.execute_raw.assert_not_awaited()
    assert report.counts == {Target.LISTING_IMAGES: Counter({Outcome.COPY_FAILED: 1})}


async def test_rerun_is_idempotent_and_repairs_stale_library_copies():
    published = _public_url("users/owner/images/a.png")
    client = _client(
        listings=[_listing_row([published])],
        library=[("lib-1", _private_url("owner", "images", "a.png"))],
    )
    client.execute_raw.return_value = 1
    copier_factory, copier = _copier_factory()
    transaction, _ = _transactions(client)

    report = await publish_cli._run(
        client,
        BUCKETS,
        apply=True,
        transaction=transaction,
        copier_factory=copier_factory,
        progress=publish.ApplyProgress(),
    )

    assert copier.calls == []
    assert client.execute_raw.await_count == 1
    query, row_id, new_value, old_value = client.execute_raw.await_args.args
    assert 'platform."LibraryAgent"' in query
    assert (row_id, new_value) == ("lib-1", published)
    assert old_value == _private_url("owner", "images", "a.png")
    assert report.counts[Target.LISTING_IMAGES] == Counter({Outcome.ALREADY_PUBLIC: 1})


def test_report_never_prints_identifiers_or_urls(capsys: pytest.CaptureFixture[str]):
    report = publish.PublishReport(
        counts={Target.PROFILE_AVATAR: Counter({Outcome.PUBLISH: 1})},
        planned_objects=1,
        planned_rows=1,
    )

    publish.print_report(report, apply=True)

    output = capsys.readouterr().out
    assert "storage.googleapis.com" not in output
    assert "users/" not in output
    assert "Profile.avatarUrl: publish=1" in output
    assert "copy_failed: 0" in output


def test_publish_queries_are_compare_and_swap_without_updated_at():
    for target, query in queries.PUBLISH_UPDATE_QUERIES.items():
        column = target.split(".")[1]
        assert "updatedAt" not in query
        assert "WHERE id = $1" in query
        assert f'"{column}" = $3' in query
        assert "$4" not in query


def test_selection_matches_the_backfill_holds():
    assert queries.VERSION_IS_PUBLIC in publish_cli.LISTING_QUERY
    assert queries.CREATOR_IS_PUBLIC in publish_cli.PROFILE_QUERY
    assert "om.status = 'ACTIVE'" in publish_cli.LISTING_QUERY
    assert 'om."orgId" = sl."owningOrgId"' in publish_cli.LISTING_QUERY


def _set_column(query: str) -> str:
    match = re.search(r'SET "(\w+)"', query)
    assert match
    return match[1]


class _Copier:
    def __init__(self, fail: set[str] | None = None):
        self.fail = fail or set()
        self.calls: list[tuple[str, str, str, str | None]] = []

    async def copy(self, bucket, object_name, destination_bucket, *, new_name=None):
        self.calls.append((bucket, object_name, destination_bucket, new_name))
        if object_name in self.fail:
            raise RuntimeError("404")
        return {}


def _copier_factory(fail: set[str] | None = None):
    copier = _Copier(fail)

    @asynccontextmanager
    async def factory():
        yield copier

    return factory, copier


def _transactions(client):
    opened: list[int] = []

    @asynccontextmanager
    async def transaction():
        opened.append(len(opened))
        yield client

    return transaction, opened


def _client(
    *,
    listings: list[dict] | None = None,
    profiles: list[dict] | None = None,
    logos: list[dict] | None = None,
    library: list[tuple[str, str]] | None = None,
):
    results = {
        publish_cli.LISTING_QUERY: listings or [],
        publish_cli.PROFILE_QUERY: profiles or [],
        publish_cli.OAUTH_LOGO_QUERY: logos or [],
        publish_cli.LIBRARY_QUERY: [
            {"record_id": record_id, "value": value}
            for record_id, value in library or []
        ],
    }
    client = AsyncMock()

    async def query_raw(query, *args):
        return results[query]

    client.query_raw.side_effect = query_raw
    return client


def _listing_row(
    image_urls: list[str],
    *,
    video_url: str | None = None,
    demo_url: str | None = None,
    owner_ids: list[str] | None = None,
):
    return {
        "record_id": "version-1",
        "image_urls": image_urls,
        "video_url": video_url,
        "demo_url": demo_url,
        "owner_ids": owner_ids or ["owner"],
    }


def _candidate(
    target: Target,
    values: list[str],
    *,
    owner_ids: list[str] | None = None,
    is_array: bool = False,
    record_id: str = "record-secret",
):
    return publish.PublishCandidate(
        target=target,
        record_id=record_id,
        values=values,
        is_array=is_array,
        owner_ids=owner_ids or [],
    )


def _private_url(owner: str, media_type: str, filename: str) -> str:
    return f"https://storage.googleapis.com/{PRIVATE}/users/{owner}/{media_type}/{filename}"


def _public_url(path: str) -> str:
    return f"https://storage.googleapis.com/{PUBLIC}/{path}"


@pytest.mark.parametrize(
    "outcome, blocks",
    [
        (publish.PublishOutcome.SKIP_FOREIGN_OWNER, True),
        (publish.PublishOutcome.SKIP_MALFORMED, True),
        (publish.PublishOutcome.SKIP_UNRECOGNIZED, True),
        (publish.PublishOutcome.COPY_FAILED, True),
        (publish.PublishOutcome.ALREADY_PUBLIC, False),
        (publish.PublishOutcome.PUBLISH, False),
    ],
)
def test_every_unpublished_live_reference_fails_the_run(outcome, blocks):
    report = publish.PublishReport(
        counts={Target.LISTING_IMAGES: Counter({outcome: 1})},
        planned_objects=0,
        copied_objects=0,
        planned_rows=0,
    )

    assert bool(publish_cli.unpublished_references(report)) is blocks
