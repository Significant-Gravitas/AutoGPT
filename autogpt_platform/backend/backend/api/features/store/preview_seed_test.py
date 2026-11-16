"""The preview seed's content, checked without a database: it passes the rules
real submissions meet, stays synthetic, and every workflow runs."""

import operator
from collections import defaultdict
from typing import Any

import pytest

from backend.blocks import get_block
from backend.blocks._base import BlockEffect, BlockType
from backend.data.dynamic_fields import merge_execution_input
from backend.data.graph import Graph, make_graph_model
from backend.util.settings import AppEnvironment
from backend.util.type import coerce_inputs_to_schema

from . import model as store_model
from . import skill_model
from .categories import StoreCategory
from .preview_seed import (
    SeedRefused,
    all_users,
    build_graph,
    ensure_seedable,
    parse_seed_skill,
    reviewer_keys,
    seed_graph_id,
    seed_marketplace,
    seed_user_id,
)
from .preview_seed_content import (
    CREATORS,
    EMAIL_DOMAIN,
    LISTINGS,
    REVIEW_COMMENTS,
    SKILLS,
    CalculatorWorkflow,
    CountWorkflow,
    ReplaceWorkflow,
    TemplateWorkflow,
)

CREATOR_KEYS = {creator.key for creator in CREATORS}


async def test_production_is_refused_before_anything_is_written(mocker):
    settings = mocker.patch("backend.api.features.store.preview_seed.Settings")
    settings.return_value.config.app_env = AppEnvironment.PRODUCTION
    settings.return_value.config.sentry_environment = ""
    blocks = mocker.patch(
        "backend.api.features.store.preview_seed._ensure_workflow_blocks"
    )

    with pytest.raises(SeedRefused):
        await seed_marketplace()

    blocks.assert_not_called()


@pytest.mark.parametrize(
    "app_env, sentry_environment, allowed",
    [
        (AppEnvironment.LOCAL, "", True),
        (AppEnvironment.DEVELOPMENT, "pr-15488", True),
        # The shared dev deployment and self-hosted installs.
        (AppEnvironment.DEVELOPMENT, "", False),
        (AppEnvironment.DEVELOPMENT, "dev", False),
        (AppEnvironment.DEVELOPMENT, "pr-", False),
        (AppEnvironment.DEVELOPMENT, "pr-latest", False),
        (AppEnvironment.PRODUCTION, "pr-15488", False),
    ],
)
def test_only_local_databases_and_previews_are_seeded(
    mocker, app_env, sentry_environment, allowed
):
    settings = mocker.patch("backend.api.features.store.preview_seed.Settings")
    settings.return_value.config.app_env = app_env
    settings.return_value.config.sentry_environment = sentry_environment

    if allowed:
        ensure_seedable()
    else:
        with pytest.raises(SeedRefused):
            ensure_seedable()


def test_marketplace_is_small_but_full():
    assert 4 <= len(CREATORS) <= 6
    assert 20 <= len(LISTINGS) <= 40
    assert len(SKILLS) >= 3
    assert {listing.creator for listing in LISTINGS} == CREATOR_KEYS


def test_every_category_leads_some_listings():
    leading = defaultdict(int)
    for listing in LISTINGS:
        leading[listing.categories[0]] += 1
    thin = [c.value for c in StoreCategory if leading[c.value] < 2]
    assert not thin, f"categories with fewer than two listings: {thin}"


def test_listings_meet_the_submission_rules():
    for listing in LISTINGS:
        request = store_model.StoreSubmissionRequest(
            graph_id=seed_graph_id(listing),
            graph_version=1,
            slug=listing.slug,
            name=listing.name,
            sub_heading=listing.sub_heading,
            description=listing.description,
            categories=list(listing.categories),
        )
        # Already canonical, so validation keeps them as they are.
        assert request.categories == list(listing.categories)
        assert request.sub_heading == listing.sub_heading
        # create_store_submission keeps only letters, digits and hyphens, and
        # the detail route lowercases the path, so the slug must already be so.
        assert listing.slug == "".join(
            c for c in listing.slug.lower() if c.isalnum() or c == "-"
        )
    assert len({listing.slug for listing in LISTINGS}) == len(LISTINGS)
    assert len({listing.name for listing in LISTINGS}) == len(LISTINGS)


def test_people_are_obviously_synthetic():
    users = all_users()
    for user in users:
        assert user.email.endswith(f"@{EMAIL_DOMAIN}")
        assert user.name.startswith("Sample ")
        assert user.username.startswith("sample-")
        # update_profile keeps only lowercase letters, digits and hyphens.
        assert user.username == "".join(
            c for c in user.username.lower() if c.isalnum() or c == "-"
        )
        assert all(link.startswith("https://example.com/") for link in user.links)
    assert len({user.username for user in users}) == len(users)
    assert len({user.email for user in users}) == len(users)


def test_reviews_are_star_ratings_from_distinct_non_owners():
    for listing in LISTINGS:
        reviewers = reviewer_keys(listing)
        assert listing.creator not in reviewers
        assert len(reviewers) == len(set(reviewers))
        assert len(listing.reviews) <= len(reviewers)
        assert all(score in REVIEW_COMMENTS for score in listing.reviews)


def test_ratings_differ_so_sorting_by_rating_means_something():
    averages = {
        round(sum(listing.reviews) / len(listing.reviews), 2)
        for listing in LISTINGS
        if listing.reviews
    }
    assert len(averages) >= 5
    assert any(not listing.reviews for listing in LISTINGS)


def test_some_listings_and_creators_are_featured():
    assert 3 <= sum(listing.featured for listing in LISTINGS) <= 8
    assert any(creator.featured for creator in CREATORS)


def test_identities_are_stable():
    assert [seed_user_id(user) for user in all_users()] == [
        seed_user_id(user) for user in all_users()
    ]
    first, second = build_graph(LISTINGS[0]), build_graph(LISTINGS[0])
    assert [node.id for node in first.nodes] == [node.id for node in second.nodes]
    assert [link.id for link in first.links] == [link.id for link in second.links]
    graph_ids = {seed_graph_id(listing) for listing in LISTINGS}
    assert len(graph_ids) == len(LISTINGS)


@pytest.mark.parametrize("listing", LISTINGS, ids=lambda listing: listing.slug)
def test_workflow_validates_for_a_run(listing):
    graph = make_graph_model(build_graph(listing), "user")
    graph.validate_graph(for_run=True)


@pytest.mark.parametrize("listing", LISTINGS, ids=lambda listing: listing.slug)
def test_workflow_uses_only_pure_blocks(listing):
    for node in build_graph(listing).nodes:
        block = get_block(node.block_id)
        assert block is not None
        assert not block.input_schema.get_credentials_fields()
        assert (
            block.block_type in (BlockType.INPUT, BlockType.OUTPUT)
            or block.effect == BlockEffect.NONE
        ), f"{block.name} is not a pure computation"


@pytest.mark.parametrize("listing", LISTINGS, ids=lambda listing: listing.slug)
async def test_workflow_turns_its_example_into_the_expected_output(listing):
    workflow = listing.workflow
    result = await _run(build_graph(listing))

    if isinstance(workflow, CalculatorWorkflow):
        assert result == pytest.approx(_calculate(workflow, float(workflow.example)))
    elif isinstance(workflow, ReplaceWorkflow):
        assert result == workflow.example.replace(workflow.old, workflow.new)
    elif isinstance(workflow, CountWorkflow):
        assert result == len(workflow.example.splitlines())
    else:
        assert isinstance(workflow, TemplateWorkflow)
        assert isinstance(result, str) and result.strip()
        assert "{{" not in result
        if f"{{{{ {workflow.input_name} }}}}" in workflow.template:
            assert workflow.example in result


CALCULATORS = [
    listing for listing in LISTINGS if isinstance(listing.workflow, CalculatorWorkflow)
]


@pytest.mark.parametrize("listing", CALCULATORS, ids=lambda listing: listing.slug)
async def test_an_amount_keeps_its_cents(listing):
    workflow = listing.workflow
    assert isinstance(workflow, CalculatorWorkflow)

    result = await _run(build_graph(listing), entered="100.55")

    assert result == pytest.approx(_calculate(workflow, 100.55))


COUNTERS = [
    listing for listing in LISTINGS if isinstance(listing.workflow, CountWorkflow)
]


@pytest.mark.parametrize("listing", COUNTERS, ids=lambda listing: listing.slug)
async def test_blank_lines_are_not_counted(listing):
    result = await _run(build_graph(listing), entered="\nfirst\n\n  second  \n\n")

    assert result == 2


@pytest.mark.parametrize(
    "url, linkedin",
    [
        (
            "https://example.com/launch",
            "https://example.com/launch?utm_source=linkedin&utm_medium=social"
            "&utm_campaign=launch",
        ),
        (
            "https://example.com/launch?ref=partner#pricing",
            "https://example.com/launch?ref=partner&utm_source=linkedin"
            "&utm_medium=social&utm_campaign=launch#pricing",
        ),
    ],
)
async def test_utm_links_keep_the_query_and_fragment(url, linkedin):
    listing = next(item for item in LISTINGS if item.slug == "utm-link-builder")

    result = await _run(build_graph(listing), entered=url)

    assert result.splitlines()[0] == f"LinkedIn: {linkedin}"


async def test_the_news_digest_numbers_each_item():
    listing = next(item for item in LISTINGS if item.slug == "news-digest-formatter")

    result = await _run(
        build_graph(listing), entered="Rates held steady\n\n  Rules proposed \n"
    )

    assert "1. Rates held steady\n2. Rules proposed" in result


@pytest.mark.parametrize("skill", SKILLS, ids=lambda skill: skill.slug)
def test_skill_is_a_valid_library_skill(skill):
    parsed = parse_seed_skill(skill)
    assert parsed.name == skill.slug
    assert skill.creator in CREATOR_KEYS
    request = skill_model.SkillSubmissionRequest(
        skill_name=skill.slug, categories=list(skill.categories)
    )
    assert request.categories == list(skill.categories)
    assert skill_model.skill_title(parsed.name, parsed.body)


def test_skill_slugs_stay_out_of_the_catalogs_way():
    # Marketplace skill slugs are global, and the catalog publisher refuses a
    # slug a user owns, so these keep a prefix the catalog does not use.
    assert all(skill.slug.startswith("sample-") for skill in SKILLS)
    assert len({skill.slug for skill in SKILLS}) == len(SKILLS)


def _calculate(workflow: CalculatorWorkflow, amount: float) -> float:
    apply = {
        "Add": operator.add,
        "Subtract": operator.sub,
        "Multiply": operator.mul,
        "Divide": operator.truediv,
    }[workflow.operation]
    return apply(amount, workflow.operand)


async def _run(graph: Graph, entered: str | None = None) -> Any:
    """Run a linear seed workflow in-process, the way the executor feeds each
    node its defaults plus whatever its incoming links carry, coerced to the
    node's input types. ``entered`` replaces the input's default value, as a
    person filling in the run form would."""
    incoming = defaultdict(list)
    for link in graph.links:
        incoming[link.sink_id].append(link)
    produced: dict[tuple[str, str], Any] = {}
    for position, node in enumerate(graph.nodes):
        block = get_block(node.block_id)
        assert block is not None
        data = dict(node.input_default)
        if position == 0 and entered is not None:
            data["value"] = entered
        for link in incoming[node.id]:
            data[link.sink_name] = produced[(link.source_id, link.source_name)]
        data = {
            name: value
            for name, value in merge_execution_input(data).items()
            if "_#_" not in name
        }
        coerce_inputs_to_schema(data, block.input_schema)
        async for name, value in block.run(block.input_schema(**data)):
            produced[(node.id, name)] = value
    return produced[(graph.nodes[-1].id, "output")]
