import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import prisma.enums
import prisma.models
import pytest

import backend.api.features.store.model as store_model
from backend.api.features.experts import experts_db, scheduling, seed
from backend.api.features.experts.models import ExpertSoulUpdate
from backend.api.model import CreateGraph
from backend.blocks.io import AgentInputBlock
from backend.data.graph import Graph, Node
from backend.data.user import get_or_create_user
from backend.usecases.sample import create_test_user
from backend.util.exceptions import ExpertRunPausedError
from backend.util.test import SpinTestServer


@pytest.fixture(scope="session", autouse=True)
def mock_embedding_functions():
    """Mock embedding functions to avoid database/API dependencies
    (mirrors backend/data/graph_test.py)."""
    with patch(
        "backend.api.features.store.db.ensure_embedding",
        new_callable=AsyncMock,
        return_value=True,
    ):
        yield


@pytest.fixture
async def test_user():
    return await create_test_user()


@pytest.fixture
async def other_user():
    return await create_test_user(alt_user=True)


async def _create_seed_user():
    suffix = uuid.uuid4().hex[:8]
    return await get_or_create_user(
        {
            "sub": str(uuid.uuid4()),
            "email": f"expert-seed-{suffix}@example.com",
            "name": "Seed Owner",
        }
    )


async def _seed_store_listing(server: SpinTestServer, approved: bool = True) -> str:
    """Create a graph plus a store listing on top of it.

    Returns the StoreListingVersion ID, ready for
    ``add_store_agent_to_library``. With ``approved=False`` the version is
    left in its submitted PENDING state. Mirrors the seeding pattern from
    ``backend/data/graph_test.py::test_access_store_listing_graph``.
    """
    owner = await _create_seed_user()
    admin = await _create_seed_user()

    graph = Graph(
        name=f"Expert seed graph {uuid.uuid4().hex[:8]}",
        description="Seed graph for expert workflow installs",
        nodes=[
            Node(
                block_id=AgentInputBlock().id,
                input_default={"name": "input_1"},
            ),
        ],
        links=[],
    )
    created_graph = await server.agent_server.test_create_graph(
        CreateGraph(graph=graph), owner.id
    )

    # A Profile is required for store submissions; get_or_create_user
    # already ensures one for every user.
    listing = await server.agent_server.test_create_store_listing(
        store_model.StoreSubmissionRequest(
            graph_id=created_graph.id,
            graph_version=created_graph.version,
            slug=created_graph.id,
            name=f"Seed listing {uuid.uuid4().hex[:8]}",
            sub_heading="Seed sub heading",
            video_url=None,
            image_urls=[],
            description="Seed description",
            categories=[],
        ),
        owner.id,
    )
    slv_id = listing.listing_version_id
    assert slv_id is not None, "Failed to create store listing"

    if approved:
        await server.agent_server.test_review_store_listing(
            store_model.ReviewSubmissionRequest(
                store_listing_version_id=slv_id,
                is_approved=True,
                comments="seed",
            ),
            user_id=admin.id,
        )
    return slv_id


async def _seed_template(
    name: str,
    preload_listings: list[str],
    preload_crons: dict[str, str] | None = None,
) -> prisma.models.Expert:
    """Create an Expert roster template plus ExpertWorkflow preload rows.

    The stored name gets a unique suffix so ad-hoc test templates never
    collide with seed.ROSTER's real roster names — seed._upsert_template
    resolves templates by name, and a bare "Maria" here would be silently
    adopted and overwritten by test_seed_roster_round_trip.
    """
    template = await prisma.models.Expert.prisma().create(
        data={
            "name": f"{name} {uuid.uuid4().hex[:8]}",
            "role": f"{name}'s role",
            "identity": f"You are {name}, an expert.",
            "isTemplate": True,
        }
    )
    for slv_id in preload_listings:
        await prisma.models.ExpertWorkflow.prisma().create(
            data={
                "expertId": template.id,
                "storeListingVersionId": slv_id,
                "scheduleCron": (preload_crons or {}).get(slv_id),
            }
        )
    return template


@pytest.mark.asyncio(loop_scope="session")
async def test_hire_expert_is_idempotent(server: SpinTestServer, test_user):
    template = await _seed_template(name="Maria", preload_listings=[])
    first = await experts_db.hire_expert(test_user.id, template.id, None)
    second = await experts_db.hire_expert(test_user.id, template.id, None)
    assert first.expert.id == second.expert.id
    assert not first.expert.is_template
    assert first.expert.source_template_id == template.id


@pytest.mark.asyncio(loop_scope="session")
async def test_raise_expert_creates_blank_owned_expert(server: SpinTestServer):
    owner = await _create_seed_user()
    raised = await experts_db.create_raised_expert(
        owner.id,
        name="Otto",
        role=None,
        voice_preferences=None,
        first_job_store_listing_version_id=None,
    )
    assert not raised.expert.is_template
    assert raised.expert.source_template_id is None
    assert raised.expert.name == "Otto"
    assert "Otto" in raised.expert.identity
    assert raised.expert.workflows == []
    assert raised.first_job_installed is False
    assert raised.expert.id in {e.id for e in await experts_db.list_experts(owner.id)}


@pytest.mark.asyncio(loop_scope="session")
async def test_raise_expert_allows_multiple_per_owner(server: SpinTestServer):
    owner = await _create_seed_user()
    first = await experts_db.create_raised_expert(owner.id, "Otto", None, None, None)
    second = await experts_db.create_raised_expert(owner.id, "Nova", None, None, None)
    assert first.expert.id != second.expert.id
    owned = {e.id for e in await experts_db.list_experts(owner.id)}
    assert {first.expert.id, second.expert.id} <= owned


@pytest.mark.asyncio(loop_scope="session")
async def test_raise_expert_is_scoped_to_owner(server: SpinTestServer, other_user):
    owner = await _create_seed_user()
    raised = await experts_db.create_raised_expert(owner.id, "Otto", None, None, None)
    assert await experts_db.get_expert(other_user.id, raised.expert.id) is None
    assert await experts_db.get_expert(owner.id, raised.expert.id) is not None


@pytest.mark.asyncio(loop_scope="session")
async def test_raise_expert_installs_first_job(server: SpinTestServer):
    owner = await _create_seed_user()
    slv_id = await _seed_store_listing(server)
    raised = await experts_db.create_raised_expert(
        owner.id,
        name="Nova",
        role="Research Assistant",
        voice_preferences="Warm and detailed.",
        first_job_store_listing_version_id=slv_id,
    )
    assert raised.expert.role == "Research Assistant"
    assert raised.expert.voice_preferences == "Warm and detailed."
    assert raised.first_job_installed is True
    assert len(raised.expert.workflows) == 1
    assert raised.expert.workflows[0].store_listing_version_id == slv_id


@pytest.mark.asyncio(loop_scope="session")
async def test_raise_expert_reports_failed_first_job(server: SpinTestServer):
    owner = await _create_seed_user()
    slv_id = await _seed_store_listing(server)
    with patch.object(
        experts_db.library_db,
        "add_store_agent_to_library",
        new_callable=AsyncMock,
        side_effect=RuntimeError("install exploded"),
    ):
        raised = await experts_db.create_raised_expert(
            owner.id, "Otto", None, None, slv_id
        )
    assert not raised.expert.is_template
    assert raised.expert.workflows == []
    assert raised.first_job_installed is False


@pytest.mark.asyncio(loop_scope="session")
async def test_raise_expert_handles_braces_in_name(server: SpinTestServer):
    owner = await _create_seed_user()
    raised = await experts_db.create_raised_expert(owner.id, "a{b", None, None, None)
    assert raised.expert.name == "a{b"
    assert "a{b" in raised.expert.identity


@pytest.mark.asyncio(loop_scope="session")
async def test_raise_expert_degrades_when_first_job_vanishes_mid_flight(
    server: SpinTestServer,
):
    """Pre-create validation passes, but the listing is withdrawn before the
    install — the raise must degrade honestly, not link the stale row."""
    owner = await _create_seed_user()
    slv_id = await _seed_store_listing(server)
    with patch.object(
        experts_db,
        "_validate_first_job_listing",
        new_callable=AsyncMock,
        side_effect=[None, experts_db.FirstJobUnavailableError(slv_id)],
    ):
        raised = await experts_db.create_raised_expert(
            owner.id, "Otto", None, None, slv_id
        )
    assert raised.expert.workflows == []
    assert raised.first_job_installed is False


@pytest.mark.asyncio(loop_scope="session")
async def test_raise_expert_rejects_unapproved_first_job(server: SpinTestServer):
    owner = await _create_seed_user()
    pending_slv_id = await _seed_store_listing(server, approved=False)

    with pytest.raises(experts_db.FirstJobUnavailableError):
        await experts_db.create_raised_expert(
            owner.id, "Otto", None, None, pending_slv_id
        )

    assert await experts_db.list_experts(owner.id) == []


@pytest.mark.asyncio(loop_scope="session")
async def test_raise_expert_enforces_active_expert_cap(server: SpinTestServer):
    owner = await _create_seed_user()
    await prisma.models.Expert.prisma().create_many(
        data=[
            {
                "ownerUserId": owner.id,
                "name": f"Filler {i}",
                "role": "",
                "identity": f"I'm Filler {i}.",
            }
            for i in range(experts_db.ACTIVE_EXPERT_LIMIT)
        ]
    )

    with pytest.raises(experts_db.ExpertLimitExceededError):
        await experts_db.create_raised_expert(
            owner.id, "One Too Many", None, None, None
        )

    filler = await prisma.models.Expert.prisma().find_first(
        where={"ownerUserId": owner.id}
    )
    assert filler is not None
    await experts_db.archive_expert(owner.id, filler.id)

    raised = await experts_db.create_raised_expert(
        owner.id, "Fits Now", None, None, None
    )
    assert raised.expert.name == "Fits Now"


@pytest.mark.asyncio(loop_scope="session")
async def test_rehire_after_archive_revives_expert(server: SpinTestServer, test_user):
    template = await _seed_template(name="Maria", preload_listings=[])
    hired = await experts_db.hire_expert(test_user.id, template.id, None)
    await experts_db.archive_expert(test_user.id, hired.expert.id)

    revived = await experts_db.hire_expert(test_user.id, template.id, None)

    assert revived.expert.id == hired.expert.id
    assert not revived.expert.is_archived
    assert hired.expert.id in {
        e.id for e in await experts_db.list_experts(test_user.id)
    }


@pytest.mark.asyncio(loop_scope="session")
async def test_install_workflow_on_archived_expert_raises(
    server: SpinTestServer, test_user
):
    slv_id = await _seed_store_listing(server)
    template = await _seed_template(name="Maria", preload_listings=[])
    hired = await experts_db.hire_expert(test_user.id, template.id, None)
    await experts_db.archive_expert(test_user.id, hired.expert.id)

    with pytest.raises(experts_db.ExpertNotFoundError):
        await experts_db.install_workflow(test_user.id, hired.expert.id, slv_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_hire_installs_preloads_into_library(server: SpinTestServer, test_user):
    slv_id = await _seed_store_listing(server)
    template = await _seed_template(name="Maria", preload_listings=[slv_id])
    result = await experts_db.hire_expert(test_user.id, template.id, None)
    wf = result.expert.workflows[0]
    assert wf.library_agent_id is not None
    assert wf.store_listing_version_id == slv_id
    assert result.failed_preloads == []


@pytest.mark.asyncio(loop_scope="session")
async def test_hire_reports_failed_preload_without_sinking_hire(
    server: SpinTestServer, test_user
):
    slv_id = await _seed_store_listing(server)
    template = await _seed_template(name="Maria", preload_listings=[slv_id])
    with patch.object(
        experts_db.library_db,
        "add_store_agent_to_library",
        new_callable=AsyncMock,
        side_effect=RuntimeError("install exploded"),
    ):
        result = await experts_db.hire_expert(test_user.id, template.id, None)
    assert not result.expert.is_template
    assert result.expert.workflows == []
    assert len(result.failed_preloads) == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_get_expert_scopes_by_owner(
    server: SpinTestServer, test_user, other_user
):
    template = await _seed_template(name="Maria", preload_listings=[])
    hired = await experts_db.hire_expert(test_user.id, template.id, None)
    assert await experts_db.get_expert(other_user.id, hired.expert.id) is None
    assert await experts_db.get_expert(test_user.id, hired.expert.id) is not None


@pytest.mark.asyncio(loop_scope="session")
async def test_get_expert_excludes_archived_experts(server: SpinTestServer, test_user):
    template = await _seed_template(name="Maria", preload_listings=[])
    hired = await experts_db.hire_expert(test_user.id, template.id, None)
    await experts_db.archive_expert(test_user.id, hired.expert.id)

    assert await experts_db.get_expert(test_user.id, hired.expert.id) is None


@pytest.mark.asyncio(loop_scope="session")
async def test_owner_can_update_expert_soul(server: SpinTestServer, test_user):
    template = await _seed_template(name="Maria", preload_listings=[])
    hired = await experts_db.hire_expert(test_user.id, template.id, None)

    updated = await experts_db.update_soul(
        test_user.id,
        hired.expert.id,
        ExpertSoulUpdate(
            name="Mara",
            identity="You are Mara, a thoughtful strategist.",
            voice_preferences="Warm, concise, and direct.",
            boundaries="Never invent customer evidence.",
        ),
    )

    assert updated.name == "Mara"
    assert updated.identity == "You are Mara, a thoughtful strategist."
    assert updated.voice_preferences == "Warm, concise, and direct."
    assert updated.boundaries == "Never invent customer evidence."


@pytest.mark.asyncio(loop_scope="session")
async def test_other_user_cannot_update_expert_soul(
    server: SpinTestServer, test_user, other_user
):
    template = await _seed_template(name="Maria", preload_listings=[])
    hired = await experts_db.hire_expert(test_user.id, template.id, None)

    with pytest.raises(experts_db.ExpertNotFoundError):
        await experts_db.update_soul(
            other_user.id,
            hired.expert.id,
            ExpertSoulUpdate(
                name="Stolen",
                identity=hired.expert.identity,
                voice_preferences="",
                boundaries="",
            ),
        )


@pytest.mark.asyncio(loop_scope="session")
async def test_templates_and_archived_experts_cannot_update_soul(
    server: SpinTestServer, test_user
):
    template = await _seed_template(name="Maria", preload_listings=[])
    hired = await experts_db.hire_expert(test_user.id, template.id, None)
    await experts_db.archive_expert(test_user.id, hired.expert.id)
    soul = ExpertSoulUpdate(
        name="Mara",
        identity="You are Mara.",
        voice_preferences="Direct.",
        boundaries="Ask before sending.",
    )

    with pytest.raises(experts_db.ExpertNotFoundError):
        await experts_db.update_soul(test_user.id, template.id, soul)
    with pytest.raises(experts_db.ExpertNotFoundError):
        await experts_db.update_soul(test_user.id, hired.expert.id, soul)


@pytest.mark.asyncio(loop_scope="session")
async def test_hire_copies_soul_fields_from_template(server: SpinTestServer, test_user):
    template = await prisma.models.Expert.prisma().create(
        data={
            "name": f"Otto {uuid.uuid4().hex[:8]}",
            "role": "Writer",
            "identity": "You are Otto, a playful writer.",
            "voicePreferences": "Direct, playful, and concise.",
            "boundaries": "Never publish without approval.",
            "isTemplate": True,
        }
    )

    hired = await experts_db.hire_expert(test_user.id, template.id, None)

    assert hired.expert.identity == "You are Otto, a playful writer."
    assert hired.expert.voice_preferences == "Direct, playful, and concise."
    assert hired.expert.boundaries == "Never publish without approval."


@pytest.mark.asyncio(loop_scope="session")
async def test_install_workflow_duplicate_returns_existing(
    server: SpinTestServer, test_user
):
    slv_id = await _seed_store_listing(server)
    template = await _seed_template(name="Maria", preload_listings=[])
    hired = await experts_db.hire_expert(test_user.id, template.id, None)
    a = await experts_db.install_workflow(test_user.id, hired.expert.id, slv_id)
    b = await experts_db.install_workflow(test_user.id, hired.expert.id, slv_id)
    assert a.id == b.id
    assert a.library_agent_id is not None
    assert a.store_listing_version_id == slv_id


@pytest.mark.asyncio(loop_scope="session")
async def test_resolve_expert_for_graph_unique_match(
    server: SpinTestServer, test_user, other_user
):
    """Manually scheduling an expert-installed workflow must keep the
    attribution: a unique (user, graph) → expert match resolves."""
    slv_id = await _seed_store_listing(server)
    template = await _seed_template(name="Maria", preload_listings=[slv_id])
    hired = await experts_db.hire_expert(test_user.id, template.id, None)
    graph_id = hired.expert.workflows[0].graph_id
    assert graph_id is not None

    resolved = await experts_db.resolve_expert_for_graph(test_user.id, graph_id)
    assert resolved == hired.expert.id
    assert await experts_db.resolve_expert_for_graph(other_user.id, graph_id) is None


@pytest.mark.asyncio(loop_scope="session")
async def test_resolve_expert_for_graph_ambiguous_returns_none(
    server: SpinTestServer, test_user
):
    """Two experts sharing one installed workflow (same LibraryAgent) make
    the join ambiguous — resolution must decline rather than guess."""
    slv_id = await _seed_store_listing(server)
    template_a = await _seed_template(name="Maria", preload_listings=[slv_id])
    template_b = await _seed_template(name="Max", preload_listings=[slv_id])
    hired_a = await experts_db.hire_expert(test_user.id, template_a.id, None)
    await experts_db.hire_expert(test_user.id, template_b.id, None)
    graph_id = hired_a.expert.workflows[0].graph_id
    assert graph_id is not None

    assert await experts_db.resolve_expert_for_graph(test_user.id, graph_id) is None


@pytest.mark.asyncio(loop_scope="session")
async def test_hire_creates_schedule_from_template_cadence(
    server: SpinTestServer, test_user
):
    """A roster preload with a cadence gets its schedule created at hire
    time, attributed to the expert, with the schedule id recorded."""
    slv_id = await _seed_store_listing(server)
    template = await _seed_template(
        name="Frankie",
        preload_listings=[slv_id],
        preload_crons={slv_id: "40 7 * * *"},
    )
    mock_scheduler = AsyncMock()
    mock_scheduler.add_execution_schedule = AsyncMock(
        return_value=SimpleNamespace(id="sched-1")
    )
    with patch.object(scheduling, "get_scheduler_client", return_value=mock_scheduler):
        result = await experts_db.hire_expert(test_user.id, template.id, None)

    wf = result.expert.workflows[0]
    assert wf.schedule_cron == "40 7 * * *"
    assert wf.schedule_id == "sched-1"
    assert result.failed_preloads == []
    call_kwargs = mock_scheduler.add_execution_schedule.call_args.kwargs
    assert call_kwargs["cron"] == "40 7 * * *"
    assert call_kwargs["expert_id"] == result.expert.id


@pytest.mark.asyncio(loop_scope="session")
async def test_hire_schedule_failure_marks_needs_setup(
    server: SpinTestServer, test_user
):
    """Schedule creation failing (e.g. missing credentials) must not sink
    the hire or the install: the cadence is kept, schedule_id stays None."""
    slv_id = await _seed_store_listing(server)
    template = await _seed_template(
        name="Frankie",
        preload_listings=[slv_id],
        preload_crons={slv_id: "40 7 * * *"},
    )
    mock_scheduler = AsyncMock()
    mock_scheduler.add_execution_schedule = AsyncMock(
        side_effect=RuntimeError("graph validation failed")
    )
    with patch.object(scheduling, "get_scheduler_client", return_value=mock_scheduler):
        result = await experts_db.hire_expert(test_user.id, template.id, None)

    wf = result.expert.workflows[0]
    assert wf.library_agent_id is not None
    assert wf.schedule_cron == "40 7 * * *"
    assert wf.schedule_id is None
    assert result.failed_preloads == []


@pytest.mark.asyncio(loop_scope="session")
async def test_archive_pauses_detaches_and_revive_reattaches(
    server: SpinTestServer, test_user
):
    """Archiving must leave no orphaned firing: presets deactivate, schedules
    delete, and the pause is logged. Re-hiring reverses all of it."""
    slv_id = await _seed_store_listing(server)
    template = await _seed_template(
        name="Frankie",
        preload_listings=[slv_id],
        preload_crons={slv_id: "40 7 * * *"},
    )
    sched = AsyncMock()
    sched.add_execution_schedule = AsyncMock(return_value=SimpleNamespace(id="sched-1"))
    sched.delete_schedule = AsyncMock()

    with patch.object(scheduling, "get_scheduler_client", return_value=sched):
        hired = await experts_db.hire_expert(test_user.id, template.id, None)
        expert_id = hired.expert.id
        wf = hired.expert.workflows[0]
        assert wf.library_agent_id is not None
        library_agent = await prisma.models.LibraryAgent.prisma().find_unique(
            where={"id": wf.library_agent_id}
        )
        assert library_agent is not None
        preset = await prisma.models.AgentPreset.prisma().create(
            data={
                "userId": test_user.id,
                "name": "Email trigger",
                "description": "",
                "agentGraphId": library_agent.agentGraphId,
                "agentGraphVersion": library_agent.agentGraphVersion,
                "expertId": expert_id,
            }
        )
        disabled_preset = await prisma.models.AgentPreset.prisma().create(
            data={
                "userId": test_user.id,
                "name": "Manually disabled trigger",
                "description": "",
                "agentGraphId": library_agent.agentGraphId,
                "agentGraphVersion": library_agent.agentGraphVersion,
                "expertId": expert_id,
                "isActive": False,
            }
        )
        sched.get_execution_schedules = AsyncMock(
            return_value=[
                SimpleNamespace(
                    kind="graph", id="sched-1", name="n", expert_id=expert_id
                )
            ]
        )
        await experts_db.archive_expert(test_user.id, expert_id)

    # Keyword args matter: passed positionally, user_id binds to graph_id
    # and the filter silently matches nothing.
    sched.get_execution_schedules.assert_awaited_with(
        user_id=test_user.id, kind="graph"
    )
    preset_row = await prisma.models.AgentPreset.prisma().find_unique(
        where={"id": preset.id}
    )
    assert preset_row is not None and preset_row.isActive is False
    assert preset_row.deactivatedByExpertArchive is True
    disabled_row = await prisma.models.AgentPreset.prisma().find_unique(
        where={"id": disabled_preset.id}
    )
    assert disabled_row is not None and disabled_row.isActive is False
    assert disabled_row.deactivatedByExpertArchive is False
    expert_row = await prisma.models.Expert.prisma().find_unique(
        where={"id": expert_id}
    )
    assert expert_row is not None and expert_row.schedulesPausedAt is not None
    events = await prisma.models.ExpertPauseEvent.prisma().find_many(
        where={"expertId": expert_id}
    )
    assert any(e.clearedAt is None for e in events)
    sched.delete_schedule.assert_awaited_once_with("sched-1", user_id=test_user.id)
    wf_row = await prisma.models.ExpertWorkflow.prisma().find_first(
        where={"expertId": expert_id}
    )
    assert wf_row is not None and wf_row.scheduleId is None

    with patch.object(scheduling, "get_scheduler_client", return_value=sched):
        revived = await experts_db.hire_expert(test_user.id, template.id, None)

    assert revived.expert.schedules_paused_at is None
    preset_row = await prisma.models.AgentPreset.prisma().find_unique(
        where={"id": preset.id}
    )
    assert preset_row is not None and preset_row.isActive is True
    assert preset_row.deactivatedByExpertArchive is False
    # The preset the user disabled before archiving must not come back on.
    disabled_row = await prisma.models.AgentPreset.prisma().find_unique(
        where={"id": disabled_preset.id}
    )
    assert disabled_row is not None and disabled_row.isActive is False
    wf_row = await prisma.models.ExpertWorkflow.prisma().find_first(
        where={"expertId": expert_id}
    )
    assert wf_row is not None and wf_row.scheduleId == "sched-1"


@pytest.mark.asyncio(loop_scope="session")
async def test_detach_keeps_pointer_when_schedule_delete_fails(
    server: SpinTestServer, test_user
):
    """A schedule that can't be deleted keeps its scheduleId pointer:
    wiping it would make re-hire create a second schedule while the
    orphaned original keeps firing."""
    slv_id = await _seed_store_listing(server)
    template = await _seed_template(
        name="Frankie",
        preload_listings=[slv_id],
        preload_crons={slv_id: "40 7 * * *"},
    )
    sched = AsyncMock()
    sched.add_execution_schedule = AsyncMock(return_value=SimpleNamespace(id="sched-1"))
    sched.delete_schedule = AsyncMock(side_effect=RuntimeError("scheduler down"))

    with patch.object(scheduling, "get_scheduler_client", return_value=sched):
        hired = await experts_db.hire_expert(test_user.id, template.id, None)
        expert_id = hired.expert.id
        sched.get_execution_schedules = AsyncMock(
            return_value=[
                SimpleNamespace(
                    kind="graph", id="sched-1", name="n", expert_id=expert_id
                )
            ]
        )
        await scheduling.detach_expert_triggers(test_user.id, expert_id)

    wf_row = await prisma.models.ExpertWorkflow.prisma().find_first(
        where={"expertId": expert_id}
    )
    assert wf_row is not None and wf_row.scheduleId == "sched-1"


@pytest.mark.asyncio(loop_scope="session")
async def test_enforce_budget_pauses_blocks_and_resumes(
    server: SpinTestServer, test_user
):
    template = await _seed_template(name="Max", preload_listings=[])
    hired = await experts_db.hire_expert(test_user.id, template.id, None)
    await prisma.models.Expert.prisma().update(
        where={"id": hired.expert.id}, data={"weeklyBudget": 100}
    )

    with patch.object(scheduling, "get_weekly_spend", new=AsyncMock(return_value=150)):
        with pytest.raises(ExpertRunPausedError):
            await scheduling.enforce_expert_run_budget(test_user.id, hired.expert.id)

    row = await prisma.models.Expert.prisma().find_unique(where={"id": hired.expert.id})
    assert row is not None and row.schedulesPausedAt is not None

    with pytest.raises(ExpertRunPausedError):
        await scheduling.enforce_expert_run_budget(test_user.id, hired.expert.id)

    # Resume must forgive the week's tracked spend, or the still-breached
    # counter re-pauses her on the very next fire and Resume is a no-op
    # until the ISO week rolls over.
    with patch.object(scheduling, "reset_weekly_spend", new=AsyncMock()) as reset_spend:
        assert await scheduling.resume_expert_schedules(test_user.id, hired.expert.id)
    reset_spend.assert_awaited_once_with(hired.expert.id)
    events = await prisma.models.ExpertPauseEvent.prisma().find_many(
        where={"expertId": hired.expert.id}
    )
    assert events and all(e.clearedAt is not None for e in events)
    with patch.object(scheduling, "get_weekly_spend", new=AsyncMock(return_value=10)):
        await scheduling.enforce_expert_run_budget(test_user.id, hired.expert.id)


@pytest.mark.asyncio(loop_scope="session")
async def test_seed_roster_round_trip(server: SpinTestServer):
    first_ids = await seed.seed_roster()
    assert len(first_ids) == 3

    templates = await experts_db.list_templates()
    seeded = {t.name: t for t in templates if t.id in first_ids}
    assert {e["name"] for e in seed.ROSTER} == set(seeded)
    for entry in seed.ROSTER:
        template = seeded[entry["name"]]
        assert template.is_template
        assert template.role == entry["role"]
        assert template.identity == entry["identity"]

    second_ids = await seed.seed_roster()
    assert first_ids == second_ids

    templates_after = await experts_db.list_templates()
    seeded_after = [t for t in templates_after if t.id in second_ids]
    assert len(seeded_after) == 3


@pytest.mark.asyncio(loop_scope="session")
async def test_list_experts_includes_last_run(server: SpinTestServer, test_user):
    """The /team card needs last-run status: the latest expert-attributed
    execution surfaces as last_run_at / last_run_status."""
    slv_id = await _seed_store_listing(server)
    template = await _seed_template(name="Maria", preload_listings=[slv_id])
    hired = await experts_db.hire_expert(test_user.id, template.id, None)
    wf = hired.expert.workflows[0]
    assert wf.library_agent_id is not None
    library_agent = await prisma.models.LibraryAgent.prisma().find_unique(
        where={"id": wf.library_agent_id}
    )
    assert library_agent is not None
    await prisma.models.AgentGraphExecution.prisma().create(
        data={
            "agentGraphId": library_agent.agentGraphId,
            "agentGraphVersion": library_agent.agentGraphVersion,
            "userId": test_user.id,
            "executionStatus": prisma.enums.AgentExecutionStatus.COMPLETED,
            "expertId": hired.expert.id,
        }
    )

    experts = await experts_db.list_experts(test_user.id)
    me = next(e for e in experts if e.id == hired.expert.id)
    assert me.last_run_at is not None
    assert me.last_run_status == "COMPLETED"

    fetched = await experts_db.get_expert(test_user.id, hired.expert.id)
    assert fetched is not None
    assert fetched.last_run_status == "COMPLETED"


@pytest.mark.asyncio(loop_scope="session")
async def test_sync_preloads_updates_template_cadence(server: SpinTestServer):
    """Re-seeding must propagate roster cadence changes onto existing
    template preload rows — the old sync was create-only."""
    slv_id = await _seed_store_listing(server)
    listing = await prisma.models.StoreListing.prisma().find_first(
        where={"activeVersionId": slv_id}
    )
    assert listing is not None
    template = await _seed_template(name="Frankie", preload_listings=[])
    entry: seed.RosterEntry = {
        "name": template.name,
        "role": template.role,
        "tagline": "",
        "avatar_url": None,
        "bio": "",
        "skills": [],
        "identity": template.identity,
        "preloads": [{"slug": listing.slug, "cron": "40 7 * * *"}],
    }

    await seed._sync_preloads(template.id, entry)
    row = await prisma.models.ExpertWorkflow.prisma().find_first(
        where={"expertId": template.id, "storeListingVersionId": slv_id}
    )
    assert row is not None
    assert row.scheduleCron == "40 7 * * *"

    entry["preloads"] = [{"slug": listing.slug, "cron": "0 8 * * *"}]
    await seed._sync_preloads(template.id, entry)
    row = await prisma.models.ExpertWorkflow.prisma().find_first(
        where={"expertId": template.id, "storeListingVersionId": slv_id}
    )
    assert row is not None
    assert row.scheduleCron == "0 8 * * *"


@pytest.mark.asyncio(loop_scope="session")
async def test_seed_backfills_presentation_fields_onto_hired_copies(
    server: SpinTestServer, test_user
):
    """A re-seed must reach experts hired before the roster changed."""
    # Unique name so _upsert_template resolves to this template and not to a
    # same-named one left behind by another test in the shared session DB.
    template = await _seed_template(
        name=f"Maria {uuid.uuid4().hex[:8]}", preload_listings=[]
    )
    hired = await experts_db.hire_expert(test_user.id, template.id, "My Maria")
    assert hired.expert.avatar_url is None
    assert hired.expert.bio is None
    assert hired.expert.skills == []

    entry: seed.RosterEntry = {
        "name": template.name,
        "role": template.role,
        "tagline": "Refreshed tagline",
        "avatar_url": "/experts/maria.svg",
        "bio": "Maria is a senior marketing strategist.",
        "skills": ["Content strategy", "SEO writing"],
        "identity": template.identity,
        "voice_preferences": "Clear and confident.",
        "boundaries": "Never invent customer evidence.",
        "preloads": [],
    }
    refreshed_template = await seed._upsert_template(entry)
    assert await seed._backfill_hired_copies(refreshed_template) == 1

    refreshed = await experts_db.get_expert(test_user.id, hired.expert.id)
    assert refreshed is not None
    assert refreshed.avatar_url == "/experts/maria.svg"
    assert refreshed.bio == "Maria is a senior marketing strategist."
    assert refreshed.skills == ["Content strategy", "SEO writing"]
    # A user's rename of their own hire survives the refresh.
    assert refreshed.name == "My Maria"
