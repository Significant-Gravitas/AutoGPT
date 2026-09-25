"""What a held call's ids point at: ``folder_id="f-111"`` becomes “Q3 reports”,
linked to its page, resolved once when the call is held and frozen into the card."""

import asyncio
import logging
from datetime import datetime
from typing import Any, Awaitable, Callable, Literal
from urllib.parse import quote, urlencode

from pydantic import BaseModel

from backend.api.features.experts.models import Expert
from backend.api.features.library.model import LibraryAgent, LibraryFolder
from backend.copilot.context import get_workspace_manager
from backend.copilot.model import ChatSession, get_chat_session_metadata
from backend.data.db_accessors import experts_db, library_db, store_db
from backend.data.redis_client import get_redis_async
from backend.executor.scheduler import GraphExecutionJobInfo
from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.util.clients import get_scheduler_client

logger = logging.getLogger(__name__)

Entity = Literal[
    "library_folder",
    "library_agent",
    "graph",
    "agent_or_graph",
    "preset",
    "schedule",
    "chat_session",
    "expert",
    "expert_or_name",
    "expert_template",
    "expert_workflow",
    "credential",
    "routine",
    "store_listing",
    "workspace_file",
    "team_change",
    "soul_change",
]

# (tool, argument) -> what its id names. None is a decision: not a platform id.
REFERENCES: dict[tuple[str, str], Entity | None] = {
    ("create_agent", "folder_id"): "library_folder",
    ("create_agent", "library_agent_ids"): "library_agent",
    ("customize_agent", "folder_id"): "library_folder",
    ("customize_agent", "library_agent_ids"): "library_agent",
    ("edit_agent", "agent_id"): "agent_or_graph",
    ("edit_agent", "library_agent_ids"): "library_agent",
    ("create_folder", "parent_id"): "library_folder",
    ("create_folder", "icon"): None,
    ("update_folder", "folder_id"): "library_folder",
    ("update_folder", "icon"): None,
    ("move_folder", "folder_id"): "library_folder",
    ("move_folder", "target_parent_id"): "library_folder",
    ("delete_folder", "folder_id"): "library_folder",
    ("move_agents_to_folder", "folder_id"): "library_folder",
    ("move_agents_to_folder", "agent_ids"): "library_agent",
    ("update_preset", "preset_id"): "preset",
    ("delete_preset", "preset_id"): "preset",
    ("pause_schedule", "schedule_id"): "schedule",
    ("resume_schedule", "schedule_id"): "schedule",
    ("delete_schedule", "schedule_id"): "schedule",
    ("schedule_followup", "session_id"): "chat_session",
    ("schedule_routine", "routine_id"): "routine",
    ("schedule_routine", "session_id"): "chat_session",
    ("schedule_routine", "expert_id"): "expert",
    ("setup_agent_webhook_trigger", "library_agent_id"): "library_agent",
    ("setup_agent_webhook_trigger", "graph_id"): "graph",
    ("hire_expert", "template_id"): "expert_template",
    ("update_expert", "expert_id"): "expert",
    ("confirm_expert_change", "confirmation_id"): "team_change",
    ("confirm_expert_soul_update", "confirmation_id"): "soul_change",
    ("install_expert_workflow", "library_agent_id"): "library_agent",
    ("install_expert_workflow", "store_listing_version_id"): "store_listing",
    ("install_expert_workflow", "expert_id"): "expert",
    ("remove_expert_workflow", "workflow_id"): "expert_workflow",
    ("remove_expert_workflow", "library_agent_id"): "library_agent",
    ("remove_expert_workflow", "expert_id"): "expert",
    ("grant_expert_credential", "credential_id"): "credential",
    ("grant_expert_credential", "expert_id"): "expert",
    ("revoke_expert_credential", "credential_id"): "credential",
    ("revoke_expert_credential", "expert_id"): "expert",
    ("delegate_to_expert", "expert_id"): "expert_or_name",
    ("delegate_to_expert", "delegated_session_id"): "chat_session",
    ("handoff_to_expert", "expert_id"): "expert",
    ("message_session", "session_id"): "chat_session",
    ("run_sub_session", "sub_autopilot_session_id"): "chat_session",
    ("delete_skill", "expert_id"): "expert",
    ("delete_workspace_file", "file_id"): "workspace_file",
    # Another system's ids, or not ids at all though described as one.
    ("create_feature_request", "existing_issue_id"): None,
    ("edit_chat_platform_message", "channel_id"): None,
    ("edit_chat_platform_message", "ref_id"): None,
    ("memory_forget_confirm", "uuids"): None,
    ("browser_act", "target"): None,
    ("post_to_chat_platform", "channel"): None,
}

LOOKUP_SECONDS = 1.0
CARD_SECONDS = 2.0
# Ids of a list resolved for the card; the rest read "+N more".
MAX_LISTED = 5
_MAX_SUMMARY_CHARS = 140
_MAX_DESCRIPTION_CHARS = 240


class Reference(BaseModel):
    key: str
    entity: Entity
    id: str
    # None: unresolved, and the card shows the raw id. href is None then too.
    name: str | None = None
    href: str | None = None
    # The hover card: what family the thing is, its own prose, short facts.
    kind: str | None = None
    description: str | None = None
    meta: list[str] = []
    # The card's facts as one line, for surfaces with room for no more.
    summary: str | None = None


async def resolve_references(
    tool_name: str, args: dict[str, Any], user_id: str, session: ChatSession
) -> list[Reference]:
    """Every id the call carries, named where its owner can see it; never raises."""
    wanted = wanted_references(tool_name, args)
    if not wanted:
        return []
    call = _Call(user_id=user_id, session=session, args=args)
    try:
        return list(
            await asyncio.wait_for(
                asyncio.gather(*(_resolve(ref, call) for ref in wanted)),
                CARD_SECONDS,
            )
        )
    except asyncio.TimeoutError:
        logger.warning(f"Gate card references for {tool_name} timed out")
        return wanted


def wanted_references(tool_name: str, args: dict[str, Any]) -> list[Reference]:
    refs: list[Reference] = []
    for (tool, key), entity in REFERENCES.items():
        if tool != tool_name or entity is None:
            continue
        value = args.get(key)
        refs.extend(
            Reference(key=key, entity=entity, id=id)
            for id in listed_ids(value)[:MAX_LISTED]
        )
    return refs


def listed_ids(value: Any) -> list[str]:
    """The ids an argument holds, blanks and non-strings dropped."""
    values = value if isinstance(value, list) else [value]
    return [v.strip() for v in values if isinstance(v, str) and v.strip()]


class _Call(BaseModel):
    user_id: str
    session: ChatSession
    args: dict[str, Any]


class _Found(BaseModel):
    kind: str
    name: str
    href: str | None = None
    description: str | None = None
    meta: list[str | None] = []


async def _resolve(ref: Reference, call: _Call) -> Reference:
    try:
        found = await asyncio.wait_for(
            _RESOLVERS[ref.entity](ref.id, call), LOOKUP_SECONDS
        )
    except asyncio.TimeoutError:
        logger.warning(f"Gate reference {ref.entity} lookup timed out")
        return ref
    except Exception as e:
        # Not found or not owned raise here as often as a real failure does.
        logger.debug("Gate reference %s unresolved: %s", ref.entity, e)
        return ref
    if found is None or not found.name.strip():
        return ref
    meta = [line for m in found.meta if (line := _one_line(m))]
    description = _clip(found.description, _MAX_DESCRIPTION_CHARS)
    return ref.model_copy(
        update={
            "name": " ".join(found.name.split()),
            "href": found.href,
            "kind": found.kind,
            "description": description,
            "meta": meta,
            "summary": _one_line(" · ".join(meta)) or _one_line(description),
        }
    )


async def _library_folder(folder_id: str, call: _Call) -> _Found | None:
    folder = await library_db().get_folder(folder_id, call.user_id)
    parent = await _parent_folder(folder.parent_id, call) if folder.parent_id else None
    return _Found(
        kind="Library folder",
        name=folder.name,
        href=f"/library?{urlencode({'folder': folder.id})}",
        meta=[
            _count(folder.agent_count, "agent"),
            _count(folder.subfolder_count, "subfolder"),
            f"In {parent.name}" if parent else None,
        ],
    )


async def _library_agent(agent_id: str, call: _Call) -> _Found | None:
    agent = await library_db().get_library_agent(agent_id, call.user_id)
    return _agent(agent)


async def _graph(graph_id: str, call: _Call) -> _Found | None:
    agent = await library_db().get_library_agent_by_graph_id(call.user_id, graph_id)
    return _agent(agent) if agent else None


async def _agent_or_graph(agent_id: str, call: _Call) -> _Found | None:
    return await _graph(agent_id, call) or await _library_agent(agent_id, call)


async def _preset(preset_id: str, call: _Call) -> _Found | None:
    preset = await library_db().get_preset(call.user_id, preset_id)
    if preset is None:
        return None
    agent = await library_db().get_library_agent_by_graph_id(
        call.user_id, preset.graph_id
    )
    # A preset with a webhook lists under Triggers, the rest under Templates.
    kind, tab, item = (
        ("Trigger", "triggers", f"preset:{preset_id}")
        if preset.webhook_id
        else ("Template", "templates", preset_id)
    )
    return _Found(
        kind=kind,
        name=preset.name,
        href=_agent_href(agent.id, activeTab=tab, activeItem=item) if agent else None,
        description=preset.description,
        meta=[
            f"Agent: {agent.name}" if agent else None,
            None if preset.is_active else "Inactive",
        ],
    )


async def _schedule(schedule_id: str, call: _Call) -> _Found | None:
    # No by-id read exists; include_paused, as the tools list them.
    jobs = await get_scheduler_client().get_execution_schedules(
        user_id=call.user_id, include_paused=True
    )
    job = next((j for j in jobs if j.id == schedule_id), None)
    if job is None:
        return None
    # include_paused lists schedules with no next run.
    when = job.next_run_time[:16].replace("T", " ")
    href, agent = "/library/followups", None
    if isinstance(job, GraphExecutionJobInfo):
        agent = await library_db().get_library_agent_by_graph_id(
            call.user_id, job.graph_id
        )
        if agent is not None:
            href = _agent_href(agent.id, activeTab="scheduled", activeItem=schedule_id)
    return _Found(
        kind="Schedule",
        name=job.name,
        href=href,
        meta=[
            f"Runs {job.cron or 'once'}",
            f"Next {when}" if when else "Paused",
            f"Agent: {agent.name}" if agent else None,
        ],
    )


async def _chat_session(session_id: str, call: _Call) -> _Found | None:
    meta = await get_chat_session_metadata(session_id, call.user_id)
    if meta is None:
        return None
    return _Found(
        kind="Chat",
        name=meta.title or "Untitled chat",
        href=f"/copilot?{urlencode({'sessionId': session_id})}",
        meta=[
            f"Started {meta.started_at:%Y-%m-%d}",
            f"Last active {meta.updated_at:%Y-%m-%d}",
        ],
    )


async def _expert(expert_id: str, call: _Call) -> _Found | None:
    expert = await experts_db().get_expert(
        call.user_id, expert_id, include_workflows=False
    )
    return _teammate(expert) if expert else None


async def _expert_or_name(reference: str, call: _Call) -> _Found | None:
    # Lazy: the tools package imports the gate.
    from backend.copilot.tools.expert_delegation import resolve_target_expert

    expert = await resolve_target_expert(call.user_id, reference)
    return _teammate(expert) if expert else None


async def _expert_template(template_id: str, call: _Call) -> _Found | None:
    templates = await experts_db().list_templates()
    template = next((t for t in templates if t.id == template_id), None)
    if template is None:
        return None
    return _Found(
        kind="Expert template",
        name=template.name,
        href=f"/marketplace/experts/{quote(template.id, safe='')}",
        description=template.tagline or template.bio,
        meta=[template.job_title or template.role],
    )


async def _expert_workflow(workflow_id: str, call: _Call) -> _Found | None:
    label = await experts_db().get_workflow_label(call.user_id, workflow_id)
    if label is None or label.name is None:
        return None
    return _Found(
        kind="Expert workflow", name=label.name, href=_team_href(label.expert_id)
    )


async def _credential(credential_id: str, call: _Call) -> _Found | None:
    creds = await IntegrationCredentialsManager().store.get_creds_by_id(
        call.user_id, credential_id
    )
    if creds is None:
        return None
    return _Found(
        kind="Credential",
        name=creds.title or str(creds.provider),
        href="/settings/integrations",
        meta=[str(creds.provider), creds.type.replace("_", " ")],
    )


async def _routine(routine_id: str, call: _Call) -> _Found | None:
    # get_routine is unscoped, so the owner's own listing is the lookup.
    expert_id = call.args.get("expert_id") or call.session.expert_id
    routines = await experts_db().list_routines(call.user_id, expert_id)
    routine = next((r for r in routines if r.id == routine_id), None)
    if routine is None:
        return None
    return _Found(
        kind="Routine",
        name=routine.title,
        href=_team_href(routine.expert_id) if routine.expert_id else None,
        description=routine.prompt,
        meta=[
            _cadence(routine.crons, routine.run_at),
            None if routine.enabled else "Paused",
        ],
    )


async def _store_listing(version_id: str, call: _Call) -> _Found | None:
    agent = await store_db().get_store_agent_by_version_id(version_id)
    return _Found(
        kind="Marketplace agent",
        name=agent.agent_name,
        href=f"/marketplace/agent/{quote(agent.creator, safe='')}/"
        f"{quote(agent.slug, safe='')}",
        description=agent.sub_heading or agent.description,
        meta=[
            f"By {agent.creator}",
            _count(agent.runs, "run"),
            f"{agent.rating:.1f} ★" if agent.rating else None,
        ],
    )


async def _workspace_file(file_id: str, call: _Call) -> _Found | None:
    manager = await get_workspace_manager(call.user_id, call.session.session_id)
    file = await manager.get_file_info(file_id)
    if file is None:
        return None
    query = f"?{urlencode({'folder': file.folder_id})}" if file.folder_id else ""
    folder = file.path.rsplit("/", 1)[0]
    return _Found(
        kind="Workspace file",
        name=file.name,
        href=f"/artifacts{query}",
        meta=[
            file.mime_type,
            _size(file.size_bytes),
            f"Modified {file.updated_at:%Y-%m-%d}",
            f"In {folder}" if folder else None,
        ],
    )


async def _team_change(confirmation_id: str, call: _Call) -> _Found | None:
    # Read without load_bound_proposal, which deletes a proposal it rejects.
    from backend.copilot.tools.expert_proposal import ExpertChangeProposal, proposal_key

    raw = await (await get_redis_async()).get(proposal_key(confirmation_id))
    if raw is None:
        return None
    proposal = ExpertChangeProposal.model_validate_json(raw)
    if not _bound(proposal.user_id, proposal.session_id, call):
        return None
    preview = proposal.preview
    return _Found(
        kind=_TEAM_CHANGE_KINDS[preview.kind],
        name=preview.name,
        href=_team_href(proposal.expert_id) if proposal.expert_id else None,
        description=preview.tagline or preview.about,
        meta=[preview.job_title or preview.role],
    )


async def _soul_change(confirmation_id: str, call: _Call) -> _Found | None:
    from backend.copilot.tools.soul_proposal import SoulEditProposal, proposal_key

    raw = await (await get_redis_async()).get(proposal_key(confirmation_id))
    if raw is None:
        return None
    proposal = SoulEditProposal.model_validate_json(raw)
    if not _bound(proposal.user_id, proposal.session_id, call):
        return None
    found = await _expert(proposal.expert_id, call)
    return found.model_copy(update={"kind": "Expert update"}) if found else None


_TEAM_CHANGE_KINDS = {
    "hire": "New hire",
    "raise": "New expert",
    "update": "Expert update",
}

_RESOLVERS: dict[Entity, Callable[[str, _Call], Awaitable[_Found | None]]] = {
    "library_folder": _library_folder,
    "library_agent": _library_agent,
    "graph": _graph,
    "agent_or_graph": _agent_or_graph,
    "preset": _preset,
    "schedule": _schedule,
    "chat_session": _chat_session,
    "expert": _expert,
    "expert_or_name": _expert_or_name,
    "expert_template": _expert_template,
    "expert_workflow": _expert_workflow,
    "credential": _credential,
    "routine": _routine,
    "store_listing": _store_listing,
    "workspace_file": _workspace_file,
    "team_change": _team_change,
    "soul_change": _soul_change,
}


def _bound(user_id: str, session_id: str, call: _Call) -> bool:
    return user_id == call.user_id and session_id == call.session.session_id


def _agent_href(library_agent_id: str, **params: str) -> str:
    href = f"/library/agents/{quote(library_agent_id, safe='')}"
    return f"{href}?{urlencode(params)}" if params else href


def _team_href(expert_id: str) -> str:
    return f"/team/{quote(expert_id, safe='')}"


async def _parent_folder(parent_id: str, call: _Call) -> LibraryFolder | None:
    try:
        return await library_db().get_folder(parent_id, call.user_id)
    except Exception:
        # The folder itself resolved; a missing parent only drops the line.
        return None


def _agent(agent: LibraryAgent) -> _Found:
    return _Found(
        kind="Agent",
        name=agent.name,
        href=_agent_href(agent.id),
        description=agent.description,
        meta=[
            f"Version {agent.graph_version}",
            f"In {agent.folder_name}" if agent.folder_name else None,
            (
                f"Last run {agent.last_run_at:%Y-%m-%d}"
                if agent.last_run_at
                else "Never run"
            ),
        ],
    )


def _teammate(expert: Expert) -> _Found:
    return _Found(
        kind="Expert",
        name=expert.name,
        href=_team_href(expert.id),
        description=expert.tagline or expert.bio,
        meta=[
            expert.job_title or expert.role,
            "Archived" if expert.is_archived else None,
        ],
    )


def _cadence(crons: list[str], run_at: datetime | None) -> str | None:
    if crons:
        return "Runs " + ", ".join(crons)
    return f"Once at {run_at:%Y-%m-%d %H:%M}" if run_at else None


def _one_line(text: str | None) -> str | None:
    return _clip(text, _MAX_SUMMARY_CHARS)


def _clip(text: str | None, limit: int) -> str | None:
    line = " ".join((text or "").split())
    if len(line) > limit:
        return line[: limit - 1] + "…"
    return line or None


def _count(n: int, noun: str) -> str:
    return f"{n:,} {noun}{'' if n == 1 else 's'}"


def _size(size_bytes: int) -> str:
    size = float(size_bytes)
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.0f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"
