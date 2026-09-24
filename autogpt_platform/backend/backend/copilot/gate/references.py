"""What a held call's ids point at: ``folder_id="f-111"`` becomes “Q3 reports”,
linked to its page, resolved once when the call is held and frozen into the card."""

import asyncio
import logging
from typing import Any, Awaitable, Callable, Literal
from urllib.parse import quote, urlencode

from pydantic import BaseModel

from backend.api.features.experts.models import Expert
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


class Reference(BaseModel):
    key: str
    entity: Entity
    id: str
    # None: unresolved, and the card shows the raw id. href is None then too.
    name: str | None = None
    href: str | None = None
    # One line for the link's hover, e.g. an agent's description.
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
    name: str
    href: str | None = None
    summary: str | None = None


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
    return ref.model_copy(
        update={
            "name": " ".join(found.name.split()),
            "href": found.href,
            "summary": _one_line(found.summary),
        }
    )


async def _library_folder(folder_id: str, call: _Call) -> _Found | None:
    folder = await library_db().get_folder(folder_id, call.user_id)
    return _Found(
        name=folder.name,
        href=f"/library?{urlencode({'folder': folder.id})}",
        summary=f"{_count(folder.agent_count, 'agent')} · "
        f"{_count(folder.subfolder_count, 'folder')}",
    )


async def _library_agent(agent_id: str, call: _Call) -> _Found | None:
    agent = await library_db().get_library_agent(agent_id, call.user_id)
    return _Found(
        name=agent.name, href=_agent_href(agent.id), summary=agent.description
    )


async def _graph(graph_id: str, call: _Call) -> _Found | None:
    agent = await library_db().get_library_agent_by_graph_id(call.user_id, graph_id)
    if agent is None:
        return None
    return _Found(
        name=agent.name, href=_agent_href(agent.id), summary=agent.description
    )


async def _agent_or_graph(agent_id: str, call: _Call) -> _Found | None:
    return await _graph(agent_id, call) or await _library_agent(agent_id, call)


async def _preset(preset_id: str, call: _Call) -> _Found | None:
    preset = await library_db().get_preset(call.user_id, preset_id)
    if preset is None:
        return None
    agent = await library_db().get_library_agent_by_graph_id(
        call.user_id, preset.graph_id
    )
    if agent is None:
        return _Found(name=preset.name, summary=preset.description)
    # A preset with a webhook lists under Triggers, the rest under Templates.
    tab, item = (
        ("triggers", f"preset:{preset_id}")
        if preset.webhook_id
        else ("templates", preset_id)
    )
    return _Found(
        name=preset.name,
        href=_agent_href(agent.id, activeTab=tab, activeItem=item),
        summary=preset.description,
    )


async def _schedule(schedule_id: str, call: _Call) -> _Found | None:
    # No by-id read exists; include_paused, as the tools list them.
    jobs = await get_scheduler_client().get_execution_schedules(
        user_id=call.user_id, include_paused=True
    )
    job = next((j for j in jobs if j.id == schedule_id), None)
    if job is None:
        return None
    cadence = job.cron or "once"
    summary = f"Runs {cadence} · next {job.next_run_time[:16].replace('T', ' ')}"
    href = "/library/followups"
    if isinstance(job, GraphExecutionJobInfo):
        agent = await library_db().get_library_agent_by_graph_id(
            call.user_id, job.graph_id
        )
        if agent is not None:
            href = _agent_href(agent.id, activeTab="scheduled", activeItem=schedule_id)
    return _Found(name=job.name, href=href, summary=summary)


async def _chat_session(session_id: str, call: _Call) -> _Found | None:
    meta = await get_chat_session_metadata(session_id, call.user_id)
    if meta is None:
        return None
    return _Found(
        name=meta.title or "Untitled chat",
        href=f"/copilot?{urlencode({'sessionId': session_id})}",
        summary=f"Last active {meta.updated_at:%Y-%m-%d}",
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
        name=template.name,
        href=f"/marketplace/experts/{quote(template.id, safe='')}",
        summary=template.tagline or template.job_title or template.role,
    )


async def _expert_workflow(workflow_id: str, call: _Call) -> _Found | None:
    label = await experts_db().get_workflow_label(call.user_id, workflow_id)
    if label is None or label.name is None:
        return None
    return _Found(name=label.name, href=_team_href(label.expert_id))


async def _credential(credential_id: str, call: _Call) -> _Found | None:
    creds = await IntegrationCredentialsManager().store.get_creds_by_id(
        call.user_id, credential_id
    )
    if creds is None:
        return None
    kind = creds.type.replace("_", " ")
    return _Found(
        name=creds.title or str(creds.provider),
        href="/settings/integrations",
        summary=f"{creds.provider} · {kind}",
    )


async def _routine(routine_id: str, call: _Call) -> _Found | None:
    # get_routine is unscoped, so the owner's own listing is the lookup.
    expert_id = call.args.get("expert_id") or call.session.expert_id
    routines = await experts_db().list_routines(call.user_id, expert_id)
    routine = next((r for r in routines if r.id == routine_id), None)
    if routine is None:
        return None
    return _Found(
        name=routine.title,
        href=_team_href(routine.expert_id) if routine.expert_id else None,
        summary=routine.prompt,
    )


async def _store_listing(version_id: str, call: _Call) -> _Found | None:
    agent = await store_db().get_store_agent_by_version_id(version_id)
    return _Found(
        name=agent.agent_name,
        href=f"/marketplace/agent/{quote(agent.creator, safe='')}/"
        f"{quote(agent.slug, safe='')}",
        summary=f"By {agent.creator} · {agent.description}",
    )


async def _workspace_file(file_id: str, call: _Call) -> _Found | None:
    manager = await get_workspace_manager(call.user_id, call.session.session_id)
    file = await manager.get_file_info(file_id)
    if file is None:
        return None
    query = f"?{urlencode({'folder': file.folder_id})}" if file.folder_id else ""
    return _Found(
        name=file.name,
        href=f"/artifacts{query}",
        summary=f"{file.mime_type} · {_size(file.size_bytes)}",
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
        name=preview.name,
        href=_team_href(proposal.expert_id) if proposal.expert_id else None,
        summary=preview.tagline or preview.job_title or preview.role,
    )


async def _soul_change(confirmation_id: str, call: _Call) -> _Found | None:
    from backend.copilot.tools.soul_proposal import SoulEditProposal, proposal_key

    raw = await (await get_redis_async()).get(proposal_key(confirmation_id))
    if raw is None:
        return None
    proposal = SoulEditProposal.model_validate_json(raw)
    if not _bound(proposal.user_id, proposal.session_id, call):
        return None
    return await _expert(proposal.expert_id, call)


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


def _teammate(expert: Expert) -> _Found:
    return _Found(
        name=expert.name,
        href=_team_href(expert.id),
        summary=expert.tagline or expert.job_title or expert.role,
    )


def _one_line(text: str | None) -> str | None:
    line = " ".join((text or "").split())
    if len(line) > _MAX_SUMMARY_CHARS:
        return line[: _MAX_SUMMARY_CHARS - 1] + "…"
    return line or None


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


def _size(size_bytes: int) -> str:
    size = float(size_bytes)
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.0f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"
