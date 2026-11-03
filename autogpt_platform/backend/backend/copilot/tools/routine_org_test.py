from types import SimpleNamespace
from unittest.mock import AsyncMock

from backend.copilot import expert_context
from backend.copilot.model import ChatSession
from backend.copilot.tools import routines
from backend.copilot.tools.models import ErrorResponse


async def test_personal_list_uses_saved_session_org(monkeypatch):
    listing = AsyncMock(return_value=[])
    monkeypatch.setattr(
        routines, "experts_db", lambda: SimpleNamespace(list_routines=listing)
    )
    session = ChatSession.new("owner", dry_run=False, organization_id="business-a")
    await routines.ListRoutinesTool()._execute("owner", session)
    listing.assert_awaited_once_with("owner", None, organization_id="business-a")


async def test_personal_pin_rejects_other_org(monkeypatch):
    session = ChatSession.new("owner", dry_run=False, organization_id="business-a")
    target = ChatSession.new("owner", dry_run=False, organization_id="business-b")
    monkeypatch.setattr(routines, "get_chat_session", AsyncMock(return_value=target))
    owner = await routines.resolve_routine_owner("owner", session, None)
    assert not isinstance(owner, ErrorResponse)
    result = await routines.ScheduleRoutineTool()._pinned_session(
        "owner",
        session,
        owner,
        {"session_mode": "PINNED", "session_id": target.session_id},
    )
    assert isinstance(result, ErrorResponse)
    assert result.error == "session_not_found"


async def test_personal_prompt_uses_saved_org(monkeypatch):
    listing = AsyncMock(return_value=[])
    monkeypatch.setattr(
        expert_context, "experts_db", lambda: SimpleNamespace(list_routines=listing)
    )
    monkeypatch.setattr(expert_context, "_team_context", AsyncMock(return_value=""))
    await expert_context.build_expert_context(
        "owner", None, organization_id="business-a"
    )
    listing.assert_awaited_once_with("owner", None, organization_id="business-a")


async def test_saved_credentials_lookup_checks_owner_and_org(monkeypatch):
    read = AsyncMock(return_value=None)
    pin = AsyncMock(return_value={})
    monkeypatch.setattr(
        routines, "experts_db", lambda: SimpleNamespace(get_routine=read)
    )
    monkeypatch.setattr(routines, "has_account_choices", AsyncMock(return_value=True))
    monkeypatch.setattr(routines, "pin_schedule_credentials", pin)
    session = ChatSession.new("owner", dry_run=False, organization_id="business-a")
    owner = await routines.resolve_routine_owner("owner", session, None)
    assert not isinstance(owner, ErrorResponse)
    await routines.ScheduleRoutineTool()._credential_pins(
        "owner", session, owner, "routine-in-other-org", True, {}
    )
    read.assert_awaited_once_with(
        "routine-in-other-org",
        user_id="owner",
        expert_id=None,
        organization_id="business-a",
    )
    assert pin.await_args.args[-1] is None
