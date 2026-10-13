import pytest
from pydantic import ValidationError

from backend.copilot.tools._test_data import make_session
from backend.copilot.tools.models import ResponseType
from backend.copilot.tools.render_ui import RenderUIInput, RenderUITool


@pytest.mark.parametrize(
    "source",
    [
        'root = Workspace("Plan", "", [table])\n'
        'table = DataTable("Stops", ["Name"], [["First"], ["Second"])',
        'root = Workspace("Plan", "", [note])\n'
        'note = Insight("Next", "Check timing", "neutral"',
        'root = Workspace("Plan", "", [})',
        'root = Workspace("Unfinished',
    ],
)
def test_rejects_incomplete_programs_before_publishing(source: str):
    with pytest.raises(ValidationError):
        RenderUIInput(source=source, summary="A plan")


@pytest.mark.parametrize(
    "source",
    [
        'root = Workspace("Brackets [({ in a title", "Closing )]} too", [])',
        'root = Workspace("It is \\"quoted\\"", "\\\\", [])',
        "root = Workspace('Single quoted [ text', '', [])",
        'root = Workspace("https://example.com/#here", "", []) // unfinished [',
        'root = Workspace("Plan", "", [])\n# ignore this unmatched ] and quote "',
        'root = Workspace("Plan", "", [tasks])\n'
        'tasks = Checklist("Things", [{title: "Task", detail: "[not syntax]"}])',
    ],
)
def test_preserves_strings_comments_and_forward_references(source: str):
    assert RenderUIInput(source=source, summary="A plan").source == source


@pytest.mark.asyncio
async def test_returns_a_repairable_error_instead_of_a_broken_view():
    result = await RenderUITool()._execute(
        "owner",
        make_session("owner"),
        source='root = Workspace("Plan", "", [',
        summary="A plan",
    )
    assert result.type == ResponseType.ERROR
    assert "Unclosed" in result.message
    assert "complete" in result.message.lower()
