import json

from backend.util.tool_call_loop import LLMToolCall

from .tool_order import writes_files


def _call(name: str, args: object) -> LLMToolCall:
    return LLMToolCall(id="tc", name=name, arguments=json.dumps(args))


def test_file_writers_named_directly_run_alone():
    assert writes_files(_call("bash_exec", {"command": "zip -r out.zip ."}))
    assert writes_files(_call("write_workspace_file", {"source_path": "out.zip"}))
    assert writes_files(_call("read_workspace_file", {"path": "a.csv"}))


def test_tools_that_write_no_files_overlap():
    assert not writes_files(_call("web_search", {"query": "weather"}))
    assert not writes_files(_call("list_workspace_files", {}))


def test_a_writer_dispatched_through_run_capability_runs_alone():
    """``skill:<name>`` dispatches ``read_skill``, which copies the skill's
    package into the working directory."""
    assert writes_files(_call("run_capability", {"id": "skill:pdf"}))


def test_unparseable_arguments_overlap():
    call = LLMToolCall(id="tc", name="bash_exec", arguments="{not json")
    assert not writes_files(call)


def test_unknown_tool_overlaps():
    assert not writes_files(_call("no_such_tool", {}))
