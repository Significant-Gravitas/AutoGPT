from unittest.mock import patch

from backend.data.execution import ExecutionContext
from backend.util.sandbox_metadata import SandboxMetadata, deployment_env

_USER = "user-1111"
_SESSION = "sess-2222"
_EXPERT = "expert-3333"
_GRAPH = "graph-4444"
_GRAPH_EXEC = "gexec-5555"
_NODE_EXEC = "nexec-6666"
_BLOCK = "block-7777"


def _context(**overrides) -> ExecutionContext:
    fields = dict(
        user_id=_USER,
        session_id=_SESSION,
        expert_id=_EXPERT,
        graph_id=_GRAPH,
        graph_exec_id=_GRAPH_EXEC,
        node_exec_id=_NODE_EXEC,
    )
    fields.update(overrides)
    return ExecutionContext(**fields)


class TestAsE2B:
    def test_prefixes_keys_and_drops_unset_fields(self):
        meta = SandboxMetadata(
            owner="session:s", kind="shell", source="copilot", env="x"
        )
        assert meta.as_e2b() == {
            "autogpt_owner": "session:s",
            "autogpt_kind": "shell",
            "autogpt_source": "copilot",
            "autogpt_env": "x",
        }

    def test_values_are_all_strings(self):
        meta = SandboxMetadata.for_block(_context(), "code", _BLOCK, "tpl")
        assert all(isinstance(v, str) for v in meta.as_e2b().values())


class TestForCopilot:
    def test_stamps_env_from_settings(self):
        with patch("backend.util.sandbox_metadata.deployment_env", return_value="prod"):
            meta = SandboxMetadata.for_copilot("session:s", "shell")
        assert meta.as_e2b()["autogpt_env"] == "prod"

    def test_carries_provenance(self):
        meta = SandboxMetadata.for_copilot(
            f"expert:{_EXPERT}",
            "desktop",
            user_id=_USER,
            session_id=_SESSION,
            expert_id=_EXPERT,
            template="desktop",
            mounts="attached",
        )
        assert meta.as_e2b() == {
            "autogpt_owner": f"expert:{_EXPERT}",
            "autogpt_kind": "desktop",
            "autogpt_source": "copilot",
            "autogpt_env": deployment_env(),
            "autogpt_user": _USER,
            "autogpt_session": _SESSION,
            "autogpt_expert": _EXPERT,
            "autogpt_template": "desktop",
            "autogpt_mounts": "attached",
        }


class TestForBlock:
    def test_traces_to_the_node_execution(self):
        meta = SandboxMetadata.for_block(_context(), "code", _BLOCK, "tpl-1")
        assert meta.as_e2b() == {
            "autogpt_owner": f"graph_exec:{_GRAPH_EXEC}",
            "autogpt_kind": "code",
            "autogpt_source": "block",
            "autogpt_env": deployment_env(),
            "autogpt_user": _USER,
            "autogpt_session": _SESSION,
            "autogpt_expert": _EXPERT,
            "autogpt_graph": _GRAPH,
            "autogpt_graph_exec": _GRAPH_EXEC,
            "autogpt_node_exec": _NODE_EXEC,
            "autogpt_block": _BLOCK,
            "autogpt_template": "tpl-1",
        }

    def test_owner_falls_back_to_user_then_block(self):
        by_user = SandboxMetadata.for_block(
            _context(graph_exec_id=None), "desktop", _BLOCK
        )
        assert by_user.owner == f"user:{_USER}"
        by_block = SandboxMetadata.for_block(
            _context(graph_exec_id=None, user_id=None), "desktop", _BLOCK
        )
        assert by_block.owner == f"block:{_BLOCK}"

    def test_empty_template_is_omitted(self):
        meta = SandboxMetadata.for_block(_context(), "code", _BLOCK, "")
        assert "autogpt_template" not in meta.as_e2b()

    def test_without_context_still_identifies_the_block(self):
        meta = SandboxMetadata.for_block(None, "claude_code", _BLOCK, "base")
        assert meta.as_e2b() == {
            "autogpt_owner": f"block:{_BLOCK}",
            "autogpt_kind": "claude_code",
            "autogpt_source": "block",
            "autogpt_env": deployment_env(),
            "autogpt_block": _BLOCK,
            "autogpt_template": "base",
        }
