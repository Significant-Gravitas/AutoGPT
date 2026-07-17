"""Unit tests for the public-share sanitizer.

The sanitizer is the security boundary for public chat sharing — every
test here is checking that data which must NOT cross the public link
genuinely does not.
"""

from datetime import datetime, timezone

from backend.copilot.model import ChatMessage as ChatMessageDomain
from backend.copilot.sharing.models import _redact_secret_keys, sanitize_chat_message


def _msg(**overrides) -> ChatMessageDomain:
    defaults = dict(
        id="m1",
        role="assistant",
        content=None,
        sequence=0,
        created_at=datetime(2026, 5, 11, tzinfo=timezone.utc),
    )
    defaults.update(overrides)
    return ChatMessageDomain(**defaults)


class TestRedactSecretKeys:
    def test_redacts_api_key_at_top_level(self):
        result = _redact_secret_keys({"api_key": "sk-xyz", "model": "claude"})
        assert result == {"api_key": "[redacted]", "model": "claude"}

    def test_redacts_nested_secrets_in_arguments(self):
        result = _redact_secret_keys(
            {
                "name": "run_agent",
                "arguments": {
                    "auth_token": "eyJ...",
                    "input": {"foo": "bar"},
                },
            }
        )
        assert result["arguments"]["auth_token"] == "[redacted]"
        assert result["arguments"]["input"] == {"foo": "bar"}

    def test_passes_through_non_secret_strings(self):
        result = _redact_secret_keys({"prompt": "hello"})
        assert result == {"prompt": "hello"}

    def test_matches_case_insensitively(self):
        result = _redact_secret_keys({"API_KEY": "x", "Cookie": "y"})
        assert result == {"API_KEY": "[redacted]", "Cookie": "[redacted]"}

    def test_redacts_password_variants(self):
        result = _redact_secret_keys({"password": "p", "passwd": "p2"})
        assert result == {"password": "[redacted]", "passwd": "[redacted]"}

    def test_walks_lists(self):
        result = _redact_secret_keys(
            [{"secret": "x"}, {"name": "y"}],
        )
        assert result == [{"secret": "[redacted]"}, {"name": "y"}]

    def test_redacts_secret_shaped_key_regardless_of_value_type(self):
        # Secret-shaped keys redact their ENTIRE subtree — strings,
        # ints, bools, lists, dicts.  Trades the (rare) ability to
        # surface bool/int metadata under a secret-shaped key for
        # sealing the leak vectors below.
        assert _redact_secret_keys({"token": 42}) == {"token": "[redacted]"}
        assert _redact_secret_keys({"requires_auth": True}) == {
            "requires_auth": "[redacted]",
        }

    def test_redacts_list_under_secret_shaped_key(self):
        # Regression: pre-fix this leaked because the list is not a
        # string and the inner items had no secret-shaped key wrapping.
        result = _redact_secret_keys({"api_keys": ["sk-prod", "sk-test"]})
        assert result == {"api_keys": "[redacted]"}

    def test_redacts_nested_dict_under_secret_shaped_key(self):
        # Regression: pre-fix this leaked because the dict value
        # recursed and the inner ``bearer`` key didn't match any hint.
        result = _redact_secret_keys({"auth": {"bearer": "tok-xyz"}})
        assert result == {"auth": "[redacted]"}

    def test_redacts_list_of_dicts_under_secret_shaped_key(self):
        # Regression: pre-fix this leaked end-to-end.
        result = _redact_secret_keys(
            {"oauth_tokens": [{"value": "tok-1"}, {"value": "tok-2"}]}
        )
        assert result == {"oauth_tokens": "[redacted]"}

    def test_does_not_mutate_input(self):
        original = {"api_key": "sk-1"}
        _redact_secret_keys(original)
        assert original == {"api_key": "sk-1"}

    def test_does_not_redact_llm_usage_token_counts(self):
        # Cursor regression: substring matching mis-flagged
        # ``prompt_tokens`` / ``completion_tokens`` / ``total_tokens`` as
        # secret-shaped because they contain ``token``.  These are
        # legitimate usage-metadata fields and must survive sanitization.
        result = _redact_secret_keys(
            {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "input_tokens": 80,
            }
        )
        assert result == {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "input_tokens": 80,
        }

    def test_does_not_redact_author_field(self):
        # Cursor regression: ``auth`` substring used to match ``author``
        # — attribution fields would silently turn into ``[redacted]``.
        result = _redact_secret_keys({"author": "alice", "title": "post"})
        assert result == {"author": "alice", "title": "post"}

    def test_redacts_authorization_header_whole_word(self):
        # ``Authorization`` is a whole word in the hint list — case-
        # insensitive boundary match still catches the HTTP-header form
        # that anything with substring matching used to catch via
        # ``auth``.
        result = _redact_secret_keys({"Authorization": "Bearer xyz"})
        assert result == {"Authorization": "[redacted]"}

    def test_redacts_token_at_underscore_boundaries(self):
        # Tightening the matcher must NOT regress the common
        # ``access_token`` / ``refresh_token`` shapes.
        result = _redact_secret_keys(
            {"access_token": "x", "refresh_token": "y", "id_token": "z"}
        )
        assert result == {
            "access_token": "[redacted]",
            "refresh_token": "[redacted]",
            "id_token": "[redacted]",
        }


class TestSanitizeChatMessage:
    def test_drops_refusal_and_metadata(self):
        msg = _msg(
            role="assistant",
            content="reply",
            refusal="I cannot do that",
            metadata={"file_ids": ["secret-file"], "model": "claude"},
        )
        sanitized = sanitize_chat_message(msg)
        # SharedChatMessage has no refusal/metadata fields, so they
        # cannot be exposed via the public payload.
        dumped = sanitized.model_dump()
        assert "refusal" not in dumped
        assert "metadata" not in dumped

    def test_redacts_tool_call_arguments(self):
        msg = _msg(
            role="assistant",
            tool_calls=[
                {
                    "id": "call_1",
                    "function": {
                        "name": "fetch",
                        "arguments": {
                            "url": "https://example.com",
                            "api_key": "sk-xyz",
                        },
                    },
                }
            ],
        )
        sanitized = sanitize_chat_message(msg)
        assert sanitized.tool_calls is not None
        args = sanitized.tool_calls[0]["function"]["arguments"]
        assert args["api_key"] == "[redacted]"
        assert args["url"] == "https://example.com"

    def test_strips_injected_context_from_user_messages(self):
        # Real injected contexts use a ``\n\n`` separator between the
        # closing tag and the user's actual text — the regex in
        # ``strip_injected_context_for_display`` requires that to anchor
        # the leading-block match.  This mirrors the production format.
        msg = _msg(
            role="user",
            content="<memory_context>secret</memory_context>\n\nhello",
        )
        sanitized = sanitize_chat_message(msg)
        assert sanitized.content is not None
        assert "secret" not in sanitized.content
        assert "hello" in sanitized.content

    def test_assistant_content_passes_through_unchanged(self):
        # The stripper is intentionally user-only — assistant content
        # may legitimately reference these tags in narrative form.
        msg = _msg(role="assistant", content="see <memory_context> docs")
        sanitized = sanitize_chat_message(msg)
        assert sanitized.content == "see <memory_context> docs"

    def test_no_tool_calls_yields_none(self):
        msg = _msg(role="assistant", content="hi")
        sanitized = sanitize_chat_message(msg)
        assert sanitized.tool_calls is None

    def test_redacts_secret_keys_in_tool_role_json_content(self):
        # Tool messages (``role=tool``) persist tool responses as
        # JSON-serialised strings.  A response payload with a secret-
        # shaped key MUST get redacted on the public share path — without
        # this, ``{"api_key": "sk-…"}`` in a tool output leaks verbatim.
        msg = _msg(
            role="tool",
            content=(
                '{"type":"agent_output",'
                '"execution":{"outputs":{"api_key":"sk-leaked",'
                '"public":"ok"}}}'
            ),
        )
        sanitized = sanitize_chat_message(msg)
        assert sanitized.content is not None
        import json as _json

        payload = _json.loads(sanitized.content)
        outputs = payload["execution"]["outputs"]
        assert outputs["api_key"] == "[redacted]"
        assert outputs["public"] == "ok"

    def test_tool_content_passes_through_when_not_json(self):
        # Non-JSON tool content (older / hand-rolled tool returns) must
        # pass through untouched — same posture as plain assistant text.
        msg = _msg(role="tool", content="completed without output")
        sanitized = sanitize_chat_message(msg)
        assert sanitized.content == "completed without output"

    def test_tool_content_redacts_secret_under_nested_list(self):
        # The recursive walk must redact secret-shaped subtrees inside
        # tool JSON arrays too.  Failure here would leak when a tool
        # returns ``{"api_keys": ["sk-1", "sk-2"]}``.
        msg = _msg(
            role="tool",
            content='{"api_keys":["sk-1","sk-2"],"count":2}',
        )
        sanitized = sanitize_chat_message(msg)
        assert sanitized.content is not None
        import json as _json

        payload = _json.loads(sanitized.content)
        assert payload["api_keys"] == "[redacted]"
        assert payload["count"] == 2
