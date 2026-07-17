"""Tests for SharedExecutionFile workspace URI extraction logic."""

from backend.data.sharing.workspace_refs import extract_workspace_file_ids


class TestExtractWorkspaceFileIds:
    def test_extracts_simple_workspace_uri(self):
        outputs = {"image": ["workspace://abc123"]}
        assert extract_workspace_file_ids(outputs) == {"abc123"}

    def test_extracts_workspace_uri_with_mime_fragment(self):
        outputs = {"image": ["workspace://abc123#image/png"]}
        assert extract_workspace_file_ids(outputs) == {"abc123"}

    def test_extracts_multiple_files_from_multiple_outputs(self):
        outputs = {
            "images": ["workspace://file1#image/png", "workspace://file2#image/jpeg"],
            "video": ["workspace://file3#video/mp4"],
        }
        assert extract_workspace_file_ids(outputs) == {"file1", "file2", "file3"}

    def test_ignores_non_workspace_strings(self):
        outputs = {
            "text": ["hello world"],
            "url": ["https://example.com/image.png"],
            "data": ["data:image/png;base64,abc"],
        }
        assert extract_workspace_file_ids(outputs) == set()

    def test_ignores_path_references(self):
        """workspace:///path/to/file is a path reference, not a file ID."""
        outputs = {"file": ["workspace:///path/to/file.txt"]}
        assert extract_workspace_file_ids(outputs) == set()

    def test_handles_nested_dicts_in_output_values(self):
        outputs = {
            "result": [{"url": "workspace://nested-file#image/png", "label": "test"}]
        }
        assert extract_workspace_file_ids(outputs) == {"nested-file"}

    def test_handles_nested_lists_in_output_values(self):
        outputs = {"result": [["workspace://inner-file"]]}
        assert extract_workspace_file_ids(outputs) == {"inner-file"}

    def test_handles_empty_outputs(self):
        assert extract_workspace_file_ids({}) == set()

    def test_handles_non_string_values(self):
        outputs = {"count": [42], "flag": [True], "empty": [None]}
        assert extract_workspace_file_ids(outputs) == set()

    def test_deduplicates_repeated_file_ids(self):
        outputs = {
            "a": ["workspace://same-file#image/png"],
            "b": ["workspace://same-file#image/jpeg"],
        }
        assert extract_workspace_file_ids(outputs) == {"same-file"}

    def test_extracts_workspace_uri_embedded_in_text(self):
        """Assistant messages embed ``workspace://<id>`` URIs inline —
        e.g. ``Here's the chart: ![chart](workspace://abc#image/png)``.

        Previously the prefix-only match missed these, so the public
        viewer's allowlist lookup 404'd on files that the chat clearly
        referenced.  The extractor now picks up any embedded URI.
        """
        outputs = {"text": ["check out workspace://fake-id for details"]}
        assert extract_workspace_file_ids(outputs) == {"fake-id"}

    def test_extracts_workspace_uri_from_markdown_image_syntax(self):
        """The motivating case for the embedded-URI extraction:
        markdown image syntax wrapping a workspace:// URI."""
        outputs = {
            "content": "Here's the chart: ![chart](workspace://chart-id#image/png)"
        }
        assert extract_workspace_file_ids(outputs) == {"chart-id"}

    def test_mixed_workspace_and_non_workspace_outputs(self):
        outputs = {
            "image": ["workspace://real-file#image/png"],
            "text": ["just some text"],
            "url": ["https://example.com"],
        }
        assert extract_workspace_file_ids(outputs) == {"real-file"}

    def test_extracts_file_id_from_attached_files_block(self):
        """``[Attached files]`` blocks in user messages embed
        ``file_id=<uuid>`` tokens — those need to make the allowlist
        too so the public viewer can render them as artifacts."""
        content = (
            "Please review this.\n\n[Attached files]\n"
            "- report.pdf (application/pdf, 191.0 KB), "
            "file_id=550e8400-e29b-41d4-a716-446655440000\n"
            "Use read_workspace_file with the file_id to access file contents."
        )
        assert extract_workspace_file_ids(content) == {
            "550e8400-e29b-41d4-a716-446655440000"
        }

    def test_attached_files_uuid_pattern_rejects_non_uuids(self):
        """``file_id=`` followed by a non-UUID is ignored."""
        content = "see file_id=not-a-uuid here"
        assert extract_workspace_file_ids(content) == set()

    def test_extracts_file_id_from_tool_response_json(self):
        """Tool messages (``role="tool"``) persist their output as a
        JSON blob in ``ChatMessage.content`` — files referenced there
        appear as ``"file_id":"<uuid>"`` not ``file_id=<uuid>``."""
        content = (
            '{"type":"workspace_file_written",'
            '"message":"Wrote FounderStoryCard.tsx",'
            '"file_id":"b5c4f6df-f043-4826-88a2-33c33389c17d",'
            '"name":"FounderStoryCard.tsx"}'
        )
        assert extract_workspace_file_ids(content) == {
            "b5c4f6df-f043-4826-88a2-33c33389c17d"
        }

    def test_extracts_file_id_from_tool_response_json_with_spaces(self):
        """Allow ``"file_id" : "<uuid>"`` with arbitrary whitespace."""
        content = '{"file_id" :  "b5c4f6df-f043-4826-88a2-33c33389c17d"}'
        assert extract_workspace_file_ids(content) == {
            "b5c4f6df-f043-4826-88a2-33c33389c17d"
        }
