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

    def test_does_not_match_workspace_substring_in_text(self):
        """Plain text that contains workspace:// as a substring should NOT be extracted
        because the value itself must start with workspace://."""
        outputs = {"text": ["check out workspace://fake-id for details"]}
        # The string starts with "check out", not "workspace://", so no match
        assert extract_workspace_file_ids(outputs) == set()

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
