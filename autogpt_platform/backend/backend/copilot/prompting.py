"""Centralized prompt building logic for CoPilot.

This module contains all prompt construction functions and constants,
handling the distinction between:
- SDK mode vs Baseline mode (tool documentation needs)
- Local mode vs E2B mode (storage/filesystem differences)
"""

from backend.copilot.prompt_constants import KEY_WORKFLOWS
from backend.copilot.tools import TOOL_REGISTRY

# Shared technical notes that apply to both SDK and baseline modes
_SHARED_TOOL_NOTES = """\

### Web search and research
- **`web_search(query)`** — Search the web for current information (uses Claude's
  native web search). Use this when you need up-to-date information, facts,
  statistics, or current events that are beyond your knowledge cutoff.
- **`web_fetch(url)`** — Retrieve and analyze content from a specific URL.
  Use this when you have a specific URL to read (documentation, articles, etc.).

### Sharing files with the user
After saving a file to the persistent workspace with `write_workspace_file`,
share it with the user by embedding the `download_url` from the response in
your message as a Markdown link or image:

- **Any file** — shows as a clickable download link:
  `[report.csv](workspace://file_id#text/csv)`
- **Image** — renders inline in chat:
  `![chart](workspace://file_id#image/png)`
- **Video** — renders inline in chat with player controls:
  `![recording](workspace://file_id#video/mp4)`

The `download_url` field in the `write_workspace_file` response is already
in the correct format — paste it directly after the `(` in the Markdown.

### Long-running tools
Long-running tools (create_agent, edit_agent, etc.) are handled
asynchronously.  You will receive an immediate response; the actual result
is delivered to the user via a background stream.

### Large tool outputs
When a tool output exceeds the display limit, it is automatically saved to
the persistent workspace.  The truncated output includes a
`<tool-output-truncated>` tag with the workspace path.  Use
`read_workspace_file(path="...", offset=N, length=50000)` to retrieve
additional sections.

### Sub-agent tasks
- When using the Task tool, NEVER set `run_in_background` to true.
  All tasks must run in the foreground.
"""

# Local mode supplement (ephemeral working directory)
_LOCAL_TOOL_SUPPLEMENT = (
    """

## Tool notes

### Shell commands
- The SDK built-in Bash tool is NOT available.  Use the `bash_exec` MCP tool
  for shell commands — it runs in a network-isolated sandbox.

### Working directory
- Your working directory is: `{cwd}`
- All SDK Read/Write/Edit/Glob/Grep tools AND `bash_exec` operate inside this
  directory.  This is the ONLY writable path — do not attempt to read or write
  anywhere else on the filesystem.
- Use relative paths or absolute paths under `{cwd}` for all file operations.

### Two storage systems — CRITICAL to understand

1. **Ephemeral working directory** (`{cwd}`):
   - Shared by SDK Read/Write/Edit/Glob/Grep tools AND `bash_exec`
   - Files here are **lost between turns** — do NOT rely on them persisting
   - Use for temporary work: running scripts, processing data, etc.

2. **Persistent workspace** (cloud storage):
   - Files here **survive across turns and sessions**
   - Use `write_workspace_file` to save important files (code, outputs, configs)
   - Use `read_workspace_file` to retrieve previously saved files
   - Use `list_workspace_files` to see what files you've saved before
   - Call `list_workspace_files(include_all_sessions=True)` to see files from
     all sessions

### Moving files between ephemeral and persistent storage
- **Ephemeral → Persistent**: Use `write_workspace_file` with either:
  - `content` param (plain text) — for text files
  - `source_path` param — to copy any file directly from the ephemeral dir
- **Persistent → Ephemeral**: Use `read_workspace_file` with `save_to_path`
  param to download a workspace file to the ephemeral dir for processing

### File persistence workflow
When you create or modify important files (code, configs, outputs), you MUST:
1. Save them using `write_workspace_file` so they persist
2. At the start of a new turn, call `list_workspace_files` to see what files
   are available from previous turns
"""
    + _SHARED_TOOL_NOTES
)

# E2B mode supplement (persistent cloud sandbox)
_E2B_TOOL_SUPPLEMENT = (
    """

## Tool notes

### Shell commands
- The SDK built-in Bash tool is NOT available.  Use the `bash_exec` MCP tool
  for shell commands — it runs in a cloud sandbox with full internet access.

### Working directory
- Your working directory is: `/home/user` (cloud sandbox)
- All file tools (`read_file`, `write_file`, `edit_file`, `glob`, `grep`)
  AND `bash_exec` operate on the **same cloud sandbox filesystem**.
- Files created by `bash_exec` are immediately visible to `read_file` and
  vice-versa — they share one filesystem.
- Use relative paths (resolved from `/home/user`) or absolute paths.

### Two storage systems — CRITICAL to understand

1. **Cloud sandbox** (`/home/user`):
   - Shared by all file tools AND `bash_exec` — same filesystem
   - Files **persist across turns** within the current session
   - Full Linux environment with internet access
   - Lost when the session expires (12 h inactivity)

2. **Persistent workspace** (cloud storage):
   - Files here **survive across sessions indefinitely**
   - Use `write_workspace_file` to save important files permanently
   - Use `read_workspace_file` to retrieve previously saved files
   - Use `list_workspace_files` to see what files you've saved before
   - Call `list_workspace_files(include_all_sessions=True)` to see files from
     all sessions

### Moving files between sandbox and persistent storage
- **Sandbox → Persistent**: Use `write_workspace_file` with `source_path`
  to copy from the sandbox to permanent storage
- **Persistent → Sandbox**: Use `read_workspace_file` with `save_to_path`
  to download into the sandbox for processing

### File persistence workflow
Important files that must survive beyond this session should be saved with
`write_workspace_file`.  Sandbox files persist across turns but are lost
when the session expires.
"""
    + _SHARED_TOOL_NOTES
)


def _generate_tool_documentation() -> str:
    """Auto-generate tool documentation from TOOL_REGISTRY.

    NOTE: This is ONLY used in baseline mode (direct OpenAI API).
    SDK mode doesn't need it since Claude gets tool schemas automatically.

    This generates a complete list of available tools with their descriptions,
    ensuring the documentation stays in sync with the actual tool implementations.
    """
    docs = "\n## AVAILABLE TOOLS\n\n"

    # Sort tools alphabetically for consistent output
    for name in sorted(TOOL_REGISTRY.keys()):
        tool = TOOL_REGISTRY[name]
        schema = tool.as_openai_tool()
        desc = schema["function"].get("description", "No description available")
        # Format as bullet list with tool name in code style
        docs += f"- **`{name}`**: {desc}\n"

    # Add workflow guidance for key tools
    docs += KEY_WORKFLOWS

    return docs


def get_sdk_supplement(use_e2b: bool, cwd: str = "") -> str:
    """Get the supplement for SDK mode (Claude Agent SDK).

    SDK mode does NOT include tool documentation because Claude automatically
    receives tool schemas from the SDK. Only includes technical notes about
    storage systems and execution environment.

    Args:
        use_e2b: Whether E2B cloud sandbox is being used
        cwd: Current working directory (only used in local mode)

    Returns:
        The supplement string to append to the system prompt
    """
    if use_e2b:
        return _E2B_TOOL_SUPPLEMENT
    return _LOCAL_TOOL_SUPPLEMENT.format(cwd=cwd)


def get_baseline_supplement(use_e2b: bool, cwd: str = "") -> str:
    """Get the supplement for baseline mode (direct OpenAI API).

    Baseline mode INCLUDES auto-generated tool documentation because the
    direct API doesn't automatically provide tool schemas to Claude.
    Also includes technical notes about storage systems.

    Args:
        use_e2b: Whether E2B cloud sandbox is being used
        cwd: Current working directory (only used in local mode)

    Returns:
        The supplement string to append to the system prompt
    """
    tool_docs = _generate_tool_documentation()

    # Append environment-specific notes
    if use_e2b:
        return tool_docs + _E2B_TOOL_SUPPLEMENT
    return tool_docs + _LOCAL_TOOL_SUPPLEMENT.format(cwd=cwd)
