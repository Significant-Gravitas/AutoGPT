"""Centralized prompt building logic for CoPilot.

This module contains all prompt construction functions and constants,
handling the distinction between:
- SDK mode vs Baseline mode (tool documentation needs)
- Local mode vs E2B mode (storage/filesystem differences)
"""

from backend.blocks.autopilot import AUTOPILOT_BLOCK_ID
from backend.copilot.tools import TOOL_REGISTRY

# Shared technical notes that apply to both SDK and baseline modes
_SHARED_TOOL_NOTES = f"""\

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

### Passing file content to tools — @@agptfile: references
Instead of copying large file contents into a tool argument, pass a file
reference and the platform will load the content for you.

Syntax: `@@agptfile:<uri>[<start>-<end>]`

- `<uri>` **must** start with `workspace://` or `/` (absolute path):
  - `workspace://<file_id>` — workspace file by ID
  - `workspace:///<path>` — workspace file by virtual path
  - `/absolute/local/path` — ephemeral or sdk_cwd file
  - E2B sandbox absolute path (e.g. `/home/user/script.py`)
- `[<start>-<end>]` is an optional 1-indexed inclusive line range.
- URIs that do not start with `workspace://` or `/` are **not** expanded.

Examples:
```
@@agptfile:workspace://abc123
@@agptfile:workspace://abc123[10-50]
@@agptfile:workspace:///reports/q1.md
@@agptfile:/tmp/copilot-<session>/output.py[1-80]
@@agptfile:/home/user/script.py
```

You can embed a reference inside any string argument, or use it as the entire
value.  Multiple references in one argument are all expanded.

**Structured data**: When the **entire** argument value is a single file
reference (no surrounding text), the platform automatically parses the file
content based on its extension or MIME type.  Supported formats: JSON, JSONL,
CSV, TSV, YAML, TOML, Parquet, and Excel (.xlsx — first sheet only).
For example, pass `@@agptfile:workspace://<id>` where the file is a `.csv` and
the rows will be parsed into `list[list[str]]` automatically.  If the format is
unrecognised or parsing fails, the content is returned as a plain string.
Legacy `.xls` files are **not** supported — only the modern `.xlsx` format.

**Type coercion**: The platform also coerces expanded values to match the
block's expected input types.  For example, if a block expects `list[list[str]]`
and the expanded value is a JSON string, it will be parsed into the correct type.

### Media file inputs (format: "file")
Some block inputs accept media files — their schema shows `"format": "file"`.
These fields accept:
- **`workspace://<file_id>`** or **`workspace://<file_id>#<mime>`** — preferred
  for large files (images, videos, PDFs). The platform passes the reference
  directly to the block without reading the content into memory.
- **`data:<mime>;base64,<payload>`** — inline base64 data URI, suitable for
  small files only.

When a block input has `format: "file"`, **pass the `workspace://` URI
directly as the value** (do NOT wrap it in `@@agptfile:`). This avoids large
payloads in tool arguments and preserves binary content (images, videos)
that would be corrupted by text encoding.

Example — committing an image file to GitHub:
```json
{{
  "files": [{{
    "path": "docs/hero.png",
    "content": "workspace://abc123#image/png",
    "operation": "upsert"
  }}]
}}
```

### Sub-agent tasks
- When using the Task tool, NEVER set `run_in_background` to true.
  All tasks must run in the foreground.

### Delegating to another autopilot (sub-autopilot pattern)
Use the **AutoPilotBlock** (`run_block` with block_id
`{AUTOPILOT_BLOCK_ID}`) to delegate a task to a fresh
autopilot instance.  The sub-autopilot has its own full tool set and can
perform multi-step work autonomously.

- **Input**: `prompt` (required) — the task description.
  Optional: `system_context` to constrain behavior, `session_id` to
  continue a previous conversation, `max_recursion_depth` (default 3).
- **Output**: `response` (text), `tool_calls` (list), `session_id`
  (for continuation), `conversation_history`, `token_usage`.

Use this when a task is complex enough to benefit from a separate
autopilot context, e.g. "research X and write a report" while the
parent autopilot handles orchestration.
"""

# E2B-only notes — E2B has full internet access so gh CLI works there.
# Not shown in local (bubblewrap) mode: --unshare-net blocks all network.
_E2B_TOOL_NOTES = """
### GitHub CLI (`gh`) and git
- If the user has connected their GitHub account, both `gh` and `git` are
  pre-authenticated — use them directly without any manual login step.
  `git` HTTPS operations (clone, push, pull) work automatically.
- If the token changes mid-session (e.g. user reconnects with a new token),
  run `gh auth setup-git` to re-register the credential helper.
- If `gh` or `git` fails with an authentication error (e.g. "authentication
  required", "could not read Username", or exit code 128), call
  `connect_integration(provider="github")` to surface the GitHub credentials
  setup card so the user can connect their account. Once connected, retry
  the operation.
- For operations that need broader access (e.g. private org repos, GitHub
  Actions), pass the required scopes: e.g.
  `connect_integration(provider="github", scopes=["repo", "read:org"])`.
"""


# Environment-specific supplement templates
def _build_storage_supplement(
    working_dir: str,
    sandbox_type: str,
    storage_system_1_name: str,
    storage_system_1_characteristics: list[str],
    storage_system_1_persistence: list[str],
    file_move_name_1_to_2: str,
    file_move_name_2_to_1: str,
    extra_notes: str = "",
) -> str:
    """Build storage/filesystem supplement for a specific environment.

    Template function handles all formatting (bullets, indentation, markdown).
    Callers provide clean data as lists of strings.

    Args:
        working_dir: Working directory path
        sandbox_type: Description of bash_exec sandbox
        storage_system_1_name: Name of primary storage (ephemeral or cloud)
        storage_system_1_characteristics: List of characteristic descriptions
        storage_system_1_persistence: List of persistence behavior descriptions
        file_move_name_1_to_2: Direction label for primary→persistent
        file_move_name_2_to_1: Direction label for persistent→primary
        extra_notes: Environment-specific notes appended after shared notes
    """
    # Format lists as bullet points with proper indentation
    characteristics = "\n".join(f"   - {c}" for c in storage_system_1_characteristics)
    persistence = "\n".join(f"   - {p}" for p in storage_system_1_persistence)

    return f"""

## Tool notes

### Shell commands
- The SDK built-in Bash tool is NOT available.  Use the `bash_exec` MCP tool
  for shell commands — it runs {sandbox_type}.

### Working directory
- Your working directory is: `{working_dir}`
- All SDK file tools AND `bash_exec` operate on the same filesystem
- Use relative paths or absolute paths under `{working_dir}` for all file operations

### Two storage systems — CRITICAL to understand

1. **{storage_system_1_name}** (`{working_dir}`):
{characteristics}
{persistence}

2. **Persistent workspace** (cloud storage):
   - Files here **survive across sessions indefinitely**

### Moving files between storages
- **{file_move_name_1_to_2}**: Copy to persistent workspace
- **{file_move_name_2_to_1}**: Download for processing

### File persistence
Important files (code, configs, outputs) should be saved to workspace to ensure they persist.

### SDK tool-result files
When tool outputs are large, the SDK truncates them and saves the full output to
a local file under `~/.claude/projects/.../tool-results/`. To read these files,
always use `read_file` or `Read` (NOT `read_workspace_file`).
`read_workspace_file` reads from cloud workspace storage, where SDK
tool-results are NOT stored.
{_SHARED_TOOL_NOTES}{extra_notes}"""


# Pre-built supplements for common environments
def _get_local_storage_supplement(cwd: str) -> str:
    """Local ephemeral storage (files lost between turns).

    Network is isolated (bubblewrap --unshare-net), so internet-dependent CLIs
    like gh will not work — no integration env-var notes are included.
    """
    return _build_storage_supplement(
        working_dir=cwd,
        sandbox_type="in a network-isolated sandbox",
        storage_system_1_name="Ephemeral working directory",
        storage_system_1_characteristics=[
            "Shared by SDK Read/Write/Edit/Glob/Grep tools AND `bash_exec`",
        ],
        storage_system_1_persistence=[
            "Files here are **lost between turns** — do NOT rely on them persisting",
            "Use for temporary work: running scripts, processing data, etc.",
        ],
        file_move_name_1_to_2="Ephemeral → Persistent",
        file_move_name_2_to_1="Persistent → Ephemeral",
    )


def _get_cloud_sandbox_supplement() -> str:
    """Cloud persistent sandbox (files survive across turns in session).

    E2B has full internet access, so integration tokens (GH_TOKEN etc.) are
    injected per command in bash_exec — include the CLI guidance notes.
    """
    return _build_storage_supplement(
        working_dir="/home/user",
        sandbox_type="in a cloud sandbox with full internet access",
        storage_system_1_name="Cloud sandbox",
        storage_system_1_characteristics=[
            "Shared by all file tools AND `bash_exec` — same filesystem",
            "Full Linux environment with internet access",
        ],
        storage_system_1_persistence=[
            "Files **persist across turns** within the current session",
            "Lost when the session expires (12 h inactivity)",
        ],
        file_move_name_1_to_2="Sandbox → Persistent",
        file_move_name_2_to_1="Persistent → Sandbox",
        extra_notes=_E2B_TOOL_NOTES,
    )


def _generate_tool_documentation() -> str:
    """Auto-generate tool documentation from TOOL_REGISTRY.

    NOTE: This is ONLY used in baseline mode (direct OpenAI API).
    SDK mode doesn't need it since Claude gets tool schemas automatically.

    This generates a complete list of available tools with their descriptions,
    ensuring the documentation stays in sync with the actual tool implementations.
    All workflow guidance is now embedded in individual tool descriptions.

    Only documents tools that are available in the current environment
    (checked via tool.is_available property).
    """
    docs = "\n## AVAILABLE TOOLS\n\n"

    # Sort tools alphabetically for consistent output
    # Filter by is_available to match get_available_tools() behavior
    for name in sorted(TOOL_REGISTRY.keys()):
        tool = TOOL_REGISTRY[name]
        if not tool.is_available:
            continue
        schema = tool.as_openai_tool()
        desc = schema["function"].get("description", "No description available")
        # Format as bullet list with tool name in code style
        docs += f"- **`{name}`**: {desc}\n"

    return docs


def get_sdk_supplement(use_e2b: bool, cwd: str = "") -> str:
    """Get the supplement for SDK mode (Claude Agent SDK).

    SDK mode does NOT include tool documentation because Claude automatically
    receives tool schemas from the SDK. Only includes technical notes about
    storage systems and execution environment.

    Args:
        use_e2b: Whether E2B cloud sandbox is being used
        cwd: Current working directory (only used in local_storage mode)

    Returns:
        The supplement string to append to the system prompt
    """
    if use_e2b:
        return _get_cloud_sandbox_supplement()
    return _get_local_storage_supplement(cwd)


def get_baseline_supplement() -> str:
    """Get the supplement for baseline mode (direct OpenAI API).

    Baseline mode INCLUDES auto-generated tool documentation because the
    direct API doesn't automatically provide tool schemas to Claude.
    Also includes shared technical notes (but NOT SDK-specific environment details).

    Returns:
        The supplement string to append to the system prompt
    """
    tool_docs = _generate_tool_documentation()
    return tool_docs + _SHARED_TOOL_NOTES
