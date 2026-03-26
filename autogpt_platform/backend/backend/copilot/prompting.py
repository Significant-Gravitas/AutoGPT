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

### Sharing files
After `write_workspace_file`, embed the `download_url` in Markdown:
- File: `[report.csv](workspace://file_id#text/csv)`
- Image: `![chart](workspace://file_id#image/png)`
- Video: `![recording](workspace://file_id#video/mp4)`

### File references — @@agptfile:
Pass large file content to tools by reference: `@@agptfile:<uri>[<start>-<end>]`
- `workspace://<file_id>` or `workspace:///<path>` — workspace files
- `/absolute/path` — local/sandbox files
- `[start-end]` — optional 1-indexed line range
- Multiple refs per argument supported. Only `workspace://` and absolute paths are expanded.

Examples:
```
@@agptfile:workspace://abc123
@@agptfile:workspace://abc123[10-50]
@@agptfile:workspace:///reports/q1.md
@@agptfile:/tmp/copilot-<session>/output.py[1-80]
@@agptfile:/home/user/script.py
```

**Structured data**: When the entire argument is a single file reference, the platform auto-parses by extension/MIME. Supported: JSON, JSONL, CSV, TSV, YAML, TOML, Parquet, Excel (.xlsx only; legacy `.xls` is NOT supported). Unrecognised formats return plain string.

**Type coercion**: The platform auto-coerces expanded string values to match block input types (e.g. JSON string → `list[list[str]]`).

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

### Writing large files — CRITICAL
**Never write an entire large document in a single tool call.**  When the
content you want to write exceeds ~2000 words the tool call's output token
limit will silently truncate the arguments, producing an empty `{{}}` input
that fails repeatedly.

**Preferred: compose from file references.**  If the data is already in
files (tool outputs, workspace files), compose the report in one call
using `@@agptfile:` references — the system expands them inline:

```bash
cat > report.md << 'EOF'
# Research Report
## Data from web research
@@agptfile:/home/user/web_results.txt
## Block execution output
@@agptfile:workspace://<file_id>
## Conclusion
<brief synthesis>
EOF
```

**Fallback: write section-by-section.**  When you must generate content
from conversation context (no files to reference), split into multiple
`bash_exec` calls — one section per call:

```bash
cat > report.md << 'EOF'
# Section 1
<content from your earlier tool call results>
EOF
```
```bash
cat >> report.md << 'EOF'
# Section 2
<content from your earlier tool call results>
EOF
```
Use `cat >` for the first chunk and `cat >>` to append subsequent chunks.
Do not re-fetch or re-generate data you already have from prior tool calls.

After building the file, reference it with `@@agptfile:` in other tools:
`@@agptfile:/home/user/report.md`

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

### Shell & filesystem
- The SDK built-in Bash tool is NOT available. Use `bash_exec` for shell commands ({sandbox_type}). Working dir: `{working_dir}`
- SDK file tools (Read/Write/Edit/Glob/Grep) and `bash_exec` share one filesystem — use relative or absolute paths under this dir.
- `read_workspace_file`/`write_workspace_file` operate on **persistent cloud workspace storage** (separate from the working dir).

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
always use `Read` (NOT `bash_exec`, NOT `read_workspace_file`).
These files are on the host filesystem — `bash_exec` runs in the sandbox and
CANNOT access them. `read_workspace_file` reads from cloud workspace storage,
where SDK tool-results are NOT stored.
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
