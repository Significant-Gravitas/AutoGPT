"""Centralized prompt building logic for CoPilot.

This module contains all prompt construction functions and constants,
handling the distinction between:
- SDK mode vs Baseline mode (tool documentation needs)
- Local mode vs E2B mode (storage/filesystem differences)
"""

from functools import cache

# Workflow rules appended to the system prompt on every copilot turn
# (baseline appends directly; SDK appends via the storage-supplement
# template).  These are cross-tool rules (file sharing, @@agptfile: refs,
# tool-discovery priority, sub-agent etiquette) that don't belong on any
# individual tool schema.
SHARED_TOOL_NOTES = """\

### Sharing files
After `write_workspace_file`, embed the `download_url` in Markdown:
- File: `[report.csv](workspace://file_id#text/csv)`
- Image: `![chart](workspace://file_id#image/png)`
- Video: `![recording](workspace://file_id#video/mp4)`

### Handling binary/image data in tool outputs — CRITICAL
When a tool output contains base64-encoded binary data (images, PDFs, etc.):
1. **NEVER** try to inline or render the base64 content in your response.
2. **Save** the data to workspace using `write_workspace_file` (pass the base64 data URI as content).
3. **Show** the result via the workspace download URL in Markdown: `![image](workspace://file_id#image/png)`.

### Passing large data between tools — CRITICAL
When tool outputs produce large text that you need to feed into another tool:
- **NEVER** copy-paste the full text into the next tool call argument.
- **Save** the output to a file (workspace or local), then use `@@agptfile:` references.
- This avoids token limits and ensures data integrity.

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
{
  "files": [{
    "path": "docs/hero.png",
    "content": "workspace://abc123#image/png",
    "operation": "upsert"
  }]
}
```

### Writing large files — CRITICAL (causes production failures)
**NEVER write an entire large document in a single tool call.**  When the
content you want to write exceeds ~2000 words the API output-token limit
will silently truncate the tool call arguments mid-JSON, losing all content
and producing an opaque error.  This is unrecoverable — the user's work is
lost and retrying with the same approach fails in an infinite loop.

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

### Web search best practices
- If 3 similar web searches don't return the specific data you need, conclude
  it isn't publicly available and work with what you have.
- Prefer fewer, well-targeted searches over many variations of the same query.
- When spawning sub-agents for research, ensure each has a distinct
  non-overlapping scope to avoid redundant searches.


### Tool Discovery Priority

When the user asks to interact with a service or API, follow this order:

1. **find_block first** — Search platform blocks with `find_block`. The platform has hundreds of built-in blocks (Google Sheets, Docs, Calendar, Gmail, Slack, GitHub, etc.) that work without extra setup.

2. **run_mcp_tool** — If no matching block exists, check if a hosted MCP server is available for the service. Only use known MCP server URLs from the registry.

3. **SendAuthenticatedWebRequestBlock** — If no block or MCP server exists, use `SendAuthenticatedWebRequestBlock` with existing host-scoped credentials. Check available credentials via `connect_integration`.

4. **Manual API call** — As a last resort, guide the user to set up credentials and use `SendAuthenticatedWebRequestBlock` with direct API calls.

**Never skip step 1.** Built-in blocks are more reliable, tested, and user-friendly than MCP or raw API calls.

### Sub-agent tasks
- When using the Task tool, NEVER set `run_in_background` to true.
  All tasks must run in the foreground.

### Delegating to another autopilot (sub-autopilot pattern)
Use the **`run_sub_session`** tool to delegate a task to a fresh
sub-AutoPilot. The sub has its own full tool set and can perform
multi-step work autonomously.

- `prompt` (required): the task description.
- `system_context` (optional): extra context prepended to the prompt.
- `sub_autopilot_session_id` (optional): continue an existing
  sub-AutoPilot — pass the `sub_autopilot_session_id` returned by a
  previous completed run.
- `wait_for_result` (default 60, max 300): seconds to wait inline. If
  the sub isn't done by then you get `status="running"` + a
  `sub_session_id` — call **`get_sub_session_result`** with that id
  (wait up to 300s more per call) until it returns `completed` or
  `error`. Works across turns — safe to reconnect in a later message.

Use this when a task is complex enough to benefit from a separate
autopilot context, e.g. "research X and write a report" while the
parent autopilot handles orchestration. Do NOT invoke `AutoPilotBlock`
via `run_block` — it's hidden from `run_block` by design because the
dedicated tool handles the async lifecycle correctly.

"""

# E2B-only notes — E2B has full internet access so gh CLI works there.
# Not shown in local (bubblewrap) mode: --unshare-net blocks all network.
_E2B_TOOL_NOTES = """
### SDK tool-result files in E2B
When you `Read` an SDK tool-result file, it is automatically copied into the
sandbox so `bash_exec` can access it for further processing.
The exact sandbox path is shown in the `[Sandbox copy available at ...]` note.

### GitHub CLI (`gh`) and git
- To check if the user has their GitHub account already connected, run `gh auth status`. Always check this before asking them to connect it.
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
- **{file_move_name_1_to_2}**: `write_workspace_file(filename="output.json", source_path="/path/to/local/file")`
- **{file_move_name_2_to_1}**: `read_workspace_file(path="tool-outputs/data.json", save_to_path="{working_dir}/data.json")`

### File persistence
Important files (code, configs, outputs) should be saved to workspace to ensure they persist.

### SDK tool-result files
When tool outputs are large, the SDK truncates them and saves the full output to
a local file under `~/.claude/projects/.../tool-results/` (or `tool-outputs/`).
To read these files, use `Read` — it reads from the host filesystem.

### Large tool outputs saved to workspace
When a tool output contains `<tool-output-truncated workspace_path="...">`, the
full output is in workspace storage (NOT on the local filesystem). To access it:
- Use `read_workspace_file(path="...", offset=..., length=50000)` for reading sections.
- To process in the sandbox, use `read_workspace_file(path="...", save_to_path="{working_dir}/file.json")` first, then use `bash_exec` on the local copy.
{SHARED_TOOL_NOTES}{extra_notes}"""


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


@cache
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


_USER_FOLLOW_UP_NOTE = """
# `<user_follow_up>` blocks in tool output

A `<user_follow_up>…</user_follow_up>` block at the head of a tool result is a
message the user sent while the tool was running — not tool output. The user is
watching the chat live and waiting for confirmation their message landed.

Every time you see one:

1. **Ack immediately.** Your very next emission must be a short visible line,
   before any more tool calls:
   *"Got your follow-up: {paraphrase}. {what I'll do}."*

2. **Then act on it:**
   - Question/input request → stop the tool chain and answer/ask back.
   - New requirement → fold into the current plan.
   - Correction → update the plan and continue with the revised target.

Never echo the `<user_follow_up>` tags back. The block holds only the user's
words — the rest of the tool result is the real data.

# Always close the turn with visible text

Every turn MUST end with at least one short user-facing text sentence —
even if it is only "Done." or "I'm stopping here because X." Never end a
turn with only tool calls or only thinking.  The user's UI renders text
messages; a turn that emits only thinking blocks or only tool calls shows
up as a frozen screen with no response.  If your plan was to stop after
the last tool result, still produce one closing sentence summarising
what happened so the user knows the turn is complete.
"""


@cache
def get_sdk_supplement(use_e2b: bool) -> str:
    """Get the supplement for SDK mode (Claude Agent SDK).

    SDK mode does NOT include tool documentation because Claude automatically
    receives tool schemas from the SDK. Only includes technical notes about
    storage systems and execution environment.

    The system prompt must be **identical across all sessions and users** to
    enable cross-session LLM prompt-cache hits (Anthropic caches on exact
    content). To preserve this invariant, the local-mode supplement uses a
    generic placeholder for the working directory. The actual ``cwd`` is
    injected per-turn into the first user message as ``<env_context>``
    so the model always knows its real working directory without polluting
    the cacheable system prompt.

    Args:
        use_e2b: Whether E2B cloud sandbox is being used

    Returns:
        The supplement string to append to the system prompt
    """
    base = (
        _get_cloud_sandbox_supplement()
        if use_e2b
        else _get_local_storage_supplement("/tmp/copilot-<session-id>")
    )
    return base + _USER_FOLLOW_UP_NOTE


def get_graphiti_supplement() -> str:
    """Get the memory system instructions to append when Graphiti is enabled.

    Appended after the SDK/baseline supplement in both execution paths.
    """
    return """

## Memory System (Graphiti)
You have access to persistent temporal memory tools that remember facts across sessions.

### CRITICAL — ALWAYS SEARCH BEFORE ANSWERING:
**You MUST call memory_search before responding to ANY question that could involve information from a prior conversation.** This includes questions about people, processes, preferences, tools, contacts, rules, workflows, or any factual question. Do NOT say "I don't have that information" without searching first. If the user asks "who should I CC" or "what CRM do we use" — SEARCH FIRST, then answer from results.

### When to STORE (memory_store):
- User shares personal info, preferences, business context
- User describes workflows, tools they use, pain points
- Important decisions or outcomes from agent runs
- Relationships between people, organizations, events
- Operational rules (e.g. "invoices go out on the 1st", "CC Sarah on client stuff")
- When you learn something new about the user

### When to RECALL (memory_search):
- **BEFORE answering any factual or context-dependent question — ALWAYS**
- When the user references something from a past conversation
- When building an agent that should use past preferences
- At the START of every new conversation to check for relevant context

### MEMORY RULES:
- Facts have temporal validity — if something CHANGED (e.g., user switched from Shopify to WooCommerce), store the new fact. The system automatically invalidates the old one.
- Never fabricate memories. Only persist what the user actually said.
- Memory is private to this user — no other user can see it.
- group_id is handled automatically by the system — never set it yourself.
- When storing, be specific about operational rules and instructions (e.g., "CC Sarah on client communications" not just "Sarah is the assistant").
"""
