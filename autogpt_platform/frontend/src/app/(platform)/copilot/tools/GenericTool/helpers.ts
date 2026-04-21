import type { ToolUIPart } from "ai";

/* ------------------------------------------------------------------ */
/*  Sub-agent tool name constants                                      */
/* ------------------------------------------------------------------ */

export const TOOL_AGENT = "Agent";
export const TOOL_TASK = "Task";
export const TOOL_TASK_OUTPUT = "TaskOutput";

/* ------------------------------------------------------------------ */
/*  Tool name helpers                                                  */
/* ------------------------------------------------------------------ */

export function extractToolName(part: ToolUIPart): string {
  return part.type.replace(/^tool-/, "");
}

// Specific-case labels for tools whose auto-formatted name reads awkwardly
// alongside a "Running …" prefix (e.g. avoid "Running Run sub session").
const TOOL_DISPLAY_NAMES: Record<string, string> = {
  run_sub_session: "Sub-AutoPilot",
  get_sub_session_result: "Sub-AutoPilot result",
  run_agent: "Agent",
  view_agent_output: "Agent output",
  run_block: "Action",
  run_mcp_tool: "MCP tool",
  get_agent_building_guide: "Agent building guide",
};

export function formatToolName(name: string): string {
  const override = TOOL_DISPLAY_NAMES[name];
  if (override) return override;
  // Drop a redundant "run_" prefix so "Running Run agent" → "Running agent".
  const stripped = name.startsWith("run_") ? name.slice(4) : name;
  return stripped.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
}

/* ------------------------------------------------------------------ */
/*  Tool categorization                                                */
/* ------------------------------------------------------------------ */

export type ToolCategory =
  | "bash"
  | "web"
  | "browser"
  | "file-read"
  | "file-write"
  | "file-delete"
  | "file-list"
  | "search"
  | "edit"
  | "todo"
  | "compaction"
  | "agent"
  | "other";

export function getToolCategory(toolName: string): ToolCategory {
  switch (toolName) {
    case "bash_exec":
      return "bash";
    case "web_fetch":
    case "WebSearch":
    case "WebFetch":
      return "web";
    case "browser_navigate":
    case "browser_act":
    case "browser_screenshot":
      return "browser";
    case "read_workspace_file":
    case "read_file":
    case "Read":
      return "file-read";
    case "write_workspace_file":
    case "write_file":
    case "Write":
      return "file-write";
    case "delete_workspace_file":
      return "file-delete";
    case "list_workspace_files":
    case "glob":
    case "Glob":
      return "file-list";
    case "grep":
    case "Grep":
      return "search";
    case "edit_file":
    case "Edit":
      return "edit";
    case "TodoWrite":
      return "todo";
    case "context_compaction":
      return "compaction";
    case TOOL_AGENT:
    case TOOL_TASK:
    case TOOL_TASK_OUTPUT:
      return "agent";
    default:
      return "other";
  }
}

/* ------------------------------------------------------------------ */
/*  Input summary                                                      */
/* ------------------------------------------------------------------ */

function getInputSummary(toolName: string, input: unknown): string | null {
  if (!input || typeof input !== "object") return null;
  const inp = input as Record<string, unknown>;

  switch (toolName) {
    case "bash_exec":
      return typeof inp.command === "string" ? inp.command : null;
    case "web_fetch":
    case "WebFetch":
      return typeof inp.url === "string" ? inp.url : null;
    case "WebSearch":
      return typeof inp.query === "string" ? inp.query : null;
    case "browser_navigate":
      return typeof inp.url === "string" ? inp.url : null;
    case "browser_act":
      if (typeof inp.action !== "string") return null;
      return typeof inp.target === "string"
        ? `${inp.action} ${inp.target}`
        : inp.action;
    case "browser_screenshot":
      return null;
    case "read_workspace_file":
    case "read_file":
    case "Read":
      return (
        (typeof inp.file_path === "string" ? inp.file_path : null) ??
        (typeof inp.path === "string" ? inp.path : null)
      );
    case "write_workspace_file":
    case "write_file":
    case "Write":
      return (
        (typeof inp.file_path === "string" ? inp.file_path : null) ??
        (typeof inp.path === "string" ? inp.path : null)
      );
    case "delete_workspace_file":
      return typeof inp.file_path === "string" ? inp.file_path : null;
    case "glob":
    case "Glob":
      return typeof inp.pattern === "string" ? inp.pattern : null;
    case "grep":
    case "Grep":
      return typeof inp.pattern === "string" ? inp.pattern : null;
    case "edit_file":
    case "Edit":
      return typeof inp.file_path === "string" ? inp.file_path : null;
    case "TodoWrite": {
      const todos = Array.isArray(inp.todos) ? inp.todos : [];
      const active = todos.find(
        (t: unknown) =>
          t !== null &&
          typeof t === "object" &&
          (t as Record<string, unknown>).status === "in_progress",
      ) as Record<string, unknown> | undefined;
      if (active && typeof active.activeForm === "string")
        return active.activeForm;
      if (active && typeof active.content === "string") return active.content;
      return null;
    }
    case TOOL_AGENT:
    case TOOL_TASK:
      return typeof inp.description === "string"
        ? inp.description
        : typeof inp.prompt === "string"
          ? truncate(inp.prompt, 60)
          : null;
    case TOOL_TASK_OUTPUT:
      return typeof inp.agentId === "string" ? inp.agentId : null;
    default:
      return null;
  }
}

export function truncate(text: string, maxLen: number): string {
  if (text.length <= maxLen) return text;
  return text.slice(0, maxLen).trimEnd() + "\u2026";
}

const STRIPPABLE_EXTENSIONS =
  /\.(md|csv|json|txt|yaml|yml|xml|html|js|ts|py|sh|toml|cfg|ini|log|pdf|png|jpg|jpeg|gif|svg|mp4|mp3|wav|zip|tar|gz)$/i;

export function humanizeFileName(filePath: string): string {
  const fileName = filePath.split("/").pop() ?? filePath;
  const stem = fileName.replace(STRIPPABLE_EXTENSIONS, "");
  const words = stem
    .replace(/[_-]/g, " ")
    .split(/\s+/)
    .filter(Boolean)
    .map((w) => {
      if (w === w.toUpperCase()) return w;
      return w.charAt(0).toUpperCase() + w.slice(1).toLowerCase();
    });
  return `"${words.join(" ")}"`;
}

/* ------------------------------------------------------------------ */
/*  Animation text                                                     */
/* ------------------------------------------------------------------ */

export function getAnimationText(
  part: ToolUIPart,
  category: ToolCategory,
): string {
  const toolName = extractToolName(part);
  const summary = getInputSummary(toolName, part.input);
  const shortSummary = summary ? truncate(summary, 60) : null;

  switch (part.state) {
    case "input-streaming":
    case "input-available": {
      switch (category) {
        case "bash":
          return shortSummary
            ? `Running: ${shortSummary}`
            : "Running command\u2026";
        case "web":
          if (toolName === "WebSearch") {
            return shortSummary
              ? `Searching "${shortSummary}"`
              : "Searching the web\u2026";
          }
          return shortSummary
            ? `Fetching ${shortSummary}`
            : "Fetching web content\u2026";
        case "browser":
          if (toolName === "browser_screenshot")
            return "Taking screenshot\u2026";
          return shortSummary
            ? `Browsing ${shortSummary}`
            : "Interacting with browser\u2026";
        case "file-read":
          return summary
            ? `Reading ${humanizeFileName(summary)}`
            : "Reading file\u2026";
        case "file-write":
          return summary
            ? `Writing ${humanizeFileName(summary)}`
            : "Writing file\u2026";
        case "file-delete":
          return summary
            ? `Deleting ${humanizeFileName(summary)}`
            : "Deleting file\u2026";
        case "file-list":
          return shortSummary
            ? `Listing ${shortSummary}`
            : "Listing files\u2026";
        case "search":
          return shortSummary
            ? `Searching for "${shortSummary}"`
            : "Searching\u2026";
        case "edit":
          return summary
            ? `Editing ${humanizeFileName(summary)}`
            : "Editing file\u2026";
        case "todo":
          return shortSummary ? `${shortSummary}` : "Updating task list\u2026";
        case "compaction":
          return "Summarizing earlier messages\u2026";
        case "agent":
          if (toolName === TOOL_TASK_OUTPUT)
            return shortSummary
              ? `Checking agent ${shortSummary}\u2026`
              : "Checking agent result\u2026";
          return shortSummary
            ? `Running agent: ${shortSummary}`
            : "Starting agent\u2026";
        default:
          return `Running ${formatToolName(toolName)}\u2026`;
      }
    }
    case "output-available": {
      switch (category) {
        case "bash":
          // Subtitle always shows WHAT ran. The accordion title + description
          // carry HOW it ended (exit code / "timed out"), so repeating the
          // exit status here would just double up.
          return shortSummary ? `Ran: ${shortSummary}` : "Command completed";
        case "web":
          if (toolName === "WebSearch") {
            return shortSummary
              ? `Searched "${shortSummary}"`
              : "Web search completed";
          }
          return shortSummary
            ? `Fetched ${shortSummary}`
            : "Fetched web content";
        case "browser":
          if (toolName === "browser_screenshot") return "Screenshot captured";
          return shortSummary
            ? `Browsed ${shortSummary}`
            : "Browser action completed";
        case "file-read":
          return summary
            ? `Read ${humanizeFileName(summary)}`
            : "File read completed";
        case "file-write":
          return summary
            ? `Wrote ${humanizeFileName(summary)}`
            : "File written";
        case "file-delete":
          return summary
            ? `Deleted ${humanizeFileName(summary)}`
            : "File deleted";
        case "file-list":
          return "Listed files";
        case "search":
          return shortSummary
            ? `Searched for "${shortSummary}"`
            : "Search completed";
        case "edit":
          return summary
            ? `Edited ${humanizeFileName(summary)}`
            : "Edit completed";
        case "todo":
          return "Updated task list";
        case "compaction":
          return "Earlier messages were summarized";
        case "agent": {
          if (toolName === TOOL_TASK_OUTPUT) {
            const taskOut =
              part.output && typeof part.output === "object"
                ? (part.output as Record<string, unknown>)
                : null;
            if (taskOut?.retrieval_status === "timeout")
              return "Agent still running\u2026";
            return "Agent result received";
          }
          const agentOut =
            part.output && typeof part.output === "object"
              ? (part.output as Record<string, unknown>)
              : null;
          if (agentOut?.isAsync || agentOut?.status === "async_launched")
            return shortSummary
              ? `Agent started (background): ${shortSummary}`
              : "Agent started in background";
          return shortSummary
            ? `Agent completed: ${shortSummary}`
            : "Agent completed";
        }
        default:
          return `${formatToolName(toolName)} completed`;
      }
    }
    case "output-error": {
      switch (category) {
        case "bash":
          return "Command failed";
        case "web":
          return toolName === "WebSearch" ? "Search failed" : "Fetch failed";
        case "browser":
          return "Browser action failed";
        default:
          return `${formatToolName(toolName)} failed`;
      }
    }
    default:
      return `Running ${formatToolName(toolName)}\u2026`;
  }
}
