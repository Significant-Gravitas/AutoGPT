import { describe, expect, it } from "vitest";
import type { ToolUIPart } from "ai";
import {
  TOOL_AGENT,
  TOOL_TASK,
  TOOL_TASK_OUTPUT,
  extractToolName,
  formatToolName,
  getToolCategory,
  truncate,
  humanizeFileName,
  getAnimationText,
} from "../helpers";

describe("extractToolName", () => {
  it("strips the tool- prefix from part.type", () => {
    const part = { type: "tool-bash_exec" } as unknown as ToolUIPart;
    expect(extractToolName(part)).toBe("bash_exec");
  });

  it("returns type unchanged when there is no tool- prefix", () => {
    const part = { type: "Read" } as unknown as ToolUIPart;
    expect(extractToolName(part)).toBe("Read");
  });
});

describe("formatToolName", () => {
  it("replaces underscores with spaces and capitalizes first letter", () => {
    expect(formatToolName("bash_exec")).toBe("Bash exec");
  });

  it("capitalizes a single word", () => {
    expect(formatToolName("read")).toBe("Read");
  });

  it("handles already capitalized names", () => {
    expect(formatToolName("WebSearch")).toBe("WebSearch");
  });

  it("uses friendly display name for sub-AutoPilot tools", () => {
    expect(formatToolName("run_sub_session")).toBe("Sub-AutoPilot");
    expect(formatToolName("get_sub_session_result")).toBe(
      "Sub-AutoPilot result",
    );
  });

  it("uses the 'Action' label for run_block (frontend parlance)", () => {
    expect(formatToolName("run_block")).toBe("Action");
  });

  it("strips redundant 'run_' prefix for other run_* tools", () => {
    // "Running Run agent" reads awkwardly — the override produces
    // "Running Agent".
    expect(formatToolName("run_agent")).toBe("Agent");
  });
});

describe("getToolCategory", () => {
  it("returns 'bash' for bash_exec", () => {
    expect(getToolCategory("bash_exec")).toBe("bash");
  });

  it("returns 'web' for web_fetch, WebSearch, WebFetch", () => {
    expect(getToolCategory("web_fetch")).toBe("web");
    expect(getToolCategory("WebSearch")).toBe("web");
    expect(getToolCategory("WebFetch")).toBe("web");
  });

  it("returns 'browser' for browser tools", () => {
    expect(getToolCategory("browser_navigate")).toBe("browser");
    expect(getToolCategory("browser_act")).toBe("browser");
    expect(getToolCategory("browser_screenshot")).toBe("browser");
  });

  it("returns 'file-read' for read tools", () => {
    expect(getToolCategory("read_workspace_file")).toBe("file-read");
    expect(getToolCategory("read_file")).toBe("file-read");
    expect(getToolCategory("Read")).toBe("file-read");
  });

  it("returns 'file-write' for write tools", () => {
    expect(getToolCategory("write_workspace_file")).toBe("file-write");
    expect(getToolCategory("write_file")).toBe("file-write");
    expect(getToolCategory("Write")).toBe("file-write");
  });

  it("returns 'file-delete' for delete tool", () => {
    expect(getToolCategory("delete_workspace_file")).toBe("file-delete");
  });

  it("returns 'file-list' for listing tools", () => {
    expect(getToolCategory("list_workspace_files")).toBe("file-list");
    expect(getToolCategory("glob")).toBe("file-list");
    expect(getToolCategory("Glob")).toBe("file-list");
  });

  it("returns 'search' for grep tools", () => {
    expect(getToolCategory("grep")).toBe("search");
    expect(getToolCategory("Grep")).toBe("search");
  });

  it("returns 'edit' for edit tools", () => {
    expect(getToolCategory("edit_file")).toBe("edit");
    expect(getToolCategory("Edit")).toBe("edit");
  });

  it("returns 'todo' for TodoWrite", () => {
    expect(getToolCategory("TodoWrite")).toBe("todo");
  });

  it("returns 'compaction' for context_compaction", () => {
    expect(getToolCategory("context_compaction")).toBe("compaction");
  });

  it("returns 'agent' for agent tools", () => {
    expect(getToolCategory(TOOL_AGENT)).toBe("agent");
    expect(getToolCategory(TOOL_TASK)).toBe("agent");
    expect(getToolCategory(TOOL_TASK_OUTPUT)).toBe("agent");
  });

  it("returns 'other' for unknown tools", () => {
    expect(getToolCategory("unknown_tool")).toBe("other");
  });
});

describe("truncate", () => {
  it("returns text unchanged when shorter than maxLen", () => {
    expect(truncate("short", 10)).toBe("short");
  });

  it("returns text unchanged when equal to maxLen", () => {
    expect(truncate("12345", 5)).toBe("12345");
  });

  it("truncates and appends ellipsis when longer than maxLen", () => {
    const result = truncate("this is a very long string", 10);
    expect(result).toBe("this is a\u2026");
    expect(result.length).toBeLessThanOrEqual(11);
  });
});

describe("humanizeFileName", () => {
  it("strips path and extension, titlecases words", () => {
    expect(humanizeFileName("/path/to/my-file.ts")).toBe('"My File"');
  });

  it("handles underscores", () => {
    expect(humanizeFileName("some_module_name.py")).toBe('"Some Module Name"');
  });

  it("preserves all-caps words", () => {
    expect(humanizeFileName("README.md")).toBe('"README"');
  });

  it("handles file with no extension", () => {
    expect(humanizeFileName("Makefile")).toBe('"Makefile"');
  });

  it("strips known extensions", () => {
    expect(humanizeFileName("data.json")).toBe('"Data"');
    expect(humanizeFileName("image.png")).toBe('"Image"');
    expect(humanizeFileName("archive.tar")).toBe('"Archive"');
  });
});

describe("getAnimationText", () => {
  function makePart(
    overrides: Partial<ToolUIPart> & { type: string },
  ): ToolUIPart {
    return {
      state: "input-streaming",
      input: undefined,
      output: undefined,
      ...overrides,
    } as unknown as ToolUIPart;
  }

  it("shows streaming text for bash with command summary", () => {
    const part = makePart({
      type: "tool-bash_exec",
      state: "input-available",
      input: { command: "ls -la" },
    });
    expect(getAnimationText(part, "bash")).toBe("Running: ls -la");
  });

  it("shows generic streaming text for bash without input", () => {
    const part = makePart({
      type: "tool-bash_exec",
      state: "input-streaming",
    });
    expect(getAnimationText(part, "bash")).toBe("Running command\u2026");
  });

  it("shows completed text for bash", () => {
    const part = makePart({
      type: "tool-bash_exec",
      state: "output-available",
      input: { command: "echo hello" },
      output: { exit_code: 0 },
    });
    expect(getAnimationText(part, "bash")).toBe("Ran: echo hello");
  });

  it("shows exit code on non-zero exit", () => {
    const part = makePart({
      type: "tool-bash_exec",
      state: "output-available",
      input: { command: "false" },
      output: { exit_code: 1 },
    });
    expect(getAnimationText(part, "bash")).toBe("Command exited with code 1");
  });

  it("shows error text for bash failure", () => {
    const part = makePart({
      type: "tool-bash_exec",
      state: "output-error",
    });
    expect(getAnimationText(part, "bash")).toBe("Command failed");
  });

  it("shows searching text for WebSearch", () => {
    const part = makePart({
      type: "tool-WebSearch",
      state: "input-available",
      input: { query: "test query" },
    });
    expect(getAnimationText(part, "web")).toBe('Searching "test query"');
  });

  it("shows fetching text for web_fetch", () => {
    const part = makePart({
      type: "tool-web_fetch",
      state: "input-available",
      input: { url: "https://example.com" },
    });
    expect(getAnimationText(part, "web")).toBe("Fetching https://example.com");
  });

  it("shows reading text for file-read", () => {
    const part = makePart({
      type: "tool-Read",
      state: "input-available",
      input: { file_path: "/src/index.ts" },
    });
    expect(getAnimationText(part, "file-read")).toBe('Reading "Index"');
  });

  it("shows writing text for file-write", () => {
    const part = makePart({
      type: "tool-Write",
      state: "input-available",
      input: { file_path: "/src/output.json" },
    });
    expect(getAnimationText(part, "file-write")).toBe('Writing "Output"');
  });

  it("shows compaction text", () => {
    const part = makePart({
      type: "tool-context_compaction",
      state: "input-streaming",
    });
    expect(getAnimationText(part, "compaction")).toBe(
      "Summarizing earlier messages\u2026",
    );
  });

  it("shows completed compaction text", () => {
    const part = makePart({
      type: "tool-context_compaction",
      state: "output-available",
    });
    expect(getAnimationText(part, "compaction")).toBe(
      "Earlier messages were summarized",
    );
  });

  it("shows agent streaming text with description", () => {
    const part = makePart({
      type: `tool-${TOOL_AGENT}`,
      state: "input-available",
      input: { description: "analyze code" },
    });
    expect(getAnimationText(part, "agent")).toBe("Running agent: analyze code");
  });

  it("shows agent completed for async launch", () => {
    const part = makePart({
      type: `tool-${TOOL_AGENT}`,
      state: "output-available",
      output: { isAsync: true },
    });
    expect(getAnimationText(part, "agent")).toBe("Agent started in background");
  });

  it("shows default streaming text for unknown tools", () => {
    const part = makePart({
      type: "tool-custom_tool",
      state: "input-streaming",
    });
    expect(getAnimationText(part, "other")).toBe("Running Custom tool\u2026");
  });

  it("shows default completed text for unknown tools", () => {
    const part = makePart({
      type: "tool-custom_tool",
      state: "output-available",
    });
    expect(getAnimationText(part, "other")).toBe("Custom tool completed");
  });

  it("shows default error text for unknown tools", () => {
    const part = makePart({
      type: "tool-custom_tool",
      state: "output-error",
    });
    expect(getAnimationText(part, "other")).toBe("Custom tool failed");
  });

  it("shows browser screenshot streaming", () => {
    const part = makePart({
      type: "tool-browser_screenshot",
      state: "input-available",
    });
    expect(getAnimationText(part, "browser")).toBe("Taking screenshot\u2026");
  });

  it("shows todo streaming text", () => {
    const part = makePart({
      type: "tool-TodoWrite",
      state: "input-available",
      input: {
        todos: [
          {
            content: "Fix bug",
            status: "in_progress",
            activeForm: "Fixing the bug",
          },
        ],
      },
    });
    expect(getAnimationText(part, "todo")).toBe("Fixing the bug");
  });

  it("shows TaskOutput timeout text", () => {
    const part = makePart({
      type: `tool-${TOOL_TASK_OUTPUT}`,
      state: "output-available",
      output: { retrieval_status: "timeout" },
    });
    expect(getAnimationText(part, "agent")).toBe("Agent still running\u2026");
  });

  it("shows agent completed with summary for sync agent", () => {
    const part = makePart({
      type: `tool-${TOOL_AGENT}`,
      state: "output-available",
      input: { description: "analyze code" },
      output: { status: "completed" },
    });
    expect(getAnimationText(part, "agent")).toBe(
      "Agent completed: analyze code",
    );
  });

  it("shows agent completed without summary", () => {
    const part = makePart({
      type: `tool-${TOOL_AGENT}`,
      state: "output-available",
      output: {},
    });
    expect(getAnimationText(part, "agent")).toBe("Agent completed");
  });

  it("shows error text for web search failure", () => {
    const part = makePart({
      type: "tool-WebSearch",
      state: "output-error",
    });
    expect(getAnimationText(part, "web")).toBe("Search failed");
  });

  it("shows error text for web fetch failure", () => {
    const part = makePart({
      type: "tool-web_fetch",
      state: "output-error",
    });
    expect(getAnimationText(part, "web")).toBe("Fetch failed");
  });

  it("shows error text for browser failure", () => {
    const part = makePart({
      type: "tool-browser_navigate",
      state: "output-error",
    });
    expect(getAnimationText(part, "browser")).toBe("Browser action failed");
  });

  it("shows fallback text for unknown state", () => {
    const part = makePart({
      type: "tool-custom_tool",
      state: "unknown-state" as any,
    });
    expect(getAnimationText(part, "other")).toBe("Running Custom tool\u2026");
  });
});
