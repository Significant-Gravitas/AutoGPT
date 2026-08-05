import { describe, expect, it } from "vitest";
import {
  isChainableToolPart,
  type MessagePart,
} from "../../ChatMessagesContainer/helpers";
import {
  buildChainSegments,
  getChainHeading,
  toChainRow,
  type ChainRow,
} from "../helpers";

function textPart(text: string): MessagePart {
  return { type: "text", text } as MessagePart;
}

function toolPart(
  toolName: string,
  input: unknown = {},
  output: unknown = {},
): MessagePart {
  return {
    type: `tool-${toolName}`,
    state: "output-available",
    toolCallId: `call-${toolName}`,
    input,
    output,
  } as MessagePart;
}

describe("buildChainSegments", () => {
  it("groups consecutive tool and reasoning parts around text", () => {
    const parts = [
      { type: "step-start" } as MessagePart,
      { type: "reasoning", text: "Plan", state: "done" } as MessagePart,
      toolPart("web_search"),
      textPart("Result"),
      toolPart("web_fetch"),
    ];

    expect(buildChainSegments(parts)).toEqual([
      { kind: "chain", parts: [parts[1], parts[2]], index: 1 },
      { kind: "part", part: parts[3], index: 3 },
      { kind: "chain", parts: [parts[4]], index: 4 },
    ]);
  });

  it("keeps every tool family in the new chain UI", () => {
    const specializedTool = toolPart("decompose_goal");
    const genericTool = toolPart("web_search");

    expect(
      buildChainSegments([specializedTool, genericTool], isChainableToolPart),
    ).toEqual([
      {
        kind: "chain",
        parts: [specializedTool, genericTool],
        index: 0,
      },
    ]);
  });

  it.each([
    "ask_question",
    "find_block",
    "find_agent",
    "find_library_agent",
    "search_docs",
    "get_doc_page",
    "connect_integration",
    "run_block",
    "continue_run_block",
    "run_mcp_tool",
    "run_agent",
    "schedule_agent",
    "setup_agent_webhook_trigger",
    "decompose_goal",
    "create_agent",
    "edit_agent",
    "view_agent_output",
    "search_feature_requests",
    "create_feature_request",
  ])("routes legacy %s cards through ToolChain", (toolName) => {
    expect(isChainableToolPart(toolPart(toolName))).toBe(true);
  });
});

describe("toChainRow", () => {
  it("normalizes a provider slug from tool output into an icon path", () => {
    const row = toChainRow(
      toolPart("run_block", {}, { provider: "Google Maps" }),
      0,
    );

    expect(row?.providerIconSrc).toBe("/integrations/google_maps.png");
  });

  it("removes unsafe characters from provider icon paths", () => {
    const row = toChainRow(
      toolPart("run_block", {}, { provider: "../../Google?<script>" }),
      0,
    );

    expect(row?.providerIconSrc).toBe("/integrations/googlescript.png");
  });

  it("pins setup requirements open with an action-oriented label", () => {
    const setupRow = toChainRow(
      toolPart(
        "connect_integration",
        { provider: "github" },
        JSON.stringify({
          type: "setup_requirements",
          setup_info: { agent_name: "GitHub" },
        }),
      ),
      0,
    );

    expect(setupRow?.requiresAction).toBe(true);
    expect(setupRow?.text).toBe("Connect GitHub to continue");
  });
});

describe("getChainHeading", () => {
  it("uses the last running row while streaming", () => {
    const rows: ChainRow[] = [
      {
        key: "1",
        category: "reasoning",
        text: "Earlier failure",
        state: "error",
      },
      { key: "2", category: "web", text: "Searching docs", state: "running" },
    ];

    expect(getChainHeading(rows, true)).toBe("Searching docs");
  });

  it("summarizes distinct settled categories", () => {
    const rows: ChainRow[] = [
      { key: "1", category: "web", text: "Search", state: "done" },
      { key: "2", category: "web", text: "Fetch", state: "done" },
      { key: "3", category: "bash", text: "Run", state: "done" },
    ];

    expect(getChainHeading(rows, false)).toBe("Searched the web, ran commands");
  });

  it("surfaces the latest error detail", () => {
    const rows: ChainRow[] = [
      { key: "1", category: "web", text: "Search", state: "done" },
      {
        key: "2",
        category: "bash",
        text: "Failed while running command",
        detail: "Permission denied",
        state: "error",
      },
    ];

    expect(getChainHeading(rows, false)).toBe("Permission denied");
  });

  it("surfaces required actions ahead of a settled summary", () => {
    const rows: ChainRow[] = [
      { key: "1", category: "web", text: "Searched", state: "done" },
      {
        key: "2",
        category: "integration",
        text: "Connect GitHub to continue",
        state: "done",
        requiresAction: true,
      },
    ];

    expect(getChainHeading(rows, false)).toBe("Connect GitHub to continue");
  });
});
