import { describe, expect, it } from "vitest";
import {
  isChainableToolPart,
  type MessagePart,
} from "../../ChatMessagesContainer/helpers";
import {
  buildChainSegments,
  getChainHeading,
  isChainPart,
  isLiftedSetupRow,
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
  it("groups consecutive tool and reasoning parts around long text", () => {
    // Long text is a real answer, not progress narration — it splits chains.
    const parts = [
      { type: "step-start" } as MessagePart,
      { type: "reasoning", text: "Plan", state: "done" } as MessagePart,
      toolPart("web_search"),
      textPart("Result. ".repeat(30)),
      toolPart("web_fetch"),
    ];

    expect(buildChainSegments(parts)).toEqual([
      { kind: "chain", parts: [parts[1], parts[2]], index: 1 },
      { kind: "part", part: parts[3], index: 3 },
      { kind: "chain", parts: [parts[4]], index: 4 },
    ]);
  });

  it("folds short narration between tool calls into the chain", () => {
    const parts = [
      toolPart("web_search"),
      textPart("Now fetching the page."),
      toolPart("web_fetch"),
    ];

    expect(buildChainSegments(parts)).toEqual([
      { kind: "chain", parts, index: 0 },
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
  it("turns short text into a narration row and skips long or empty text", () => {
    const narration = toChainRow(textPart("Now searching for hotels."), 3);
    expect(narration).toEqual({
      key: "narration-3",
      category: "narration",
      text: "Now searching for hotels.",
      state: "done",
    });

    expect(toChainRow(textPart("word ".repeat(60)), 0)).toBeNull();
    expect(toChainRow(textPart("   "), 0)).toBeNull();
    expect(toChainRow({ type: "step-start" } as MessagePart, 0)).toBeNull();
  });

  it("maps streaming and settled reasoning parts", () => {
    const streaming = toChainRow(
      {
        type: "reasoning",
        text: "Weighing options",
        state: "streaming",
      } as MessagePart,
      1,
    );
    expect(streaming).toMatchObject({
      key: "reasoning-1",
      category: "reasoning",
      text: "Thinking…",
      state: "running",
      reasoningText: "Weighing options",
    });

    const settled = toChainRow(
      {
        type: "reasoning",
        text: "Weighed options",
        state: "done",
      } as MessagePart,
      2,
    );
    expect(settled).toMatchObject({ text: "Thought", state: "done" });
  });

  it("marks failed tool calls with their error detail", () => {
    const row = toChainRow(
      {
        type: "tool-web_search",
        state: "output-error",
        toolCallId: "call-err",
        input: { query: "hotels" },
        errorText: "Rate limited",
      } as MessagePart,
      0,
    );

    expect(row).toMatchObject({
      state: "error",
      detail: "Rate limited",
      text: 'Failed while searching the web for "hotels"',
    });
  });

  it("ignores partial input while the input JSON is still streaming", () => {
    const row = toChainRow(
      {
        type: "tool-web_search",
        state: "input-streaming",
        toolCallId: "call-stream",
        input: { query: "par" },
      } as MessagePart,
      0,
    );

    expect(row).toMatchObject({
      state: "running",
      text: "Searching the web for…",
    });
  });

  it("falls back to generic categorization for non-catalog tools", () => {
    const grep = toChainRow(
      toolPart("Grep", { pattern: "TODO" }, "3 matches"),
      0,
    );
    expect(grep).toMatchObject({
      category: "search",
      text: 'Searched for "TODO"',
      state: "done",
    });

    const custom = toChainRow(toolPart("polish_widgets", {}, {}), 0);
    expect(custom).toMatchObject({
      category: "other",
      text: "Polish widgets completed",
    });
  });

  it.each([
    [
      { type: "review_required", block_name: "Send Email" },
      "Review Send Email",
    ],
    [{ type: "review_required" }, "Review this action"],
    [{ type: "suggested_goal" }, "Review the suggested goal"],
    [{ type: "need_login", message: "Log in first" }, "Log in first"],
    [{ type: "need_login" }, "Action required"],
    [{ type: "setup_requirements" }, "Complete setup to continue"],
  ])("labels action-required output %j", (output, expected) => {
    const row = toChainRow(toolPart("run_block", {}, output), 0);

    expect(row?.requiresAction).toBe(true);
    expect(row?.text).toBe(expected);
  });

  it("does not require action for plain successful output", () => {
    const row = toChainRow(toolPart("run_block", {}, { ok: true }), 0);
    expect(row?.requiresAction).toBe(false);
  });
});

describe("isLiftedSetupRow", () => {
  it("lifts only setup_requirements rows that carry setup_info", () => {
    const lifted = toChainRow(
      toolPart(
        "connect_integration",
        {},
        { type: "setup_requirements", setup_info: { agent_name: "GitHub" } },
      ),
      0,
    );
    const notLifted = toChainRow(
      toolPart("run_block", {}, { type: "review_required" }),
      0,
    );

    expect(lifted && isLiftedSetupRow(lifted)).toBe(true);
    expect(notLifted && isLiftedSetupRow(notLifted)).toBe(false);
  });
});

describe("isChainPart", () => {
  it("accepts reasoning and tool parts only", () => {
    expect(isChainPart({ type: "reasoning", text: "" } as MessagePart)).toBe(
      true,
    );
    expect(isChainPart(toolPart("web_search"))).toBe(true);
    expect(isChainPart(textPart("Hello"))).toBe(false);
  });
});

describe("toChainRow provider icons", () => {
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

  it("skips lifted action rows and falls back to the summary", () => {
    const rows: ChainRow[] = [
      { key: "1", category: "web", text: "Searched", state: "done" },
      {
        key: "2",
        category: "question",
        text: "Asked you a question",
        state: "done",
        requiresAction: true,
        lifted: true,
      },
    ];

    expect(getChainHeading(rows, false)).toBe(
      "Searched the web, asked you questions",
    );
  });

  it("falls back to Working… when there is nothing to summarize", () => {
    expect(getChainHeading([], false)).toBe("Working…");
    expect(
      getChainHeading(
        [{ key: "1", category: "narration", text: "On it", state: "done" }],
        false,
      ),
    ).toBe("Working…");
  });

  it("summarizes settled rows while streaming when nothing is running", () => {
    const rows: ChainRow[] = [
      { key: "1", category: "bash", text: "Ran command", state: "done" },
    ];

    expect(getChainHeading(rows, true)).toBe("Ran commands");
  });

  it("caps the settled summary at three distinct categories", () => {
    const rows: ChainRow[] = [
      { key: "1", category: "web", text: "a", state: "done" },
      { key: "2", category: "bash", text: "b", state: "done" },
      { key: "3", category: "todo", text: "c", state: "done" },
      { key: "4", category: "docs", text: "d", state: "done" },
    ];

    expect(getChainHeading(rows, false)).toBe(
      "Searched the web, ran commands, updated tasks",
    );
  });
});

describe("buildChainSegments edge cases", () => {
  it("folds trailing narration optimistically while streaming", () => {
    const parts = [toolPart("web_search"), textPart("Wrapping up.")];

    expect(buildChainSegments(parts, isChainPart, true)).toEqual([
      { kind: "chain", parts, index: 0 },
    ]);
  });

  it("keeps trailing narration out of a settled chain", () => {
    const parts = [toolPart("web_search"), textPart("Here you go.")];

    expect(buildChainSegments(parts, isChainPart, false)).toEqual([
      { kind: "chain", parts: [parts[0]], index: 0 },
      { kind: "part", part: parts[1], index: 1 },
    ]);
  });

  it("looks past narration and step-start when checking for tools ahead", () => {
    const parts = [
      toolPart("web_search"),
      textPart("First note."),
      { type: "step-start" } as MessagePart,
      textPart("Second note."),
      toolPart("web_fetch"),
    ];

    expect(buildChainSegments(parts)).toEqual([
      {
        kind: "chain",
        parts: [parts[0], parts[1], parts[3], parts[4]],
        index: 0,
      },
    ]);
  });

  it("keeps narration out when only a real answer follows", () => {
    const parts = [
      toolPart("web_search"),
      textPart("Almost there."),
      textPart("Answer. ".repeat(40)),
    ];

    expect(buildChainSegments(parts)).toEqual([
      { kind: "chain", parts: [parts[0]], index: 0 },
      { kind: "part", part: parts[1], index: 1 },
      { kind: "part", part: parts[2], index: 2 },
    ]);
  });

  it("emits leading text before any chain as its own part", () => {
    const parts = [textPart("Intro."), toolPart("web_search")];

    expect(buildChainSegments(parts)).toEqual([
      { kind: "part", part: parts[0], index: 0 },
      { kind: "chain", parts: [parts[1]], index: 1 },
    ]);
  });
});
