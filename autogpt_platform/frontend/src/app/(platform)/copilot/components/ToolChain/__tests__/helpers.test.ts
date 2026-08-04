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

  it("keeps custom-card tools outside chains with the UI predicate", () => {
    const customTool = toolPart("decompose_goal");
    const genericTool = toolPart("web_search");

    expect(
      buildChainSegments([customTool, genericTool], isChainableToolPart),
    ).toEqual([
      { kind: "part", part: customTool, index: 0 },
      { kind: "chain", parts: [genericTool], index: 1 },
    ]);
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
});

describe("getChainHeading", () => {
  it("uses the last running row while streaming", () => {
    const rows: ChainRow[] = [
      { key: "1", category: "reasoning", text: "Thought", state: "done" },
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
});
