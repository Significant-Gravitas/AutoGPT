import { describe, expect, it } from "vitest";
import { getAnimationText, kindLabel, parseOutput } from "../helpers";

const OUTPUT = {
  type: "capability_list",
  query: "linear issue",
  count: 2,
  service: "linear",
  capabilities: [
    {
      id: "block:1",
      name: "LinearCreateIssueBlock",
      purpose: "Create an issue.",
      kind: "block",
      connected: true,
    },
    {
      id: "mcp:mcp.linear.app",
      name: "Linear",
      purpose: "Linear MCP server.",
      kind: "mcp_server",
      connected: false,
    },
  ],
  fallback: [
    {
      id: "block:2",
      name: "SendWebRequestBlock",
      purpose: "HTTP.",
      kind: "block",
      class: "primitive",
    },
  ],
};

describe("find_capability helpers", () => {
  it("parses capability_list outputs as string or object", () => {
    expect(parseOutput(JSON.stringify(OUTPUT))?.count).toBe(2);
    expect(parseOutput(OUTPUT)?.capabilities).toHaveLength(2);
    expect(parseOutput({ type: "block_list", blocks: [] })).toBeNull();
    expect(parseOutput("")).toBeNull();
  });

  it("labels kinds for the card footer", () => {
    expect(kindLabel(OUTPUT.capabilities[0])).toBe("action");
    expect(kindLabel(OUTPUT.capabilities[1])).toBe("integration");
    expect(kindLabel(OUTPUT.fallback[0])).toBe("building block");
  });

  it("animates the search state with the query", () => {
    const base = {
      type: "tool-find_capability",
      toolCallId: "c",
      input: { query: "linear issue" },
    };
    expect(getAnimationText({ ...base, state: "input-available" })).toBe(
      'Searching capabilities for "linear issue"',
    );
    expect(
      getAnimationText({ ...base, state: "output-available", output: OUTPUT }),
    ).toBe('Found 2 capabilities for "linear issue"');
    expect(getAnimationText({ ...base, state: "output-error" })).toBe(
      'Search failed for "linear issue"',
    );
  });

  it("survives a query that is not a string yet", () => {
    const base = { type: "tool-find_capability", toolCallId: "c" };
    for (const input of [
      { query: 1 },
      { query: null },
      "not an object",
      null,
    ]) {
      expect(
        getAnimationText({ ...base, state: "input-streaming", input }),
      ).toBe("Searching capabilities");
    }
  });
});
