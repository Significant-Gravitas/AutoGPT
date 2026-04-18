import { describe, expect, it } from "vitest";
import {
  getNodeDisplayName,
  serializeGraphForChat,
  buildSeedPrompt,
  MAX_SEED_SUMMARY_CHARS,
  MAX_BACKEND_MESSAGE_CHARS,
  SEED_PROMPT_PREFIX,
} from "../helpers";
import type { CustomNode } from "../../FlowEditor/nodes/CustomNode/CustomNode";
import type { CustomEdge } from "../../FlowEditor/edges/CustomEdge";

function makeNode(id: string, title = "Node"): CustomNode {
  return {
    id,
    data: {
      title,
      description: "",
      hardcodedValues: {},
      inputSchema: {},
      outputSchema: {},
      uiType: 1,
      block_id: id,
      costs: [],
      categories: [],
    },
    type: "custom" as const,
    position: { x: 0, y: 0 },
  } as unknown as CustomNode;
}

function makeEdge(source: string, target: string): CustomEdge {
  return {
    id: `${source}-${target}`,
    source,
    target,
    sourceHandle: "result",
    targetHandle: "text",
    type: "custom",
  } as unknown as CustomEdge;
}

describe("serializeGraphForChat – truncation", () => {
  it("includes a truncation note when node count exceeds MAX_NODES (100)", () => {
    const nodes = Array.from({ length: 101 }, (_, i) => makeNode(`n${i}`));
    const result = serializeGraphForChat(nodes, []);
    expect(result).toContain("1 additional nodes not shown");
  });

  it("does NOT include a truncation note when node count is exactly MAX_NODES", () => {
    const nodes = Array.from({ length: 100 }, (_, i) => makeNode(`n${i}`));
    const result = serializeGraphForChat(nodes, []);
    expect(result).not.toContain("additional nodes not shown");
  });

  it("includes a truncation note when edge count exceeds MAX_EDGES (200)", () => {
    const nodes = [makeNode("src"), makeNode("dst")];
    const edges = Array.from({ length: 201 }, (_, i) =>
      makeEdge(`src${i}`, `dst${i}`),
    );
    const result = serializeGraphForChat(nodes, edges);
    expect(result).toContain("1 additional connections not shown");
  });

  it("does NOT include an edge truncation note when edge count is exactly MAX_EDGES", () => {
    const nodes = [makeNode("src"), makeNode("dst")];
    const edges = Array.from({ length: 200 }, (_, i) =>
      makeEdge(`src${i}`, `dst${i}`),
    );
    const result = serializeGraphForChat(nodes, edges);
    expect(result).not.toContain("additional connections not shown");
  });
});

describe("buildSeedPrompt", () => {
  it("includes SEED_PROMPT_PREFIX, graph context, and user message", () => {
    const result = buildSeedPrompt("graph summary", "user question");
    expect(result).toContain(SEED_PROMPT_PREFIX);
    expect(result).toContain("<graph_context>");
    expect(result).toContain("graph summary");
    expect(result).toContain("User request: user question");
  });

  it("truncates graph summary to MAX_SEED_SUMMARY_CHARS and appends notice", () => {
    const longSummary = "x".repeat(MAX_SEED_SUMMARY_CHARS + 100);
    const result = buildSeedPrompt(longSummary, "hello");
    expect(result).toContain("Graph context truncated");
    // Slice limit is MAX_SEED_SUMMARY_CHARS minus the notice reservation
    // (~100 chars) so the notice fits inside the summary budget without
    // clipping the user message. The retained summary must be close to — but
    // slightly below — MAX_SEED_SUMMARY_CHARS.
    const xRun = "x".repeat(MAX_SEED_SUMMARY_CHARS - 200);
    expect(result.indexOf(xRun)).toBeGreaterThan(-1);
  });

  it("does not truncate summary exactly at MAX_SEED_SUMMARY_CHARS", () => {
    const exactSummary = "y".repeat(MAX_SEED_SUMMARY_CHARS);
    const result = buildSeedPrompt(exactSummary, "hi");
    expect(result).not.toContain("Graph context truncated");
  });

  it("caps total output at MAX_BACKEND_MESSAGE_CHARS", () => {
    const hugeSummary = "s".repeat(MAX_SEED_SUMMARY_CHARS + 1);
    const shortUserMsg = "hello";
    const result = buildSeedPrompt(hugeSummary, shortUserMsg);
    expect(result.length).toBeLessThanOrEqual(MAX_BACKEND_MESSAGE_CHARS);
  });

  it("preserves user message when summary + overhead would overflow the limit", () => {
    const hugeSummary = "s".repeat(MAX_SEED_SUMMARY_CHARS);
    // User message that, combined with fixed overhead, fits within the limit
    const userMsg = "u".repeat(1000);
    const result = buildSeedPrompt(hugeSummary, userMsg);
    expect(result.length).toBeLessThanOrEqual(MAX_BACKEND_MESSAGE_CHARS);
    expect(result).toContain(userMsg);
  });

  it("omits graph context entirely when user message alone fills the limit", () => {
    const hugeSummary = "s".repeat(MAX_SEED_SUMMARY_CHARS);
    // User message that leaves no room for any graph context
    const hugeUserMsg = "u".repeat(MAX_BACKEND_MESSAGE_CHARS - 500);
    const result = buildSeedPrompt(hugeSummary, hugeUserMsg);
    expect(result.length).toBeLessThanOrEqual(MAX_BACKEND_MESSAGE_CHARS);
    // Graph context should not appear since there is no room for it
    expect(result).not.toContain("sssss");
  });

  it("preserves a short summary and user message without truncation", () => {
    const result = buildSeedPrompt("tiny graph", "short question");
    expect(result.length).toBeLessThan(MAX_BACKEND_MESSAGE_CHARS);
    expect(result).toContain("tiny graph");
    expect(result).toContain("short question");
  });

  it("does not silently truncate the user message when graph summary is capped", () => {
    // Craft inputs so the summary-budget path is exercised AND the final output
    // lands just under MAX_BACKEND_MESSAGE_CHARS. If the truncation notice
    // isn't subtracted from the slice limit, the safety `.slice()` at the end
    // trims the tail of `userMsg`, which must never happen.
    const hugeSummary = "s".repeat(MAX_BACKEND_MESSAGE_CHARS);
    const userMsg = "USERMSG_TAIL_MARKER";
    const result = buildSeedPrompt(hugeSummary, userMsg);
    expect(result.length).toBeLessThanOrEqual(MAX_BACKEND_MESSAGE_CHARS);
    // The user message (including its tail) must appear intact.
    expect(result).toContain(userMsg);
    expect(result).toContain("Graph context truncated");
  });
});

describe("serializeGraphForChat – XML injection prevention", () => {
  it("escapes < and > in node names before embedding in prompt", () => {
    const nodes = [
      {
        id: "1",
        data: {
          title: "<script>alert(1)</script>",
          description: "",
          hardcodedValues: {},
          inputSchema: {},
          outputSchema: {},
          uiType: 1,
          block_id: "b1",
          costs: [],
          categories: [],
        },
        type: "custom" as const,
        position: { x: 0, y: 0 },
      },
    ] as unknown as CustomNode[];

    const result = serializeGraphForChat(nodes, []);
    expect(result).not.toContain("<script>");
    expect(result).toContain("&lt;script&gt;");
  });

  it("escapes < and > in node descriptions", () => {
    const nodes = [
      {
        id: "1",
        data: {
          title: "Node",
          description: "desc with <injection>",
          hardcodedValues: {},
          inputSchema: {},
          outputSchema: {},
          uiType: 1,
          block_id: "b1",
          costs: [],
          categories: [],
        },
        type: "custom" as const,
        position: { x: 0, y: 0 },
      },
    ] as unknown as CustomNode[];

    const result = serializeGraphForChat(nodes, []);
    expect(result).not.toContain("<injection>");
    expect(result).toContain("&lt;injection&gt;");
  });
});

function makeAgentNode(
  overrides: Partial<CustomNode["data"]> = {},
): CustomNode {
  return {
    id: "node-1",
    data: {
      title: "AgentExecutorBlock",
      description: "",
      hardcodedValues: {},
      inputSchema: {},
      outputSchema: {},
      uiType: "agent",
      block_id: "b1",
      costs: [],
      categories: [],
      ...overrides,
    },
    type: "custom" as const,
    position: { x: 0, y: 0 },
  } as unknown as CustomNode;
}

describe("getNodeDisplayName", () => {
  it("returns fallback when node is undefined", () => {
    expect(getNodeDisplayName(undefined, "fallback-id")).toBe("fallback-id");
  });

  it("returns customized_name when set", () => {
    const node = makeAgentNode({
      metadata: { customized_name: "My Agent" } as any,
    });
    expect(getNodeDisplayName(node, "fallback")).toBe("My Agent");
  });

  it("returns agent_name with version via getNodeDisplayTitle delegation", () => {
    const node = makeAgentNode({
      hardcodedValues: { agent_name: "Researcher", graph_version: 3 },
    });
    expect(getNodeDisplayName(node, "fallback")).toBe("Researcher v3");
  });

  it("returns block title when no custom or agent name", () => {
    const node = makeAgentNode({ title: "SomeBlock" });
    expect(getNodeDisplayName(node, "fallback")).toBe("SomeBlock");
  });

  it("returns fallback when title is empty", () => {
    const node = makeAgentNode({ title: "" });
    expect(getNodeDisplayName(node, "fallback")).toBe("fallback");
  });
});
