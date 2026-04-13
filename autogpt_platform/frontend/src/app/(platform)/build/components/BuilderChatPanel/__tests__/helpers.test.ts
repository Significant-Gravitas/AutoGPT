import { describe, expect, it } from "vitest";
import { serializeGraphForChat } from "../helpers";
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
