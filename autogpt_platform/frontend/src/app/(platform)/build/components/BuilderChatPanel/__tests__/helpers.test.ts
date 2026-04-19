import { describe, expect, it } from "vitest";
import { getNodeDisplayName, serializeGraphForChat } from "../helpers";
import type { CustomNode } from "../../FlowEditor/nodes/CustomNode/CustomNode";

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

function makeNode(overrides: Partial<CustomNode["data"]> = {}): CustomNode {
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
    const node = makeNode({
      metadata: { customized_name: "My Agent" } as any,
    });
    expect(getNodeDisplayName(node, "fallback")).toBe("My Agent");
  });

  it("returns agent_name with version via getNodeDisplayTitle delegation", () => {
    const node = makeNode({
      hardcodedValues: { agent_name: "Researcher", graph_version: 3 },
    });
    expect(getNodeDisplayName(node, "fallback")).toBe("Researcher v3");
  });

  it("returns block title when no custom or agent name", () => {
    const node = makeNode({ title: "SomeBlock" });
    expect(getNodeDisplayName(node, "fallback")).toBe("SomeBlock");
  });

  it("returns fallback when title is empty", () => {
    const node = makeNode({ title: "" });
    expect(getNodeDisplayName(node, "fallback")).toBe("fallback");
  });
});
