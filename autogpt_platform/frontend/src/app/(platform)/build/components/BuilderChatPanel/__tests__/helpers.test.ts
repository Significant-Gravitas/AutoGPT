import { describe, expect, it } from "vitest";
import { serializeGraphForChat } from "../helpers";
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
