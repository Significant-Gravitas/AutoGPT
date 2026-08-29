import { describe, it, expect } from "vitest";
import { getNodeDisplayTitle, formatNodeDisplayTitle } from "../helpers";
import { CustomNodeData } from "../CustomNode";

function makeNodeData(overrides: Partial<CustomNodeData> = {}): CustomNodeData {
  return {
    title: "AgentExecutorBlock",
    description: "",
    hardcodedValues: {},
    inputSchema: {},
    outputSchema: {},
    uiType: "agent",
    block_id: "block-1",
    costs: [],
    categories: [],
    ...overrides,
  } as CustomNodeData;
}

describe("getNodeDisplayTitle", () => {
  it("returns customized_name when set (tier 1)", () => {
    const data = makeNodeData({
      metadata: { customized_name: "My Custom Agent" } as any,
      hardcodedValues: { agent_name: "Researcher", graph_version: 2 },
    });
    expect(getNodeDisplayTitle(data)).toBe("My Custom Agent");
  });

  it("returns agent_name with version when no customized_name (tier 2)", () => {
    const data = makeNodeData({
      hardcodedValues: { agent_name: "Researcher", graph_version: 2 },
    });
    expect(getNodeDisplayTitle(data)).toBe("Researcher v2");
  });

  it("returns agent_name without version when graph_version is undefined (tier 2)", () => {
    const data = makeNodeData({
      hardcodedValues: { agent_name: "Researcher" },
    });
    expect(getNodeDisplayTitle(data)).toBe("Researcher");
  });

  it("returns agent_name with version 0 (tier 2)", () => {
    const data = makeNodeData({
      hardcodedValues: { agent_name: "Researcher", graph_version: 0 },
    });
    expect(getNodeDisplayTitle(data)).toBe("Researcher v0");
  });

  it("returns generic block title when no custom or agent name (tier 3)", () => {
    const data = makeNodeData({ title: "AgentExecutorBlock" });
    expect(getNodeDisplayTitle(data)).toBe("AgentExecutorBlock");
  });

  it("prioritizes customized_name over agent_name", () => {
    const data = makeNodeData({
      metadata: { customized_name: "Renamed" } as any,
      hardcodedValues: { agent_name: "Original Agent", graph_version: 1 },
    });
    expect(getNodeDisplayTitle(data)).toBe("Renamed");
  });
});

describe("formatNodeDisplayTitle", () => {
  it("returns custom name as-is without beautifying", () => {
    const data = makeNodeData({
      metadata: { customized_name: "my_custom_name" } as any,
    });
    expect(formatNodeDisplayTitle(data)).toBe("my_custom_name");
  });

  it("returns agent name as-is without beautifying", () => {
    const data = makeNodeData({
      hardcodedValues: { agent_name: "Blockchain Agent", graph_version: 1 },
    });
    expect(formatNodeDisplayTitle(data)).toBe("Blockchain Agent v1");
  });

  it("beautifies generic block title and strips Block suffix", () => {
    const data = makeNodeData({ title: "AgentExecutorBlock" });
    const result = formatNodeDisplayTitle(data);
    expect(result).not.toContain("Block");
    expect(result).toBe("Agent Executor");
  });

  it("does not corrupt agent names containing 'Block'", () => {
    const data = makeNodeData({
      hardcodedValues: { agent_name: "Blockchain Agent", graph_version: 2 },
    });
    expect(formatNodeDisplayTitle(data)).toBe("Blockchain Agent v2");
  });
});
