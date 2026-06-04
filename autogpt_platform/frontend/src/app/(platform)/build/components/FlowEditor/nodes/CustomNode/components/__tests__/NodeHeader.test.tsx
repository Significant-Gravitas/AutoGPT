import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@/tests/integrations/test-utils";
import { NodeHeader } from "../NodeHeader";
import { CustomNodeData } from "../../CustomNode";
import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";

vi.mock("../NodeCost", () => ({
  NodeCost: () => <div data-testid="node-cost" />,
}));

vi.mock("../NodeContextMenu", () => ({
  NodeContextMenu: () => <div data-testid="node-context-menu" />,
}));

vi.mock("../NodeBadges", () => ({
  NodeBadges: () => <div data-testid="node-badges" />,
}));

function makeData(overrides: Partial<CustomNodeData> = {}): CustomNodeData {
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

describe("NodeHeader", () => {
  const mockUpdateNodeData = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    useNodeStore.setState({ updateNodeData: mockUpdateNodeData } as any);
  });

  it("renders beautified generic block title", () => {
    render(<NodeHeader data={makeData()} nodeId="abc-123" />);
    expect(screen.getByText("Agent Executor")).toBeTruthy();
  });

  it("renders agent name with version from hardcodedValues", () => {
    const data = makeData({
      hardcodedValues: { agent_name: "Researcher", graph_version: 2 },
    });
    render(<NodeHeader data={data} nodeId="abc-123" />);
    expect(screen.getByText("Researcher v2")).toBeTruthy();
  });

  it("renders customized_name over agent name", () => {
    const data = makeData({
      metadata: { customized_name: "My Custom Node" } as any,
      hardcodedValues: { agent_name: "Researcher", graph_version: 1 },
    });
    render(<NodeHeader data={data} nodeId="abc-123" />);
    expect(screen.getByText("My Custom Node")).toBeTruthy();
  });

  it("shows node ID prefix", () => {
    render(<NodeHeader data={makeData()} nodeId="abc-123" />);
    expect(screen.getByText("#abc")).toBeTruthy();
  });

  it("enters edit mode on double-click and saves on blur", () => {
    render(<NodeHeader data={makeData()} nodeId="node-1" />);
    const titleEl = screen.getByText("Agent Executor");
    fireEvent.doubleClick(titleEl);

    const input = screen.getByDisplayValue("AgentExecutorBlock");
    fireEvent.change(input, { target: { value: "New Name" } });
    fireEvent.blur(input);

    expect(mockUpdateNodeData).toHaveBeenCalledWith("node-1", {
      metadata: { customized_name: "New Name" },
    });
  });

  it("does not save when title is unchanged on blur", () => {
    const data = makeData({
      hardcodedValues: { agent_name: "Researcher", graph_version: 2 },
    });
    render(<NodeHeader data={data} nodeId="node-1" />);
    const titleEl = screen.getByText("Researcher v2");
    fireEvent.doubleClick(titleEl);

    const input = screen.getByDisplayValue("Researcher v2");
    fireEvent.blur(input);

    expect(mockUpdateNodeData).not.toHaveBeenCalled();
  });

  it("saves on Enter key", () => {
    render(<NodeHeader data={makeData()} nodeId="node-1" />);
    fireEvent.doubleClick(screen.getByText("Agent Executor"));

    const input = screen.getByDisplayValue("AgentExecutorBlock");
    fireEvent.change(input, { target: { value: "Renamed" } });
    fireEvent.keyDown(input, { key: "Enter" });

    expect(mockUpdateNodeData).toHaveBeenCalledWith("node-1", {
      metadata: { customized_name: "Renamed" },
    });
  });

  it("cancels edit on Escape key", () => {
    render(<NodeHeader data={makeData()} nodeId="node-1" />);
    fireEvent.doubleClick(screen.getByText("Agent Executor"));

    const input = screen.getByDisplayValue("AgentExecutorBlock");
    fireEvent.change(input, { target: { value: "Changed" } });
    fireEvent.keyDown(input, { key: "Escape" });

    expect(mockUpdateNodeData).not.toHaveBeenCalled();
    expect(screen.getByText("Agent Executor")).toBeTruthy();
  });
});
