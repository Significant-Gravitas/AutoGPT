import { cleanup, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { BlockUIType } from "../components/types";
import { OutputHandler } from "../components/FlowEditor/nodes/OutputHandler";

const mockEdges = vi.hoisted(() => [
  { source: "test-node", sourceHandle: "dynamic_#_field" },
  { source: "test-node", sourceHandle: "nested_#___proto___#_polluted" },
]);

vi.mock("@/app/(platform)/build/stores/edgeStore", () => ({
  useEdgeStore: () => ({
    isOutputConnected: () => false,
    edges: mockEdges,
  }),
}));

vi.mock(
  "@/app/(platform)/build/components/FlowEditor/nodes/useBrokenOutputs",
  () => ({
    useBrokenOutputs: () => new Set(),
  }),
);

vi.mock(
  "@/app/(platform)/build/components/FlowEditor/handlers/NodeHandle",
  () => ({
    OutputNodeHandle: ({ field_name }: { field_name: string }) => (
      <div data-testid="output-node-handle" data-field-name={field_name} />
    ),
  }),
);

describe("OutputHandler", () => {
  beforeEach(() => {
    cleanup();
  });

  it("materializes dynamic output handles and rejects reserved paths", () => {
    render(
      <OutputHandler
        outputSchema={{ type: "object", properties: {} }}
        nodeId="test-node"
        uiType={BlockUIType.STANDARD}
      />,
    );

    expect(screen.getByText("dynamic")).toBeDefined();
    expect(screen.getByText("field")).toBeDefined();

    const handleIds = screen
      .getAllByTestId("output-node-handle")
      .map((handle) => handle.getAttribute("data-field-name"));
    expect(handleIds).toContain("dynamic_#_field");

    expect(screen.queryByText("nested")).toBeNull();
    expect(screen.queryByText("__proto__")).toBeNull();
    expect(screen.queryByText("polluted")).toBeNull();
    expect(handleIds).not.toContain("nested_#___proto___#_polluted");
  });
});
