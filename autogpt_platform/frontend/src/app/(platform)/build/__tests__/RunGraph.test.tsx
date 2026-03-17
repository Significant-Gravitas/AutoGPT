import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { screen, fireEvent, cleanup } from "@testing-library/react";
import { render } from "@/tests/integrations/test-utils";
import React from "react";
import { useGraphStore } from "../stores/graphStore";

vi.mock(
  "@/app/(platform)/build/components/BuilderActions/components/RunGraph/useRunGraph",
  () => ({
    useRunGraph: vi.fn(),
  }),
);

vi.mock(
  "@/app/(platform)/build/components/BuilderActions/components/RunInputDialog/RunInputDialog",
  () => ({
    RunInputDialog: ({ isOpen }: { isOpen: boolean }) =>
      isOpen ? <div data-testid="run-input-dialog">Dialog</div> : null,
  }),
);

// Must import after mocks
import { useRunGraph } from "../components/BuilderActions/components/RunGraph/useRunGraph";
import { RunGraph } from "../components/BuilderActions/components/RunGraph/RunGraph";

const mockUseRunGraph = vi.mocked(useRunGraph);

function createMockReturnValue(
  overrides: Partial<ReturnType<typeof useRunGraph>> = {},
) {
  return {
    handleRunGraph: vi.fn(),
    handleStopGraph: vi.fn(),
    openRunInputDialog: false,
    setOpenRunInputDialog: vi.fn(),
    isExecutingGraph: false,
    isTerminatingGraph: false,
    isSaving: false,
    ...overrides,
  };
}

// RunGraph uses Tooltip which requires TooltipProvider
import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";

function renderRunGraph(flowID: string | null = "test-flow-id") {
  return render(
    <TooltipProvider>
      <RunGraph flowID={flowID} />
    </TooltipProvider>,
  );
}

describe("RunGraph", () => {
  beforeEach(() => {
    cleanup();
    mockUseRunGraph.mockReturnValue(createMockReturnValue());
    useGraphStore.setState({ isGraphRunning: false });
  });

  afterEach(() => {
    cleanup();
  });

  it("renders an enabled button when flowID is provided", () => {
    renderRunGraph("test-flow-id");
    const button = screen.getByRole("button");
    expect((button as HTMLButtonElement).disabled).toBe(false);
  });

  it("renders a disabled button when flowID is null", () => {
    renderRunGraph(null);
    const button = screen.getByRole("button");
    expect((button as HTMLButtonElement).disabled).toBe(true);
  });

  it("disables the button when isExecutingGraph is true", () => {
    mockUseRunGraph.mockReturnValue(
      createMockReturnValue({ isExecutingGraph: true }),
    );
    renderRunGraph();
    expect((screen.getByRole("button") as HTMLButtonElement).disabled).toBe(
      true,
    );
  });

  it("disables the button when isTerminatingGraph is true", () => {
    mockUseRunGraph.mockReturnValue(
      createMockReturnValue({ isTerminatingGraph: true }),
    );
    renderRunGraph();
    expect((screen.getByRole("button") as HTMLButtonElement).disabled).toBe(
      true,
    );
  });

  it("disables the button when isSaving is true", () => {
    mockUseRunGraph.mockReturnValue(createMockReturnValue({ isSaving: true }));
    renderRunGraph();
    expect((screen.getByRole("button") as HTMLButtonElement).disabled).toBe(
      true,
    );
  });

  it("uses data-id run-graph-button when not running", () => {
    renderRunGraph();
    const button = screen.getByRole("button");
    expect(button.getAttribute("data-id")).toBe("run-graph-button");
  });

  it("uses data-id stop-graph-button when running", () => {
    useGraphStore.setState({ isGraphRunning: true });
    renderRunGraph();
    const button = screen.getByRole("button");
    expect(button.getAttribute("data-id")).toBe("stop-graph-button");
  });

  it("calls handleRunGraph when clicked and graph is not running", () => {
    const handleRunGraph = vi.fn();
    mockUseRunGraph.mockReturnValue(createMockReturnValue({ handleRunGraph }));
    renderRunGraph();
    fireEvent.click(screen.getByRole("button"));
    expect(handleRunGraph).toHaveBeenCalledOnce();
  });

  it("calls handleStopGraph when clicked and graph is running", () => {
    const handleStopGraph = vi.fn();
    mockUseRunGraph.mockReturnValue(createMockReturnValue({ handleStopGraph }));
    useGraphStore.setState({ isGraphRunning: true });
    renderRunGraph();
    fireEvent.click(screen.getByRole("button"));
    expect(handleStopGraph).toHaveBeenCalledOnce();
  });

  it("renders RunInputDialog hidden by default", () => {
    renderRunGraph();
    expect(screen.queryByTestId("run-input-dialog")).toBeNull();
  });

  it("renders RunInputDialog when openRunInputDialog is true", () => {
    mockUseRunGraph.mockReturnValue(
      createMockReturnValue({ openRunInputDialog: true }),
    );
    renderRunGraph();
    expect(screen.getByTestId("run-input-dialog")).toBeDefined();
  });
});
