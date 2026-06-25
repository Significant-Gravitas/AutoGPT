import { describe, expect, test, vi } from "vitest";
import { render, screen, fireEvent } from "@/tests/integrations/test-utils";
import { PulseChips } from "../PulseChips";
import type { PulseChipData } from "../types";

function makeChip(overrides: Partial<PulseChipData> = {}): PulseChipData {
  return {
    id: "chip-1",
    agentID: "agent-1",
    name: "Test Agent",
    status: "running",
    priority: "running",
    shortMessage: "Doing work…",
    ...overrides,
  };
}

describe("PulseChips", () => {
  test("renders nothing when chips array is empty", () => {
    const { container } = render(<PulseChips chips={[]} />);
    expect(container.innerHTML).toBe("");
  });

  test("renders chip names and messages", () => {
    const chips = [
      makeChip({ id: "1", name: "Alpha Bot", shortMessage: "Running task A" }),
      makeChip({ id: "2", name: "Beta Bot", shortMessage: "Running task B" }),
    ];

    render(<PulseChips chips={chips} />);

    expect(screen.getByText("Alpha Bot")).toBeDefined();
    expect(screen.getByText("Running task A")).toBeDefined();
    expect(screen.getByText("Beta Bot")).toBeDefined();
    expect(screen.getByText("Running task B")).toBeDefined();
  });

  test("renders section heading and View all link", () => {
    render(<PulseChips chips={[makeChip()]} />);

    expect(screen.getByText("What's happening with your agents")).toBeDefined();
    expect(screen.getByText("View all")).toBeDefined();
  });

  test("shows Completed badge for success priority chips", () => {
    render(
      <PulseChips
        chips={[makeChip({ priority: "success", status: "idle" })]}
      />,
    );

    expect(screen.getByText("Completed")).toBeDefined();
  });

  test("calls onChipClick with generated prompt when Ask is clicked", () => {
    const onChipClick = vi.fn();
    render(
      <PulseChips
        chips={[
          makeChip({
            name: "Error Agent",
            status: "error",
            priority: "error",
          }),
        ]}
        onChipClick={onChipClick}
      />,
    );

    fireEvent.click(screen.getByText("Ask"));

    expect(onChipClick).toHaveBeenCalledWith(
      "What happened with Error Agent? It has an error — can you check?",
    );
  });

  test("generates success prompt for completed chips", () => {
    const onChipClick = vi.fn();
    render(
      <PulseChips
        chips={[
          makeChip({
            name: "Done Agent",
            priority: "success",
            status: "idle",
          }),
        ]}
        onChipClick={onChipClick}
      />,
    );

    fireEvent.click(screen.getByText("Ask"));

    expect(onChipClick).toHaveBeenCalledWith(
      "Done Agent just finished a run — can you summarize what it did?",
    );
  });

  test("renders See link pointing to agent detail page", () => {
    render(<PulseChips chips={[makeChip({ agentID: "agent-xyz" })]} />);

    const seeLink = screen.getByText("See").closest("a");
    expect(seeLink?.getAttribute("href")).toBe("/library/agents/agent-xyz");
  });
});
