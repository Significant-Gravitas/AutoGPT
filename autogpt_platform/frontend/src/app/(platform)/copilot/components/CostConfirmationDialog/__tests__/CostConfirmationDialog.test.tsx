import {
  cleanup,
  fireEvent,
  render,
  screen,
} from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { CostConfirmationDialog } from "../CostConfirmationDialog";

afterEach(cleanup);

const estimate = {
  resolved_path: "sdk" as const,
  resolved_model: "claude-opus-4-1",
  estimated_llm_calls: 3,
  estimated_cost_usd: 1.234,
  confirmation_threshold_usd: 1,
  rationale: "3 model calls with an attached file",
};

describe("Cost confirmation dialog", () => {
  it("shows the estimated charge and lets the user continue", () => {
    const onConfirm = vi.fn();
    render(
      <CostConfirmationDialog
        estimate={estimate}
        onConfirm={onConfirm}
        onCancel={vi.fn()}
      />,
    );

    expect(screen.getByText("$1.23")).toBeDefined();
    expect(screen.getByText("Threshold: $1.00")).toBeDefined();
    expect(
      screen.getByText("Path: sdk - Model: claude-opus-4-1"),
    ).toBeDefined();
    expect(screen.getByText("Estimated model calls: 3")).toBeDefined();
    expect(screen.getByText(estimate.rationale)).toBeDefined();
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));
    expect(onConfirm).toHaveBeenCalledOnce();
  });

  it("lets the user cancel without approving the charge", () => {
    const onConfirm = vi.fn();
    const onCancel = vi.fn();
    render(
      <CostConfirmationDialog
        estimate={estimate}
        onConfirm={onConfirm}
        onCancel={onCancel}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(onCancel).toHaveBeenCalledOnce();
    expect(onConfirm).not.toHaveBeenCalled();
  });

  it("renders no confirmation when there is no estimate", () => {
    const { container } = render(
      <CostConfirmationDialog
        estimate={null}
        onConfirm={vi.fn()}
        onCancel={vi.fn()}
      />,
    );

    expect(container.firstChild).toBeNull();
  });
});
