import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ReasoningCollapse } from "../ReasoningCollapse";

vi.mock("@phosphor-icons/react", () => ({
  LightbulbIcon: ({ className }: { className?: string }) => (
    <span data-testid="reasoning-bulb-icon" className={className} />
  ),
  CaretRightIcon: ({ className }: { className?: string }) => (
    <span data-testid="reasoning-caret-icon" className={className} />
  ),
}));

describe("ReasoningCollapse", () => {
  afterEach(() => {
    cleanup();
  });

  it("renders a collapsed 'Reasoning' trigger", () => {
    render(
      <ReasoningCollapse>
        <span data-testid="reasoning-body">secret thoughts</span>
      </ReasoningCollapse>,
    );

    expect(screen.getByRole("button", { name: /reasoning/i })).toBeDefined();
    // Collapsed by default: content should not be visible.
    expect(screen.queryByTestId("reasoning-body")).toBeNull();
  });

  it("shows the pulsing bulb icon while actively reasoning", () => {
    render(
      <ReasoningCollapse isActive>
        <span>thought</span>
      </ReasoningCollapse>,
    );

    expect(screen.getByTestId("reasoning-bulb-icon")).toBeDefined();
    expect(screen.queryByTestId("reasoning-caret-icon")).toBeNull();
  });

  it("swaps the bulb for a caret once reasoning has ended", () => {
    render(
      <ReasoningCollapse>
        <span>thought</span>
      </ReasoningCollapse>,
    );

    expect(screen.getByTestId("reasoning-caret-icon")).toBeDefined();
    expect(screen.queryByTestId("reasoning-bulb-icon")).toBeNull();
  });

  it("applies a pulse animation class to the trigger ONLY while active", () => {
    const inactive = render(
      <ReasoningCollapse>
        <span>thought</span>
      </ReasoningCollapse>,
    );
    expect(
      inactive.getByRole("button", { name: /reasoning/i }).className,
    ).not.toContain("animate-pulse");
    inactive.unmount();

    const active = render(
      <ReasoningCollapse isActive>
        <span>thought</span>
      </ReasoningCollapse>,
    );
    expect(
      active.getByRole("button", { name: /reasoning/i }).className,
    ).toContain("animate-pulse");
  });

  it("reveals the reasoning body when the trigger is clicked", async () => {
    render(
      <ReasoningCollapse>
        <span data-testid="reasoning-body">secret thoughts</span>
      </ReasoningCollapse>,
    );

    fireEvent.click(screen.getByRole("button", { name: /reasoning/i }));

    expect(await screen.findByTestId("reasoning-body")).toBeDefined();
  });

  it("collapses back when the inner Collapse button is clicked", async () => {
    render(
      <ReasoningCollapse>
        <span data-testid="reasoning-body">secret thoughts</span>
      </ReasoningCollapse>,
    );

    fireEvent.click(screen.getByRole("button", { name: /reasoning/i }));
    expect(await screen.findByTestId("reasoning-body")).toBeDefined();

    fireEvent.click(screen.getByRole("button", { name: /collapse/i }));
    await waitFor(() => {
      expect(screen.queryByTestId("reasoning-body")).toBeNull();
    });
  });
});
