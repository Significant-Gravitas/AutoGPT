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
    <span data-testid="reasoning-icon" className={className} />
  ),
}));

describe("ReasoningCollapse", () => {
  afterEach(() => {
    cleanup();
  });

  it("renders a collapsed 'Reasoning' trigger with a bulb icon", () => {
    render(
      <ReasoningCollapse>
        <span data-testid="reasoning-body">secret thoughts</span>
      </ReasoningCollapse>,
    );

    expect(screen.getByRole("button", { name: /reasoning/i })).toBeDefined();
    expect(screen.getByTestId("reasoning-icon")).toBeDefined();
    // Collapsed by default: content should not be visible.
    expect(screen.queryByTestId("reasoning-body")).toBeNull();
  });

  it("applies a pulse animation class to the trigger", () => {
    render(
      <ReasoningCollapse>
        <span>thought</span>
      </ReasoningCollapse>,
    );

    const trigger = screen.getByRole("button", { name: /reasoning/i });
    expect(trigger.className).toContain("animate-pulse");
  });

  it("renders no chevron / extra svg inside the trigger", () => {
    render(
      <ReasoningCollapse>
        <span>thought</span>
      </ReasoningCollapse>,
    );

    const trigger = screen.getByRole("button", { name: /reasoning/i });
    // The bulb icon is mocked to a <span>; the previous design appended a
    // <ChevronDown /> svg. After the redesign no svg should remain.
    expect(trigger.querySelector("svg")).toBeNull();
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
