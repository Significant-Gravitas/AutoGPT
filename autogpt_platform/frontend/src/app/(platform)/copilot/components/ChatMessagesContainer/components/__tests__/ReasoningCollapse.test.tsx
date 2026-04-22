import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import { ReasoningCollapse } from "../ReasoningCollapse";

vi.mock("@phosphor-icons/react", () => ({
  ChatCircleDotsIcon: ({ className }: { className?: string }) => (
    <span data-testid="reasoning-icon" className={className} />
  ),
}));

describe("ReasoningCollapse", () => {
  afterEach(() => {
    cleanup();
  });

  it("renders the 'Reasoning:' label with its children inline", () => {
    render(
      <ReasoningCollapse>
        <span data-testid="reasoning-body">secret thoughts</span>
      </ReasoningCollapse>,
    );

    expect(screen.getByText("Reasoning:")).toBeDefined();
    // No longer collapsible — children are always in the DOM.
    expect(screen.getByTestId("reasoning-body")).toBeDefined();
    expect(screen.getByTestId("reasoning-icon")).toBeDefined();
  });

  it("has no toggle button (component is always inline)", () => {
    render(
      <ReasoningCollapse>
        <span>body</span>
      </ReasoningCollapse>,
    );

    expect(screen.queryByRole("button")).toBeNull();
  });
});
