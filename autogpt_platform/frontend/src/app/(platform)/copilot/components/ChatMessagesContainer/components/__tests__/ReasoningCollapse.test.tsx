import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { ReasoningCollapse } from "../ReasoningCollapse";

vi.mock("@phosphor-icons/react", () => ({
  CaretRightIcon: ({ className }: { className?: string }) => (
    <span data-testid="caret" className={className} />
  ),
  LightbulbIcon: () => <span data-testid="lightbulb" />,
}));

describe("ReasoningCollapse", () => {
  afterEach(() => {
    cleanup();
  });

  it("starts collapsed with 'Show reasoning' and no children in the DOM", () => {
    render(
      <ReasoningCollapse>
        <div data-testid="reasoning-body">secret thoughts</div>
      </ReasoningCollapse>,
    );

    expect(screen.getByText("Show reasoning")).toBeDefined();
    expect(screen.queryByTestId("reasoning-body")).toBeNull();
    const trigger = screen.getByRole("button");
    expect(trigger.getAttribute("aria-expanded")).toBe("false");
  });

  it("expands on click: toggles label and reveals children", () => {
    render(
      <ReasoningCollapse>
        <div data-testid="reasoning-body">secret thoughts</div>
      </ReasoningCollapse>,
    );

    const trigger = screen.getByRole("button");
    fireEvent.click(trigger);

    expect(screen.getByText("Hide reasoning")).toBeDefined();
    expect(screen.getByTestId("reasoning-body")).toBeDefined();
    expect(trigger.getAttribute("aria-expanded")).toBe("true");
  });

  it("collapses again on a second click", () => {
    render(
      <ReasoningCollapse>
        <div data-testid="reasoning-body">secret thoughts</div>
      </ReasoningCollapse>,
    );

    const trigger = screen.getByRole("button");
    fireEvent.click(trigger);
    fireEvent.click(trigger);

    expect(screen.getByText("Show reasoning")).toBeDefined();
    expect(screen.queryByTestId("reasoning-body")).toBeNull();
    expect(trigger.getAttribute("aria-expanded")).toBe("false");
  });

  it("applies the rotate-90 caret class only when expanded", () => {
    render(
      <ReasoningCollapse>
        <div>body</div>
      </ReasoningCollapse>,
    );

    const caret = screen.getByTestId("caret");
    expect(caret.className).not.toContain("rotate-90");

    fireEvent.click(screen.getByRole("button"));
    expect(screen.getByTestId("caret").className).toContain("rotate-90");
  });
});
