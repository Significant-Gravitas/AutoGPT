import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { ShimmerText } from "../ShimmerText";
import { SwapText } from "../SwapText";

describe("ShimmerText", () => {
  afterEach(cleanup);

  it("renders the text and duplicates it for the shimmer sweep", () => {
    render(<ShimmerText text="Searching the web…" className="extra" />);

    const span = screen.getByText("Searching the web…");
    expect(span.getAttribute("data-text")).toBe("Searching the web…");
    expect(span.className).toContain("extra");
  });

  it("renders without an extra class", () => {
    render(<ShimmerText text="Thinking…" />);

    expect(screen.getByText("Thinking…").getAttribute("data-text")).toBe(
      "Thinking…",
    );
  });
});

describe("SwapText", () => {
  afterEach(cleanup);

  it("renders plain text when not shimmering", () => {
    render(<SwapText text="Updated tasks" className="text-sm" />);

    const label = screen.getByText("Updated tasks");
    expect(label).toBeDefined();
    expect(label.getAttribute("data-text")).toBeNull();
  });

  it("wraps the label in shimmer while running", () => {
    render(<SwapText text="Running command…" shimmer />);

    expect(screen.getByText("Running command…").getAttribute("data-text")).toBe(
      "Running command…",
    );
  });

  it("swaps to the new label when the text changes", () => {
    const { rerender } = render(<SwapText text="Searching…" />);
    expect(screen.getByText("Searching…")).toBeDefined();

    rerender(<SwapText text="Searched the web" />);

    expect(screen.getByText("Searched the web")).toBeDefined();
  });
});
