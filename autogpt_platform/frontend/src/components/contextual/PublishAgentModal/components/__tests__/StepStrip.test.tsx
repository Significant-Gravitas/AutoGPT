import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { StepStrip } from "../StepStrip";

describe("StepStrip", () => {
  it.each([
    ["select" as const, 0],
    ["info" as const, 1],
    ["review" as const, 2],
  ])("marks step %s as current", (step, expectedIndex) => {
    render(<StepStrip currentStep={step} />);
    const list = screen.getByRole("list", { name: "Publish progress" });
    const items = list.querySelectorAll("li[aria-current='step']");
    expect(items).toHaveLength(1);
    const labels = ["Agent", "Listing", "Review"];
    expect(items[0].textContent).toContain(labels[expectedIndex]);
  });

  it("renders all three steps", () => {
    render(<StepStrip currentStep="select" />);
    expect(screen.getByText("Agent")).toBeDefined();
    expect(screen.getByText("Listing")).toBeDefined();
    expect(screen.getByText("Review")).toBeDefined();
    expect(screen.getByText("Publish agent")).toBeDefined();
  });
});
