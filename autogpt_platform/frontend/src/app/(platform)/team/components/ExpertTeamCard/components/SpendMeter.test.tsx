import { render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, test } from "vitest";
import { SpendMeter } from "./SpendMeter";

describe("SpendMeter", () => {
  test("exposes a named progress bar with the real figures in dollars", () => {
    render(<SpendMeter spent={1200} budget={5000} />);

    const meter = screen.getByRole("progressbar", { name: "Weekly spend" });
    expect(meter.getAttribute("aria-valuenow")).toBe("1200");
    expect(meter.getAttribute("aria-valuemax")).toBe("5000");
    expect(meter.getAttribute("aria-valuetext")).toBe(
      "$12 of $50 spent this week",
    );
  });

  test("clamps the value and announces the over-budget state", () => {
    render(<SpendMeter spent={8025} budget={5000} />);

    const meter = screen.getByRole("progressbar", { name: "Weekly spend" });
    expect(meter.getAttribute("aria-valuenow")).toBe("5000");
    expect(meter.getAttribute("aria-valuetext")).toBe(
      "$80.25 of $50 spent this week (over budget)",
    );
  });
});
