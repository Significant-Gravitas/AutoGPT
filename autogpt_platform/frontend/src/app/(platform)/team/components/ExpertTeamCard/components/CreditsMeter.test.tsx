import { render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, test } from "vitest";
import { CreditsMeter } from "./CreditsMeter";

describe("CreditsMeter", () => {
  test("exposes a named progress bar with the real figures", () => {
    render(<CreditsMeter spent={12} budget={50} />);

    const meter = screen.getByRole("progressbar", {
      name: "Weekly credit usage",
    });
    expect(meter.getAttribute("aria-valuenow")).toBe("12");
    expect(meter.getAttribute("aria-valuemax")).toBe("50");
    expect(meter.getAttribute("aria-valuetext")).toBe(
      "12 of 50 credits used this week",
    );
  });

  test("clamps the value and announces the over-budget state", () => {
    render(<CreditsMeter spent={80} budget={50} />);

    const meter = screen.getByRole("progressbar", {
      name: "Weekly credit usage",
    });
    expect(meter.getAttribute("aria-valuenow")).toBe("50");
    expect(meter.getAttribute("aria-valuetext")).toBe(
      "80 of 50 credits used this week (over budget)",
    );
  });
});
