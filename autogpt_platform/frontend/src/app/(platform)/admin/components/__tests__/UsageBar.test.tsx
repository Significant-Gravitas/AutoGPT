import { render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, it } from "vitest";
import { UsageBar } from "../UsageBar";

describe("UsageBar", () => {
  it('renders "Unlimited" when the limit is negative (no cap)', () => {
    // A negative limit is the backend's "no cap" sentinel and the self-hosted
    // default, so GET /api/copilot/admin/rate_limit returns -1 for both
    // windows. Formatting it as money read "$-0.00 limit, 0.0% used".
    render(<UsageBar used={1_234_567} limit={-1} />);
    expect(screen.getByText("$1.23 spent")).toBeDefined();
    expect(screen.getByText("Unlimited")).toBeDefined();
    expect(screen.queryByText(/limit$/)).toBeNull();
    expect(screen.queryByText(/% used/)).toBeNull();
  });

  it('renders "No allowance" when the limit is 0 (spending blocked)', () => {
    // 0 does not mean unlimited: the backend blocks every spend at 0, so
    // labelling it "Unlimited" told the admin the opposite of the truth.
    render(<UsageBar used={0} limit={0} />);
    expect(screen.getByText("No allowance")).toBeDefined();
    expect(screen.queryByText("Unlimited")).toBeNull();
  });

  it("renders spent + limit in USD", () => {
    render(<UsageBar used={1_500_000} limit={10_000_000} />);
    expect(screen.getByText("$1.50 spent")).toBeDefined();
    expect(screen.getByText("$10.00 limit")).toBeDefined();
  });

  it("renders the computed percentage", () => {
    render(<UsageBar used={500_000} limit={1_000_000} />);
    expect(screen.getByText("50.0% used")).toBeDefined();
  });

  it("clamps percentage at 100% when over limit", () => {
    render(<UsageBar used={2_000_000} limit={1_000_000} />);
    expect(screen.getByText("100.0% used")).toBeDefined();
  });

  it("clamps percentage at 0% for negative used", () => {
    render(<UsageBar used={-100} limit={1_000_000} />);
    expect(screen.getByText("0.0% used")).toBeDefined();
  });

  it("fills the bar green, then yellow from 70%, then red from 90%", () => {
    const fill = (used: number) => {
      const { container, unmount } = render(
        <UsageBar used={used} limit={1_000_000} />,
      );
      const bar = container.querySelector("[style]")?.className ?? "";
      unmount();
      return bar;
    };
    expect(fill(500_000)).toContain("bg-success");
    expect(fill(750_000)).toContain("bg-yellow");
    expect(fill(950_000)).toContain("bg-error");
  });
});
