import { render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, it } from "vitest";
import { UsageBar } from "../UsageBar";

describe("UsageBar", () => {
  it('renders "Unlimited" when limit is 0', () => {
    render(<UsageBar used={100} limit={0} />);
    expect(screen.getByText("Unlimited")).toBeDefined();
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
});
