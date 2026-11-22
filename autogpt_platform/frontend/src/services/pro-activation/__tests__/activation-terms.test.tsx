import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ActivationTerms } from "@/components/organisms/ProActivation/ActivationTerms";
import { quote } from "./fixtures";

const terms = {
  ...quote.terms!,
  currency: "usd",
  amount_due: 6000,
  renewal_unit_amount: 5000,
  billing_interval: "month" as const,
  renewal_discounts: [],
  trial_ends_at: new Date("2026-10-12T12:00:00Z"),
  period_starts_at: new Date("2026-10-07T12:00:00Z"),
  next_payment_at: new Date("2026-11-07T12:00:00Z"),
  tax_amount: 1000,
};

describe("Pro payment review", () => {
  it("shows the early trial end, paid period, actual tax and next payment", () => {
    render(<ActivationTerms terms={terms} />);
    expect(screen.getByText(/instead of Oct 12/)).toBeDefined();
    expect(screen.getByText(/billing date moves to the 7th/)).toBeDefined();
    expect(screen.getByText("Pro, Oct 7 to Nov 7")).toBeDefined();
    expect(screen.getByText("Tax")).toBeDefined();
    expect(screen.getByText("$10.00")).toBeDefined();
    expect(screen.getByText("Due today")).toBeDefined();
    expect(screen.getByText("$60.00")).toBeDefined();
    expect(screen.getByText(/Next payment of \$50.00 plus tax/)).toBeDefined();
    expect(screen.getByText(/After payment succeeds/)).toBeDefined();
    expect(screen.queryByText(/Quote valid until/)).toBeNull();
    const disclosure = screen
      .getByText("Full renewal terms")
      .closest("details");
    expect(disclosure?.open).toBe(false);
    fireEvent.click(screen.getByText("Full renewal terms"));
  });

  it("distinguishes unavailable tax from a quoted zero and included tax", () => {
    const result = render(
      <ActivationTerms terms={{ ...terms, tax_amount: null }} />,
    );
    expect(screen.queryByText("$0.00")).toBeNull();
    expect(screen.getByText("Included in total")).toBeDefined();
    result.rerender(
      <ActivationTerms terms={{ ...terms, tax_amount: 0, amount_due: 5000 }} />,
    );
    expect(screen.getByText("$0.00")).toBeDefined();
    result.rerender(
      <ActivationTerms
        terms={{
          ...terms,
          amount_due: 5000,
          renewal_tax: { price_tax_behavior: "inclusive" },
        }}
      />,
    );
    expect(screen.getByText("Tax (included)")).toBeDefined();
    expect(screen.queryByText(/plus tax/)).toBeNull();
  });

  it("names discounts and credits when the quote differs from the recurring price", () => {
    render(<ActivationTerms terms={{ ...terms, amount_due: 5400 }} />);
    expect(screen.getByText("Discounts, credits & adjustments")).toBeDefined();
    expect(screen.getByText("-$6.00")).toBeDefined();
    expect(screen.getByText("$54.00")).toBeDefined();
  });

  it("uses yearly terms without promising a monthly renewal", () => {
    render(
      <ActivationTerms
        terms={{
          ...terms,
          billing_interval: "year",
          next_payment_at: new Date("2027-10-07T12:00:00Z"),
        }}
      />,
    );
    expect(screen.getByText("Yearly")).toBeDefined();
    expect(screen.getByText(/then every year/)).toBeDefined();
    expect(screen.queryByText(/then every month/)).toBeNull();
  });

  it("keeps legacy saved attempts readable without invented dates", () => {
    render(
      <ActivationTerms
        terms={{
          ...terms,
          trial_ends_at: null,
          period_starts_at: null,
          next_payment_at: null,
        }}
      />,
    );
    expect(screen.queryByText(/instead of/)).toBeNull();
    expect(
      screen.getByText(/Your paid billing period begins when you confirm/),
    ).toBeDefined();
  });
});
