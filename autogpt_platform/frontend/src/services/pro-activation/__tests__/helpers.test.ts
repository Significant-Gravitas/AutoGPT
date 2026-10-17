import { describe, expect, it } from "vitest";
import {
  formatAmount,
  safeReturnTo,
  invoiceURL,
  intervalLabel,
} from "../helpers";

describe("activation payment terms", () => {
  it("uses the currency's minor unit, including zero and three digit currencies", () => {
    expect(formatAmount(1234, "usd")).toBe("$12.34");
    expect(formatAmount(1234, "jpy")).toBe("¥1,234");
    expect(formatAmount(1234, "kwd")).toContain("1.234");
  });
  it("preserves local return context but rejects external destinations", () => {
    expect(safeReturnTo("/copilot/thread?resume=1#draft")).toBe(
      "/copilot/thread?resume=1#draft",
    );
    for (const path of [
      "//evil.example",
      "/\\evil.example",
      "https://evil.example",
      "/\nfoo",
    ]) {
      expect(safeReturnTo(path)).toBe("/settings/billing");
    }
  });
  it("allows only Stripe hosted invoice payment recovery", () => {
    expect(invoiceURL("https://invoice.stripe.com/i/123")).toBe(
      "https://invoice.stripe.com/i/123",
    );
    expect(invoiceURL("javascript:alert(1)")).toBeNull();
    expect(
      invoiceURL("https://invoice.stripe.com.evil.example/i/123"),
    ).toBeNull();
  });
  it("does not assume all quotes are monthly", () => {
    expect(intervalLabel("year", 1)).toBe("year");
    expect(intervalLabel("month", 3)).toBe("3 months");
  });
});
