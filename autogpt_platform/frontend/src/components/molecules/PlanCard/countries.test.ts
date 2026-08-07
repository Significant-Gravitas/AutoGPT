import { describe, expect, test } from "vitest";
import { formatPrice } from "./countries";

describe("formatPrice", () => {
  test("USD shows two decimals for sub-100 amounts", () => {
    expect(formatPrice(50, "USD", "$")).toBe("$50.00");
    expect(formatPrice(9.5, "USD", "$")).toBe("$9.50");
  });

  test("USD preserves cents for amounts >= 100 (no rounding)", () => {
    expect(formatPrice(249.45, "BRL", "R$")).toBe("R$249.45");
    expect(formatPrice(3264.99, "BRL", "R$")).toBe("R$3,264.99");
    expect(formatPrice(1000, "USD", "$")).toBe("$1,000.00");
  });

  test("zero-decimal currencies (JPY, KRW, HUF) show no decimals", () => {
    expect(formatPrice(7976, "JPY", "¥")).toBe("¥7,976");
    expect(formatPrice(1473.4, "KRW", "₩")).toBe("₩1,473");
  });

  test("rounds zero-decimal currencies to integers (Stripe rule)", () => {
    expect(formatPrice(1473.7, "KRW", "₩")).toBe("₩1,474");
  });

  test("zero is rendered as a real price, not skipped", () => {
    expect(formatPrice(0, "USD", "$")).toBe("$0.00");
    expect(formatPrice(0, "JPY", "¥")).toBe("¥0");
  });
});
