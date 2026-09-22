import { describe, expect, it } from "vitest";
import { escapeCurrencyAmounts } from "./markdown-math";

describe("escapeCurrencyAmounts", () => {
  it("escapes a price that would otherwise open a formula", () => {
    expect(escapeCurrencyAmounts("$5 and $10 buys lunch.")).toBe(
      "\\$5 and \\$10 buys lunch.",
    );
  });

  it("leaves math alone, including math that opens with a digit", () => {
    expect(escapeCurrencyAmounts("Growth is $5x^2$ and $(s+1)^{2}$.")).toBe(
      "Growth is $5x^2$ and $(s+1)^{2}$.",
    );
  });

  it("leaves $$ delimiters alone", () => {
    expect(escapeCurrencyAmounts("$$\n5x\n$$")).toBe("$$\n5x\n$$");
    expect(escapeCurrencyAmounts("Inline $$5x$$ here.")).toBe(
      "Inline $$5x$$ here.",
    );
  });

  it("skips fenced and inline code", () => {
    expect(
      escapeCurrencyAmounts("```\n$5 and $10\n```\nprose $5 and $10"),
    ).toBe("```\n$5 and $10\n```\nprose \\$5 and \\$10");
    expect(escapeCurrencyAmounts("Run `$5 and $10` now.")).toBe(
      "Run `$5 and $10` now.",
    );
  });

  it("keeps a fence open past a shorter closing fence", () => {
    const markdown = "````\n```\n$5 and $10\n````\nprose $5 and $10";
    expect(escapeCurrencyAmounts(markdown)).toBe(
      "````\n```\n$5 and $10\n````\nprose \\$5 and \\$10",
    );
  });

  it.each([
    [
      "after a blank line",
      "Run it:\n\n    echo $5 and $10\n\nprose $5 and $10",
    ],
    ["at the start of the text", "    echo $5 and $10\n\nprose $5 and $10"],
    ["indented with a tab", "\techo $5 and $10\n\nprose $5 and $10"],
    [
      "nested in a list item",
      "- item\n\n      echo $5 and $10\n\nprose $5 and $10",
    ],
  ])("skips indented code %s", (_, markdown) => {
    const [code] = markdown.split("\n\nprose");
    expect(escapeCurrencyAmounts(markdown)).toBe(
      `${code}\n\nprose \\$5 and \\$10`,
    );
  });

  it("escapes indented lines that are not code", () => {
    expect(
      escapeCurrencyAmounts(
        "- item\n\n    more $5 and $10\n\npara\n    lazy $5 and $10",
      ),
    ).toBe(
      "- item\n\n    more \\$5 and \\$10\n\npara\n    lazy \\$5 and \\$10",
    );
  });

  it("is idempotent, so an already-escaped price is left as one", () => {
    const once = escapeCurrencyAmounts("It costs $5.");
    expect(once).toBe("It costs \\$5.");
    expect(escapeCurrencyAmounts(once)).toBe(once);
  });
});
