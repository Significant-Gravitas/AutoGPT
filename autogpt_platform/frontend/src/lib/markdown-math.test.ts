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

  it.each([
    ["a thematic break", "---\n    echo $5 and $10"],
    ["a setext heading", "Title\n===\n    echo $5 and $10"],
  ])("skips indented code right after %s", (_, markdown) => {
    expect(escapeCurrencyAmounts(markdown)).toBe(markdown);
  });

  it.each([
    ["a thematic break", "-   item\n    ***\n        echo $5 and $10"],
    ["a tab-indented thematic break", "-\titem\n\t***\n\t    echo $5 and $10"],
    [
      "a setext heading",
      "-   item\n    para\n    ---\n        echo $5 and $10",
    ],
    ["an ATX heading", "-   item\n    # Head\n        echo $5 and $10"],
    ["a fenced block", "-   item\n\n    ```\n    echo $5 and $10\n    ```"],
  ])("reads %s indented inside a list item as a block", (_, markdown) => {
    expect(escapeCurrencyAmounts(markdown)).toBe(markdown);
  });

  it.each([
    ["```\n``` not a closer\necho $5 and $10\n```"],
    ["~~~\n~~~ x\necho $5 and $10\n~~~"],
  ])(
    "keeps a fence open past a fence line with text after it: %j",
    (markdown) => {
      expect(escapeCurrencyAmounts(markdown)).toBe(markdown);
    },
  );

  it("does not open a backtick fence whose info string has a backtick", () => {
    expect(escapeCurrencyAmounts("```a`b\n$5 and $10")).toBe(
      "```a`b\n\\$5 and \\$10",
    );
  });

  it("does not open a fence indented by a tab, which is indented code", () => {
    expect(escapeCurrencyAmounts("\t```\nprose $5 and $10")).toBe(
      "\t```\nprose \\$5 and \\$10",
    );
  });

  it("measures a tab after a list marker in columns", () => {
    expect(
      escapeCurrencyAmounts("- \titem\n\n       continued $5 and $10"),
    ).toBe("- \titem\n\n       continued \\$5 and \\$10");
  });

  it.each([
    ["at the top level", "    - echo $5 and $10"],
    ["inside a list item", "- item\n\n      - echo $5 and $10"],
  ])("skips indented code that starts like a list marker %s", (_, markdown) => {
    expect(escapeCurrencyAmounts(markdown)).toBe(markdown);
  });

  it.each([
    ["- a\n  - b\n\n  back in a\n\n    more $5 and $10"],
    ["1. a\n   - b\n\n   back in a\n\n     more $5 and $10"],
    ["- a\n  - b\n    - c\n\n    back in b\n\n      more $5 and $10"],
  ])("returns to the parent list item after a nested one: %j", (markdown) => {
    expect(escapeCurrencyAmounts(markdown)).toBe(
      markdown.replace("$5 and $10", "\\$5 and \\$10"),
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
