import { describe, expect, test } from "vitest";
import { aboutPlaceholderFor } from "./helpers";

describe("aboutPlaceholderFor", () => {
  test("returns a generic placeholder without a name", () => {
    expect(aboutPlaceholderFor(null)).toBe(
      "How they should work, what you care about, anything that helps them sound like yours…",
    );
    expect(aboutPlaceholderFor("   ")).toBe(
      "How they should work, what you care about, anything that helps them sound like yours…",
    );
  });

  test("personalizes the placeholder with the expert name", () => {
    expect(aboutPlaceholderFor("Nova")).toBe(
      "How Nova should work, what you care about, anything that helps them sound like yours…",
    );
    expect(aboutPlaceholderFor("  Ada  ")).toBe(
      "How Ada should work, what you care about, anything that helps them sound like yours…",
    );
  });
});
