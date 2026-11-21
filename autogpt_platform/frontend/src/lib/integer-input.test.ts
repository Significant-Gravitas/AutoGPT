import { describe, expect, it } from "vitest";
import { parseIntegerInput } from "./integer-input";

describe("parseIntegerInput", () => {
  it.each([
    ["42", 42],
    ["-42", -42],
    [" 7 ", 7],
    ["9007199254740991", 9007199254740991],
    ["3.7", 3],
    ["-3.7", -3],
    ["1e3", 1000],
  ])("parses %j as the number %j", (text, expected) => {
    expect(parseIntegerInput(text)).toBe(expected);
  });

  it.each([
    ["9007199254740993", "9007199254740993"],
    ["-9007199254740993", "-9007199254740993"],
    ["9223372036854775807", "9223372036854775807"],
    [" 9007199254740993 ", "9007199254740993"],
  ])("keeps %j as the exact digit string", (text, expected) => {
    expect(parseIntegerInput(text)).toBe(expected);
  });

  it.each(["", "  ", "1,234", "abc", "1e309"])(
    "returns undefined for %j",
    (text) => {
      expect(parseIntegerInput(text)).toBeUndefined();
    },
  );
});
