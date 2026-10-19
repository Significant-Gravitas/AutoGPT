import { describe, expect, it } from "vitest";
import { isJsonTextFor } from "./json-text";

describe("isJsonTextFor", () => {
  it("matches the same value however it is formatted", () => {
    expect(isJsonTextFor('{"a":1}', { a: 1 })).toBe(true);
    expect(isJsonTextFor('{\n  "a": 1\n}', { a: 1 })).toBe(true);
    expect(isJsonTextFor("[1, 2]", [1, 2])).toBe(true);
  });

  it("does not match a different value", () => {
    expect(isJsonTextFor('{"a":1}', { a: 2 })).toBe(false);
    expect(isJsonTextFor("[1, 2]", [2, 1])).toBe(false);
  });

  it("does not match text that is not JSON yet", () => {
    expect(isJsonTextFor('{"a":', { a: 1 })).toBe(false);
  });

  it("matches empty text only to no value", () => {
    expect(isJsonTextFor("  ", undefined)).toBe(true);
    expect(isJsonTextFor("", null)).toBe(true);
    expect(isJsonTextFor("", {})).toBe(false);
  });

  // The stored value would be a rounded copy, so the field must show it
  // rather than keep digits that will not be saved.
  it("does not match numbers JSON parsing can't hold exactly", () => {
    const id = "1234567890123456789";
    expect(
      isJsonTextFor(
        `{"channel_id": ${id}}`,
        JSON.parse(`{"channel_id": ${id}}`),
      ),
    ).toBe(false);
    expect(isJsonTextFor("[1e400]", [Infinity])).toBe(false);
  });

  it("still matches safe numbers and digits inside strings", () => {
    expect(
      isJsonTextFor("[9007199254740991, -2.5e3]", [9007199254740991, -2500]),
    ).toBe(true);
    expect(
      isJsonTextFor('{"id": "1234567890123456789"}', {
        id: "1234567890123456789",
      }),
    ).toBe(true);
  });
});
