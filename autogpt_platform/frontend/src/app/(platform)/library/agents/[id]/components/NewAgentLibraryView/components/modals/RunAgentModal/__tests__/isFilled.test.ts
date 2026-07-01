import { describe, expect, test } from "vitest";
import { isFilled } from "../useAgentRunModal";

describe("isFilled", () => {
  test("treats null/undefined/empty string/empty collections as missing", () => {
    expect(isFilled(null)).toBe(false);
    expect(isFilled(undefined)).toBe(false);
    expect(isFilled("")).toBe(false);
    expect(isFilled({})).toBe(false);
    expect(isFilled([])).toBe(false);
  });

  test("treats falsy primitives 0 and false as filled (matches the backend)", () => {
    expect(isFilled(0)).toBe(true);
    expect(isFilled(false)).toBe(true);
  });

  test("treats non-empty values as filled", () => {
    expect(isFilled("x")).toBe(true);
    expect(isFilled([1])).toBe(true);
    expect(isFilled({ a: 1 })).toBe(true);
  });
});
