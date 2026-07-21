import { describe, expect, test } from "vitest";
import { isFilled } from "../useAgentRunModal";

// `isFilled` mirrors the backend's `_is_filled`
// (backend/copilot/tools/setup_agent_webhook_trigger.py), whose 0/false/empty
// behavior is pinned by
// `test_is_filled_treats_falsy_valid_values_as_filled` in
// setup_agent_webhook_trigger_test.py. Keep the two in sync.
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

  test("is a shallow check — non-empty containers of empty values are filled", () => {
    // Only the top-level container's emptiness matters (matches the backend);
    // nested null/undefined do not make it "missing".
    expect(isFilled({ a: null })).toBe(true);
    expect(isFilled({ a: undefined })).toBe(true);
    expect(isFilled([null, undefined])).toBe(true);
  });
});
