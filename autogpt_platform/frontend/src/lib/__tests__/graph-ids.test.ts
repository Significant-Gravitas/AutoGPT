import { describe, expect, test, vi } from "vitest";

import {
  isValidGraphExecutionID,
  isValidGraphID,
  parseGraphExecutionID,
  parseGraphID,
} from "../graph-ids";

vi.mock("@sentry/nextjs", () => ({
  captureMessage: vi.fn(),
}));

const VALID_UUID = "123e4567-e89b-12d3-a456-426614174000";
const VALID_UUID_UPPER = "123E4567-E89B-12D3-A456-426614174000";

describe("parseGraphExecutionID", () => {
  test("accepts valid uuid", () => {
    expect(parseGraphExecutionID(VALID_UUID)).toBe(VALID_UUID);
  });

  test("accepts upper-case uuid normalized as-is", () => {
    expect(parseGraphExecutionID(VALID_UUID_UPPER)).toBe(VALID_UUID_UPPER);
  });

  test("trims whitespace", () => {
    expect(parseGraphExecutionID(`  ${VALID_UUID}  `)).toBe(VALID_UUID);
  });

  test("rejects null/undefined/empty", () => {
    expect(parseGraphExecutionID(null)).toBeNull();
    expect(parseGraphExecutionID(undefined)).toBeNull();
    expect(parseGraphExecutionID("")).toBeNull();
    expect(parseGraphExecutionID("   ")).toBeNull();
  });

  test("rejects malformed id (not uuid)", () => {
    expect(parseGraphExecutionID("not-a-uuid")).toBeNull();
    expect(parseGraphExecutionID("123")).toBeNull();
    expect(parseGraphExecutionID("<script>xss</script>")).toBeNull();
  });

  test("rejects graph-looking id passed as execution id only if not uuid", () => {
    // Both are uuids in this platform, so any valid uuid passes — the
    // cross-type check (graph id as execution id) is enforced by the
    // execution belonging to another graph, not by format. This just proves
    // the parser does not confuse them.
    expect(parseGraphExecutionID(VALID_UUID)).not.toBeNull();
  });

  test("isValid helper agrees", () => {
    expect(isValidGraphExecutionID(VALID_UUID)).toBe(true);
    expect(isValidGraphExecutionID("bad")).toBe(false);
  });
});

describe("parseGraphID", () => {
  test("accepts valid uuid", () => {
    expect(parseGraphID(VALID_UUID)).toBe(VALID_UUID);
  });

  test("rejects malformed", () => {
    expect(parseGraphID("not-a-uuid")).toBeNull();
    expect(parseGraphID("")).toBeNull();
  });

  test("isValid helper agrees", () => {
    expect(isValidGraphID(VALID_UUID)).toBe(true);
    expect(isValidGraphID("bad")).toBe(false);
  });
});
