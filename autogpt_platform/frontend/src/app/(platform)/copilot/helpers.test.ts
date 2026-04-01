import { describe, expect, it } from "vitest";
import {
  ORIGINAL_TITLE,
  formatNotificationTitle,
  parseSessionIDs,
} from "./helpers";

describe("formatNotificationTitle", () => {
  it("returns base title when count is 0", () => {
    expect(formatNotificationTitle(0)).toBe(ORIGINAL_TITLE);
  });

  it("returns formatted title with count", () => {
    expect(formatNotificationTitle(3)).toBe(
      `(3) AutoPilot is ready - ${ORIGINAL_TITLE}`,
    );
  });

  it("returns base title for negative count", () => {
    expect(formatNotificationTitle(-1)).toBe(ORIGINAL_TITLE);
  });

  it("returns base title for NaN", () => {
    expect(formatNotificationTitle(NaN)).toBe(ORIGINAL_TITLE);
  });

  it("returns formatted title for count of 1", () => {
    expect(formatNotificationTitle(1)).toBe(
      `(1) AutoPilot is ready - ${ORIGINAL_TITLE}`,
    );
  });
});

describe("parseSessionIDs", () => {
  it("returns empty set for null", () => {
    expect(parseSessionIDs(null)).toEqual(new Set());
  });

  it("returns empty set for undefined", () => {
    expect(parseSessionIDs(undefined)).toEqual(new Set());
  });

  it("returns empty set for empty string", () => {
    expect(parseSessionIDs("")).toEqual(new Set());
  });

  it("parses valid JSON array of strings", () => {
    expect(parseSessionIDs('["a","b","c"]')).toEqual(new Set(["a", "b", "c"]));
  });

  it("filters out non-string elements", () => {
    expect(parseSessionIDs('[1,"valid",null,true,"also-valid"]')).toEqual(
      new Set(["valid", "also-valid"]),
    );
  });

  it("returns empty set for non-array JSON", () => {
    expect(parseSessionIDs('{"key":"value"}')).toEqual(new Set());
  });

  it("returns empty set for JSON string value", () => {
    expect(parseSessionIDs('"oops"')).toEqual(new Set());
  });

  it("returns empty set for JSON number value", () => {
    expect(parseSessionIDs("42")).toEqual(new Set());
  });

  it("returns empty set for malformed JSON", () => {
    expect(parseSessionIDs("{broken")).toEqual(new Set());
  });

  it("deduplicates entries", () => {
    expect(parseSessionIDs('["a","a","b"]')).toEqual(new Set(["a", "b"]));
  });
});
