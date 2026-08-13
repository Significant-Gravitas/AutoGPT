import { describe, expect, it } from "vitest";
import { shouldShowSessionProcessingIndicator } from "../sessionActivity";

const baseArgs = {
  sessionId: "s1",
  currentSessionId: "s2",
  isProcessing: true,
  hasCompletedIndicator: false,
  needsReload: false,
} as const;

describe("shouldShowSessionProcessingIndicator", () => {
  it("returns true for an inactive session that's processing without a completion or reload", () => {
    expect(shouldShowSessionProcessingIndicator({ ...baseArgs })).toBe(true);
  });

  it("returns false when the session is the currently focused one", () => {
    expect(
      shouldShowSessionProcessingIndicator({
        ...baseArgs,
        currentSessionId: "s1",
      }),
    ).toBe(false);
  });

  it("returns false when the session already shows a completion indicator", () => {
    expect(
      shouldShowSessionProcessingIndicator({
        ...baseArgs,
        hasCompletedIndicator: true,
      }),
    ).toBe(false);
  });

  it("returns false when the session needs a reload", () => {
    expect(
      shouldShowSessionProcessingIndicator({
        ...baseArgs,
        needsReload: true,
      }),
    ).toBe(false);
  });

  it("returns false when the session is not actively processing", () => {
    expect(
      shouldShowSessionProcessingIndicator({
        ...baseArgs,
        isProcessing: false,
      }),
    ).toBe(false);
  });

  it("returns true when no session is currently focused (null)", () => {
    expect(
      shouldShowSessionProcessingIndicator({
        ...baseArgs,
        currentSessionId: null,
      }),
    ).toBe(true);
  });
});
