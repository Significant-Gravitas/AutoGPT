import { describe, expect, test } from "vitest";
import {
  activeItemParamFor,
  isClientError,
  parseActiveItemParam,
  retryUnlessClientError,
} from "../components/NewAgentLibraryView/helpers";

describe("activeItem param prefix contract", () => {
  test("parses agent: and preset: prefixes", () => {
    expect(parseActiveItemParam("agent:abc-123")).toEqual({
      activeItemId: "abc-123",
      triggerKindHint: "trigger-agent",
    });
    expect(parseActiveItemParam("preset:def-456")).toEqual({
      activeItemId: "def-456",
      triggerKindHint: "webhook-trigger",
    });
  });

  test("passes through bare IDs and null with no hint", () => {
    expect(parseActiveItemParam("bare-id")).toEqual({
      activeItemId: "bare-id",
      triggerKindHint: null,
    });
    expect(parseActiveItemParam(null)).toEqual({
      activeItemId: null,
      triggerKindHint: null,
    });
  });

  test("round-trips through activeItemParamFor", () => {
    for (const kind of ["trigger-agent", "webhook-trigger"] as const) {
      const param = activeItemParamFor(kind, "some-id");
      expect(parseActiveItemParam(param)).toEqual({
        activeItemId: "some-id",
        triggerKindHint: kind,
      });
    }
  });
});

describe("retryUnlessClientError", () => {
  test("does not retry 4xx errors (the ~7s stale-link stall)", () => {
    const notFound = Object.assign(new Error("Preset #x not found"), {
      status: 404,
    });
    expect(isClientError(notFound)).toBe(true);
    expect(retryUnlessClientError(0, notFound)).toBe(false);
  });

  test("retries server errors and unknown error shapes up to 3 times", () => {
    const serverError = Object.assign(new Error("boom"), { status: 500 });
    expect(retryUnlessClientError(0, serverError)).toBe(true);
    expect(retryUnlessClientError(2, serverError)).toBe(true);
    expect(retryUnlessClientError(3, serverError)).toBe(false);

    expect(retryUnlessClientError(0, new Error("network down"))).toBe(true);
    expect(retryUnlessClientError(0, null)).toBe(true);
    expect(retryUnlessClientError(0, { status: "404" })).toBe(true);
  });
});
