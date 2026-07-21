import { describe, expect, test } from "vitest";
import type { LibraryAgentPreset } from "@/app/api/__generated__/models/libraryAgentPreset";
import {
  activeItemParamFor,
  deriveSelectedTriggerKind,
  isClientError,
  isNotFoundError,
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
    expect(retryUnlessClientError(0, { status: 401 })).toBe(false);
    expect(retryUnlessClientError(0, { status: 403 })).toBe(false);
    expect(retryUnlessClientError(0, { status: 422 })).toBe(false);
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

  test("isNotFoundError only matches a numeric 404 status", () => {
    expect(isNotFoundError({ status: 404 })).toBe(true);
    expect(isNotFoundError({ status: 403 })).toBe(false);
    expect(isNotFoundError({ status: 422 })).toBe(false);
    expect(isNotFoundError({ status: "404" })).toBe(false);
    expect(isNotFoundError(new Error("not found"))).toBe(false);
    expect(isNotFoundError(null)).toBe(false);
  });
});

describe("deriveSelectedTriggerKind", () => {
  const webhookPreset = {
    id: "preset-1",
    webhook_id: "webhook-1",
  } as LibraryAgentPreset;
  const template = { id: "template-1", webhook_id: null } as LibraryAgentPreset;
  const settled = {
    triggerAgents: [{ id: "agent-1" }],
    presets: [webhookPreset, template],
    presetsComplete: true,
    listsResolved: true,
    anyListFailed: false,
    triggerKindHint: null,
  };

  test("returns null without a selection", () => {
    expect(
      deriveSelectedTriggerKind({ ...settled, activeItemId: null }),
    ).toBeNull();
  });

  test("resolves membership, ignoring a contradicting hint", () => {
    expect(
      deriveSelectedTriggerKind({
        ...settled,
        activeItemId: "agent-1",
        triggerKindHint: "webhook-trigger",
      }),
    ).toBe("trigger-agent");
    expect(
      deriveSelectedTriggerKind({
        ...settled,
        activeItemId: "preset-1",
        triggerKindHint: "trigger-agent",
      }),
    ).toBe("webhook-trigger");
  });

  test("does not classify a non-webhook template as a trigger", () => {
    expect(
      deriveSelectedTriggerKind({ ...settled, activeItemId: "template-1" }),
    ).toBe("not-found");
  });

  test("uses the hint while lists are unresolved, else loading/error", () => {
    const unresolved = {
      ...settled,
      activeItemId: "unknown-id",
      triggerAgents: undefined,
      presets: undefined,
      listsResolved: false,
    };
    expect(
      deriveSelectedTriggerKind({
        ...unresolved,
        triggerKindHint: "trigger-agent",
      }),
    ).toBe("trigger-agent");
    expect(deriveSelectedTriggerKind(unresolved)).toBe("loading");
    expect(
      deriveSelectedTriggerKind({ ...unresolved, anyListFailed: true }),
    ).toBe("error");
    // A hint outranks a failed list: the hinted view can still render and
    // fetch its own data even when the other list errored.
    expect(
      deriveSelectedTriggerKind({
        ...unresolved,
        anyListFailed: true,
        triggerKindHint: "webhook-trigger",
      }),
    ).toBe("webhook-trigger");
  });

  test("falls back to the preset detail view when the presets page is incomplete", () => {
    expect(
      deriveSelectedTriggerKind({
        ...settled,
        activeItemId: "unknown-id",
        presetsComplete: false,
      }),
    ).toBe("webhook-trigger");
  });

  test("concludes not-found only when both lists are complete and resolved", () => {
    expect(
      deriveSelectedTriggerKind({ ...settled, activeItemId: "unknown-id" }),
    ).toBe("not-found");
  });
});
