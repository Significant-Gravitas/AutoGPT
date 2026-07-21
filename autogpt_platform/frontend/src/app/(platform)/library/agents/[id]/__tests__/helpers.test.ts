import type { LibraryAgentPreset } from "@/app/api/__generated__/models/libraryAgentPreset";
import { describe, expect, test } from "vitest";
import {
  activeItemParamFor,
  deriveSelectedTriggerKind,
  parseActiveItemParam,
  retryUnlessClientError,
} from "../components/NewAgentLibraryView/helpers";

describe("activeItem param prefix contract", () => {
  test("round-trips both kinds and passes through bare IDs and null", () => {
    for (const kind of ["trigger-agent", "webhook-trigger"] as const) {
      expect(parseActiveItemParam(activeItemParamFor(kind, "some-id"))).toEqual(
        { activeItemId: "some-id", triggerKindHint: kind },
      );
    }
    expect(parseActiveItemParam("bare-id")).toEqual({
      activeItemId: "bare-id",
      triggerKindHint: null,
    });
    expect(parseActiveItemParam(null)).toEqual({
      activeItemId: null,
      triggerKindHint: null,
    });
  });
});

describe("retryUnlessClientError", () => {
  test("fails fast on 4xx, retries server/unknown errors up to 3 times", () => {
    const notFound = Object.assign(new Error("Preset #x not found"), {
      status: 404,
    });
    expect(retryUnlessClientError(0, notFound)).toBe(false);

    const serverError = Object.assign(new Error("boom"), { status: 500 });
    expect(retryUnlessClientError(0, serverError)).toBe(true);
    expect(retryUnlessClientError(3, serverError)).toBe(false);
    expect(retryUnlessClientError(0, new Error("network down"))).toBe(true);
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
  });

  test("only concludes not-found when the presets page is complete", () => {
    expect(
      deriveSelectedTriggerKind({ ...settled, activeItemId: "unknown-id" }),
    ).toBe("not-found");
    expect(
      deriveSelectedTriggerKind({
        ...settled,
        activeItemId: "unknown-id",
        presetsComplete: false,
      }),
    ).toBe("webhook-trigger");
  });
});
