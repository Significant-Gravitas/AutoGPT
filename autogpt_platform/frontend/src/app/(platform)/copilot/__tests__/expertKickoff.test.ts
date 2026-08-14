import type { UIMessage } from "ai";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  buildKickoffMessage,
  clearKickoffPending,
  getKickoffExpertId,
  getKickoffExpertIdFromMetadata,
  getKickoffStatus,
  isKickoffMessage,
  isKickoffText,
  kickoffStorageKey,
  markKickoffDone,
  markKickoffPending,
  parseLegacyKickoffExpertId,
  stripKickoffMessages,
  stripLegacyKickoffMarker,
  withKickoffLock,
} from "../expertKickoff";

const EXPERT_ID = "3f8b0f7e-9f30-4a3b-a6a1-000000000001";

function userMessage(
  id: string,
  text: string,
  metadata?: UIMessage["metadata"],
): UIMessage {
  return {
    id,
    role: "user",
    parts: [{ type: "text", text }],
    metadata,
  } as UIMessage;
}

function assistantMessage(
  id: string,
  text: string,
  metadata?: UIMessage["metadata"],
): UIMessage {
  return {
    id,
    role: "assistant",
    parts: [{ type: "text", text }],
    metadata,
  } as UIMessage;
}

describe("buildKickoffMessage", () => {
  it("keeps routing data in metadata and sends a clean day-one prompt", () => {
    const message = buildKickoffMessage(EXPERT_ID);

    expect(message.metadata).toEqual({
      kind: "expert_kickoff",
      expertId: EXPERT_ID,
    });
    expect(message.text).toContain("You were just hired.");
    expect(message.text).toContain("Introduce yourself in 2-3 sentences");
    expect(message.text).toContain("run_agent");
    expect(message.text).toContain("If no workflow is installed");
    expect(message.text).toContain("Never pretend a run succeeded.");
    expect(message.text).not.toContain("EXPERT_KICKOFF");
    expect(message.text).not.toContain(EXPERT_ID);
  });
});

describe("kickoff message identification", () => {
  it("uses first-class metadata without matching ordinary prose", () => {
    const kickoff = buildKickoffMessage(EXPERT_ID);

    expect(
      isKickoffMessage(userMessage("m1", kickoff.text, kickoff.metadata)),
    ).toBe(true);
    expect(
      isKickoffMessage(
        userMessage(
          "m2",
          "You were just hired. Introduce yourself in 2-3 sentences in your voice",
        ),
      ),
    ).toBe(false);
    expect(
      isKickoffMessage(assistantMessage("m3", kickoff.text, kickoff.metadata)),
    ).toBe(false);
  });

  it("accepts persisted snake-case metadata and rejects malformed metadata", () => {
    expect(
      getKickoffExpertIdFromMetadata({
        kind: "expert_kickoff",
        expert_id: EXPERT_ID,
      }),
    ).toBe(EXPERT_ID);
    expect(
      getKickoffExpertIdFromMetadata({
        kind: "expert_kickoff",
        expert_id: "not-a-uuid",
      }),
    ).toBeNull();
    expect(
      getKickoffExpertIdFromMetadata({ kind: "ordinary", expertId: EXPERT_ID }),
    ).toBeNull();
  });

  it("removes only kickoff messages from the transcript", () => {
    const kickoff = buildKickoffMessage(EXPERT_ID);
    const messages = [
      userMessage("m1", kickoff.text, kickoff.metadata),
      assistantMessage("m2", "Hi, I'm Maria. Here's my day-one plan."),
      userMessage("m3", "Sounds good, go ahead."),
    ];

    expect(stripKickoffMessages(messages).map((message) => message.id)).toEqual(
      ["m2", "m3"],
    );
  });

  it("strictly recognizes legacy markers for backward compatibility", () => {
    const legacy = `[[EXPERT_KICKOFF:${EXPERT_ID}]]\n\nLegacy prompt`;

    expect(parseLegacyKickoffExpertId(legacy)).toBe(EXPERT_ID);
    expect(isKickoffText(legacy)).toBe(true);
    expect(stripLegacyKickoffMarker(legacy)).toBe("Legacy prompt");
    expect(getKickoffExpertId(userMessage("legacy", legacy))).toBe(EXPERT_ID);
    expect(
      parseLegacyKickoffExpertId(`[[EXPERT_KICKOFF:${EXPERT_ID}] Legacy`),
    ).toBeNull();
    expect(
      parseLegacyKickoffExpertId("[[EXPERT_KICKOFF:not-a-uuid]]"),
    ).toBeNull();
    expect(
      parseLegacyKickoffExpertId(`prefix [[EXPERT_KICKOFF:${EXPERT_ID}]]`),
    ).toBeNull();
  });
});

describe("kickoff status state machine", () => {
  afterEach(() => {
    window.localStorage.clear();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("uses a distinct, expert-scoped storage namespace", () => {
    expect(kickoffStorageKey(EXPERT_ID)).toBe(
      `expert-kickoff-status:${EXPERT_ID}`,
    );
  });

  it("walks idle → pending → done independently per expert", () => {
    const otherExpert = "3f8b0f7e-9f30-4a3b-a6a1-000000000002";

    expect(getKickoffStatus(EXPERT_ID)).toBe("idle");
    markKickoffPending(EXPERT_ID);
    expect(getKickoffStatus(EXPERT_ID)).toBe("pending");
    markKickoffDone(EXPERT_ID);
    expect(getKickoffStatus(EXPERT_ID)).toBe("done");
    expect(getKickoffStatus(otherExpert)).toBe("idle");
  });

  it("clearKickoffPending releases pending but never done", () => {
    markKickoffPending(EXPERT_ID);
    clearKickoffPending(EXPERT_ID);
    expect(getKickoffStatus(EXPERT_ID)).toBe("idle");

    markKickoffDone(EXPERT_ID);
    clearKickoffPending(EXPERT_ID);
    expect(getKickoffStatus(EXPERT_ID)).toBe("done");
  });

  it("expires stale pending state so a crashed tab cannot consume kickoff", () => {
    window.localStorage.setItem(
      kickoffStorageKey(EXPERT_ID),
      `pending:${Date.now() - 10 * 60 * 1000}`,
    );
    expect(getKickoffStatus(EXPERT_ID)).toBe("idle");
  });

  it("reads the previous key and value formats during rollout", () => {
    window.localStorage.setItem(`expert-kickoff-${EXPERT_ID}`, "1");
    expect(getKickoffStatus(EXPERT_ID)).toBe("done");
  });

  it("serializes cross-tab work with the Web Locks API", async () => {
    const request = vi.fn(
      async (_name: string, action: () => Promise<string>) => action(),
    );
    vi.stubGlobal("navigator", { locks: { request } });

    await expect(
      withKickoffLock(EXPERT_ID, async () => "started"),
    ).resolves.toBe("started");
    expect(request).toHaveBeenCalledWith(
      `expert-kickoff-status:${EXPERT_ID}`,
      expect.any(Function),
    );
  });
});
