import type { UIMessage } from "ai";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  buildKickoffMessage,
  clearKickoffPending,
  getKickoffAttemptTokenFromMetadata,
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
const USER_ID = "user-1";

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
    const message = buildKickoffMessage(EXPERT_ID, "attempt-1");

    expect(message.metadata).toEqual({
      kind: "expert_kickoff",
      expertId: EXPERT_ID,
      attemptToken: "attempt-1",
    });
    expect(getKickoffAttemptTokenFromMetadata(message.metadata)).toBe(
      "attempt-1",
    );
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

  it("uses a distinct, user-and-expert-scoped storage namespace", () => {
    expect(kickoffStorageKey(USER_ID, EXPERT_ID)).toBe(
      `expert-kickoff-status:${USER_ID}:${EXPERT_ID}`,
    );
  });

  it("walks idle → pending → done independently per user and expert", () => {
    const otherExpert = "3f8b0f7e-9f30-4a3b-a6a1-000000000002";
    const otherUser = "user-2";

    expect(getKickoffStatus(USER_ID, EXPERT_ID)).toBe("idle");
    const attemptToken = markKickoffPending(USER_ID, EXPERT_ID);
    expect(getKickoffStatus(USER_ID, EXPERT_ID)).toBe("pending");
    markKickoffDone(USER_ID, EXPERT_ID, attemptToken);
    expect(getKickoffStatus(USER_ID, EXPERT_ID)).toBe("done");
    expect(getKickoffStatus(USER_ID, otherExpert)).toBe("idle");
    expect(getKickoffStatus(otherUser, EXPERT_ID)).toBe("idle");
  });

  it("clearKickoffPending releases pending but never done", () => {
    const abandonedAttempt = markKickoffPending(USER_ID, EXPERT_ID);
    clearKickoffPending(USER_ID, EXPERT_ID, abandonedAttempt);
    expect(getKickoffStatus(USER_ID, EXPERT_ID)).toBe("idle");

    const completedAttempt = markKickoffPending(USER_ID, EXPERT_ID);
    markKickoffDone(USER_ID, EXPERT_ID, completedAttempt);
    clearKickoffPending(USER_ID, EXPERT_ID, completedAttempt);
    expect(getKickoffStatus(USER_ID, EXPERT_ID)).toBe("done");
  });

  it("expires stale pending state so a crashed tab cannot consume kickoff", () => {
    window.localStorage.setItem(
      kickoffStorageKey(USER_ID, EXPERT_ID),
      `pending:${Date.now() - 10 * 60 * 1000}:expired-attempt`,
    );
    expect(getKickoffStatus(USER_ID, EXPERT_ID)).toBe("idle");
  });

  it("does not let an expired attempt overwrite or clear its retry", () => {
    const now = Date.now();
    const dateNow = vi.spyOn(Date, "now");
    dateNow.mockReturnValue(now - 10 * 60 * 1000);
    const expiredAttempt = markKickoffPending(USER_ID, EXPERT_ID);

    dateNow.mockReturnValue(now);
    expect(getKickoffStatus(USER_ID, EXPERT_ID)).toBe("idle");
    const retryAttempt = markKickoffPending(USER_ID, EXPERT_ID);

    expect(markKickoffDone(USER_ID, EXPERT_ID, expiredAttempt)).toBe(false);
    clearKickoffPending(USER_ID, EXPERT_ID, expiredAttempt);
    expect(getKickoffStatus(USER_ID, EXPERT_ID)).toBe("pending");

    expect(markKickoffDone(USER_ID, EXPERT_ID, retryAttempt)).toBe(true);
    expect(getKickoffStatus(USER_ID, EXPERT_ID)).toBe("done");
  });

  it("ignores unscoped legacy state so it cannot leak between accounts", () => {
    window.localStorage.setItem(`expert-kickoff-${EXPERT_ID}`, "1");
    window.localStorage.setItem(`expert-kickoff-status:${EXPERT_ID}`, "done");

    expect(getKickoffStatus(USER_ID, EXPERT_ID)).toBe("idle");
  });

  it("serializes cross-tab work with the Web Locks API", async () => {
    const request = vi.fn(
      async (_name: string, action: () => Promise<string>) => action(),
    );
    vi.stubGlobal("navigator", { locks: { request } });

    await expect(
      withKickoffLock(USER_ID, EXPERT_ID, async () => "started"),
    ).resolves.toBe("started");
    await expect(
      withKickoffLock("user-2", EXPERT_ID, async () => "also-started"),
    ).resolves.toBe("also-started");
    expect(request).toHaveBeenNthCalledWith(
      1,
      `expert-kickoff-status:${USER_ID}:${EXPERT_ID}`,
      expect.any(Function),
    );
    expect(request).toHaveBeenNthCalledWith(
      2,
      `expert-kickoff-status:user-2:${EXPERT_ID}`,
      expect.any(Function),
    );
  });
});
