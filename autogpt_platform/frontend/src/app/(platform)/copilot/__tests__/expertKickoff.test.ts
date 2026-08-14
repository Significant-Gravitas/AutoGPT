import type { UIMessage } from "ai";
import { afterEach, describe, expect, it } from "vitest";
import {
  buildKickoffMessage,
  clearKickoffPending,
  deriveKickoffMessageId,
  getKickoffStatus,
  isKickoffMessage,
  isKickoffText,
  kickoffStorageKey,
  markKickoffDone,
  markKickoffPending,
  stripKickoffMessages,
} from "../expertKickoff";

const EXPERT_ID = "3f8b0f7e-9f30-4a3b-a6a1-000000000001";

function userMessage(id: string, text: string): UIMessage {
  return { id, role: "user", parts: [{ type: "text", text }] } as UIMessage;
}

function assistantMessage(id: string, text: string): UIMessage {
  return {
    id,
    role: "assistant",
    parts: [{ type: "text", text }],
  } as UIMessage;
}

describe("buildKickoffMessage", () => {
  it("prefixes the structural marker and instructs the day-one job", () => {
    const message = buildKickoffMessage(EXPERT_ID);
    expect(message.startsWith(`[[EXPERT_KICKOFF:${EXPERT_ID}]]`)).toBe(true);
    expect(message).toContain("You were just hired.");
    expect(message).toContain("Introduce yourself in 2-3 sentences");
    expect(message).toContain("run_agent");
    expect(message).toContain("Never pretend a run succeeded.");
  });
});

describe("isKickoffMessage / stripKickoffMessages", () => {
  it("keys on the structural marker, never on prose content", () => {
    expect(
      isKickoffMessage(userMessage("m1", buildKickoffMessage(EXPERT_ID))),
    ).toBe(true);
    // A user pasting the kickoff WORDING (no marker) keeps their bubble.
    expect(
      isKickoffMessage(
        userMessage(
          "m2",
          "You were just hired. Introduce yourself in 2-3 sentences in your voice",
        ),
      ),
    ).toBe(false);
    // Same marked text on an assistant message is never hidden.
    expect(
      isKickoffMessage(assistantMessage("m3", buildKickoffMessage(EXPERT_ID))),
    ).toBe(false);
  });

  it("removes only kickoff messages from the transcript", () => {
    const messages = [
      userMessage("m1", buildKickoffMessage(EXPERT_ID)),
      assistantMessage("m2", "Hi, I'm Maria. Here's my day-one plan."),
      userMessage("m3", "Sounds good, go ahead."),
    ];

    expect(stripKickoffMessages(messages).map((m) => m.id)).toEqual([
      "m2",
      "m3",
    ]);
  });

  it("isKickoffText detects the marker prefix", () => {
    expect(isKickoffText(buildKickoffMessage(EXPERT_ID))).toBe(true);
    expect(isKickoffText("You were just hired.")).toBe(false);
  });
});

describe("deriveKickoffMessageId", () => {
  it("derives a deterministic attempt-0 id for the first kickoff", () => {
    const id = deriveKickoffMessageId([
      userMessage("m1", buildKickoffMessage(EXPERT_ID)),
    ]);
    expect(id).toBe(`expert-kickoff-${EXPERT_ID}-0`);
    expect(id!.length).toBeLessThanOrEqual(64);
  });

  it("increments the attempt when a prior kickoff exists (retry path)", () => {
    const id = deriveKickoffMessageId([
      userMessage("m1", buildKickoffMessage(EXPERT_ID)),
      assistantMessage("m2", "stream failed"),
      userMessage("m3", buildKickoffMessage(EXPERT_ID)),
    ]);
    expect(id).toBe(`expert-kickoff-${EXPERT_ID}-1`);
  });

  it("returns null for ordinary user messages", () => {
    expect(deriveKickoffMessageId([userMessage("m1", "Hello there")])).toBe(
      null,
    );
    expect(deriveKickoffMessageId([])).toBe(null);
  });
});

describe("kickoff status state machine", () => {
  afterEach(() => {
    window.localStorage.clear();
  });

  it("namespaces the key per expert", () => {
    expect(kickoffStorageKey("expert-maria")).toBe(
      "expert-kickoff-expert-maria",
    );
  });

  it("walks idle → pending → done", () => {
    expect(getKickoffStatus(EXPERT_ID)).toBe("idle");
    markKickoffPending(EXPERT_ID);
    expect(getKickoffStatus(EXPERT_ID)).toBe("pending");
    markKickoffDone(EXPERT_ID);
    expect(getKickoffStatus(EXPERT_ID)).toBe("done");
    // A different expert stays independent.
    expect(getKickoffStatus("expert-juan")).toBe("idle");
  });

  it("clearKickoffPending releases pending but never done", () => {
    markKickoffPending(EXPERT_ID);
    clearKickoffPending(EXPERT_ID);
    expect(getKickoffStatus(EXPERT_ID)).toBe("idle");

    markKickoffDone(EXPERT_ID);
    clearKickoffPending(EXPERT_ID);
    expect(getKickoffStatus(EXPERT_ID)).toBe("done");
  });

  it("expires a stale pending so a crashed tab can't consume the kickoff", () => {
    window.localStorage.setItem(
      kickoffStorageKey(EXPERT_ID),
      `pending:${Date.now() - 10 * 60 * 1000}`,
    );
    expect(getKickoffStatus(EXPERT_ID)).toBe("idle");
  });

  it("reads the legacy '1' value as done", () => {
    window.localStorage.setItem(kickoffStorageKey(EXPERT_ID), "1");
    expect(getKickoffStatus(EXPERT_ID)).toBe("done");
  });
});
