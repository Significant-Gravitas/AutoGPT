import type { UIMessage } from "ai";
import { afterEach, describe, expect, it } from "vitest";
import {
  buildKickoffMessage,
  hasKickedOff,
  isKickoffMessage,
  kickoffStorageKey,
  markKickedOff,
  stripKickoffMessages,
} from "../expertKickoff";

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
  it("tells the expert to introduce itself and start its day-one job", () => {
    const message = buildKickoffMessage();
    expect(message).toContain("You were just hired.");
    expect(message).toContain("Introduce yourself in 2-3 sentences");
    expect(message).toContain("run_agent");
    expect(message).toContain("Never pretend a run succeeded.");
  });

  it("is a stable single string", () => {
    expect(buildKickoffMessage()).toBe(buildKickoffMessage());
  });
});

describe("kickoff localStorage latch", () => {
  afterEach(() => {
    window.localStorage.clear();
  });

  it("namespaces the key per expert", () => {
    expect(kickoffStorageKey("expert-maria")).toBe(
      "expert-kickoff-expert-maria",
    );
  });

  it("round-trips the once-per-expert flag", () => {
    expect(hasKickedOff("expert-maria")).toBe(false);
    markKickedOff("expert-maria");
    expect(hasKickedOff("expert-maria")).toBe(true);
    // A different expert stays independent.
    expect(hasKickedOff("expert-juan")).toBe(false);
  });
});

describe("isKickoffMessage / stripKickoffMessages", () => {
  it("matches only the exact kickoff user message", () => {
    expect(isKickoffMessage(userMessage("m1", buildKickoffMessage()))).toBe(
      true,
    );
    // Same text but assistant role is never hidden.
    expect(
      isKickoffMessage(assistantMessage("m2", buildKickoffMessage())),
    ).toBe(false);
    // A real user message with different text stays visible.
    expect(isKickoffMessage(userMessage("m3", "Hello there"))).toBe(false);
  });

  it("removes the kickoff message while keeping the rest of the thread", () => {
    const messages = [
      userMessage("m1", buildKickoffMessage()),
      assistantMessage("m2", "Hi, I'm Maria. Here's my day-one plan."),
      userMessage("m3", "Sounds good, go ahead."),
    ];

    const visible = stripKickoffMessages(messages);

    expect(visible.map((m) => m.id)).toEqual(["m2", "m3"]);
  });
});
