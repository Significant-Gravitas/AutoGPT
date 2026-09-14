import type { UIMessage } from "ai";
import { describe, expect, it } from "vitest";
import {
  getLatestAssistantStatusMessage,
  isBookkeepingPart,
} from "../messageParts";

type Part = UIMessage["parts"][number];

function part(type: string, data?: unknown): Part {
  return { type, data } as unknown as Part;
}

function assistant(parts: Part[]): UIMessage {
  return { id: "m1", role: "assistant", parts } as UIMessage;
}

const STATUS = part("data-status", { message: "Looking up your agents" });

describe("isBookkeepingPart", () => {
  it("covers every transient part the stream drives chrome with", () => {
    expect(isBookkeepingPart({ type: "data-cursor" })).toBe(true);
    expect(isBookkeepingPart({ type: "data-status" })).toBe(true);
    expect(isBookkeepingPart({ type: "data-dream-operations" })).toBe(true);
    expect(isBookkeepingPart({ type: "data-compaction" })).toBe(true);
  });

  it("leaves content alone", () => {
    expect(isBookkeepingPart({ type: "text" })).toBe(false);
    expect(isBookkeepingPart({ type: "reasoning" })).toBe(false);
    expect(isBookkeepingPart({ type: "step-start" })).toBe(false);
    expect(isBookkeepingPart({ type: "tool-run_agent" })).toBe(false);
    expect(isBookkeepingPart({ type: "tool-context_compaction" })).toBe(false);
  });
});

describe("getLatestAssistantStatusMessage", () => {
  it("reads the latest status", () => {
    expect(getLatestAssistantStatusMessage([assistant([STATUS])])).toBe(
      "Looking up your agents",
    );
  });

  it("keeps the status while a compaction runs behind it", () => {
    // A compaction starting is not the model answering — the status is
    // still the most recent thing the user was told.
    const messages = [
      assistant([
        STATUS,
        part("data-compaction", {
          phase: "summarizing",
          tokensBefore: 128_000,
        }),
        part("data-compaction", { phase: "rebuilding" }),
      ]),
    ];
    expect(getLatestAssistantStatusMessage(messages)).toBe(
      "Looking up your agents",
    );
  });

  it("drops the status once real content lands past it", () => {
    const messages = [
      assistant([
        STATUS,
        part("data-compaction", { phase: "summarizing" }),
        { type: "text", text: "Here you go." } as unknown as Part,
      ]),
    ];
    expect(getLatestAssistantStatusMessage(messages)).toBeNull();
  });

  it("ignores non-assistant tails and malformed payloads", () => {
    expect(
      getLatestAssistantStatusMessage([
        { id: "u1", role: "user", parts: [STATUS] } as UIMessage,
      ]),
    ).toBeNull();
    expect(
      getLatestAssistantStatusMessage([assistant([part("data-status", {})])]),
    ).toBeNull();
  });
});
