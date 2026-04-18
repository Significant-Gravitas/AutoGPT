import type { UIMessage } from "ai";
import { describe, expect, it } from "vitest";

import { stripReplayPrefix } from "../stripReplayPrefix";

function user(id: string, text: string): UIMessage {
  return {
    id,
    role: "user",
    parts: [{ type: "text" as const, text, state: "done" as const }],
  };
}

function assistant(id: string, ...segments: string[]): UIMessage {
  return {
    id,
    role: "assistant",
    parts: segments.map((text) => ({
      type: "text" as const,
      text,
      state: "done" as const,
    })),
  };
}

/** Helper — extract the joined text content of an assistant's parts. */
function textOf(m: UIMessage): string {
  return (m.parts ?? [])
    .map((p) => ("text" in p && typeof p.text === "string" ? p.text : ""))
    .join("");
}

describe("stripReplayPrefix", () => {
  it("passes through when there is no replay", () => {
    const msgs = [
      user("u1", "hi"),
      assistant("a1", "hello"),
      user("u2", "how are you"),
      assistant("a2", "good thanks"),
    ];
    const result = stripReplayPrefix(msgs);
    expect(result.map((m) => m.id)).toEqual(["u1", "a1", "u2", "a2"]);
    expect(textOf(result[3])).toBe("good thanks");
  });

  it("drops an assistant that is a pure replay of an earlier one", () => {
    // a2 has the same text as a1 — CLI --resume replayed it as the new turn's
    // assistant before Claude emitted any new content yet.
    const msgs = [
      user("u1", "hi"),
      assistant("a1", "hello world"),
      user("u2", "tell me more"),
      assistant("a2", "hello world"),
    ];
    const result = stripReplayPrefix(msgs);
    const ids = result.map((m) => m.id);
    expect(ids).not.toContain("a2");
    expect(ids).toEqual(["u1", "a1", "u2"]);
  });

  it("strips the replayed prefix leaving only the new content", () => {
    // a2 starts with a1's text then continues with new content.
    const msgs = [
      user("u1", "hi"),
      assistant("a1", "hello world"),
      user("u2", "tell me more"),
      assistant("a2", "hello world\n\nAnd here is more info"),
    ];
    const result = stripReplayPrefix(msgs);
    const a2 = result.find((m) => m.id === "a2")!;
    expect(a2).toBeDefined();
    expect(textOf(a2)).toBe("\n\nAnd here is more info");
  });

  it("drops an assistant whose text is still a streaming-catch-up prefix of an earlier one", () => {
    // a2 is shorter than a1 and is a prefix of a1 — the CLI replay is
    // still streaming. Once a2 grows past a1 we'll hit the strip path;
    // for now, drop a2 to avoid the duplicate flash.
    const msgs = [
      user("u1", "hi"),
      assistant("a1", "hello world how are you today"),
      user("u2", "tell me more"),
      assistant("a2", "hello worl"),
    ];
    const result = stripReplayPrefix(msgs);
    const ids = result.map((m) => m.id);
    expect(ids).not.toContain("a2");
  });

  it("preserves non-text parts (step-start / tool-* ) during stripping", () => {
    // a2 is "prefix + new-text" with non-text parts interspersed. Only the
    // leading replay text should be stripped; structural parts stay.
    const a2: UIMessage = {
      id: "a2",
      role: "assistant",
      parts: [
        { type: "text" as const, text: "hello", state: "done" as const },
        {
          type: "tool-run_block" as const,
          toolCallId: "tc1",
          input: {},
          state: "output-available" as const,
          output: "ok",
        },
        {
          type: "text" as const,
          text: " world + extra",
          state: "done" as const,
        },
      ],
    };
    const msgs: UIMessage[] = [
      user("u1", "hi"),
      assistant("a1", "hello world"),
      user("u2", "tell me more"),
      a2,
    ];
    const result = stripReplayPrefix(msgs);
    const trimmed = result.find((m) => m.id === "a2");
    expect(trimmed).toBeDefined();
    // Leading "hello" text part consumed entirely, " world" partially stripped
    // (11 chars stripped, but " world" is 6 chars so the first 5 of " world"
    // which is part of replay stripped — tool-run_block and remaining text kept)
    const parts = trimmed!.parts ?? [];
    expect(parts[0]).toMatchObject({ type: "tool-run_block" });
  });

  it("leaves user messages untouched regardless of prefix matches", () => {
    const msgs = [
      user("u1", "hello"),
      user("u2", "hello world"), // user u2 starts with u1's content
    ];
    const result = stripReplayPrefix(msgs);
    expect(result.map((m) => m.id)).toEqual(["u1", "u2"]);
    expect(textOf(result[1])).toBe("hello world");
  });

  it("passes empty-text assistants through (nothing to compare)", () => {
    const msgs = [
      user("u1", "hi"),
      assistant("a1"), // no parts
      user("u2", "x"),
      assistant("a2", "response"),
    ];
    const result = stripReplayPrefix(msgs);
    const ids = result.map((m) => m.id);
    expect(ids).toContain("a1");
    expect(ids).toContain("a2");
  });

  it("preserves non-text parts that follow the stripped text prefix", () => {
    // a2 shares the leading "hello" text with a1, but has a non-text
    // structural part after the text (e.g. a tool-call) — that part must
    // survive the strip. Exercises the `remaining === 0` branch.
    const msgs: UIMessage[] = [
      user("u1", "hi"),
      assistant("a1", "hello"),
      user("u2", "more"),
      {
        id: "a2",
        role: "assistant",
        parts: [
          { type: "text" as const, text: "hello", state: "done" as const },
          // Non-text part (opaque to stripLeadingTextChars) must be kept.
          { type: "step-start" as const },
          { type: "text" as const, text: "!", state: "done" as const },
        ],
      } as UIMessage,
    ];

    const result = stripReplayPrefix(msgs);
    const a2 = result.find((m) => m.id === "a2")!;
    const kinds = (a2.parts ?? []).map((p) => p.type);
    // "hello" text is dropped; step-start + trailing "!" text remain.
    expect(kinds).toEqual(["step-start", "text"]);
    expect(textOf(a2)).toBe("!");
  });

  it("returns an empty result array for an empty input", () => {
    expect(stripReplayPrefix([])).toEqual([]);
  });

  it("handles the longest matching earlier assistant as the strip anchor", () => {
    // a3 matches a2's full text (which is longer than a1's). Should strip a2's
    // length, not a1's.
    const msgs = [
      user("u1", "hi"),
      assistant("a1", "hello"),
      user("u2", "again"),
      assistant("a2", "hello and more"),
      user("u3", "third turn"),
      assistant("a3", "hello and more\n\n[new content]"),
    ];
    const result = stripReplayPrefix(msgs);
    const a3 = result.find((m) => m.id === "a3")!;
    expect(textOf(a3)).toBe("\n\n[new content]");
  });
});
