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

  it("returns the input array itself when nothing is a replay", () => {
    const msgs = [
      user("u1", "hi"),
      assistant("a1", "hello"),
      user("u2", "how are you"),
      assistant("a2", "good thanks"),
    ];
    expect(stripReplayPrefix(msgs)).toBe(msgs);
  });

  it("returns a new array when it drops or trims a message", () => {
    const dropped = [
      user("u1", "hi"),
      assistant("a1", "hello world"),
      user("u2", "tell me more"),
      assistant("a2", "hello world"),
    ];
    expect(stripReplayPrefix(dropped)).not.toBe(dropped);

    const trimmed = [
      user("u1", "hi"),
      assistant("a1", "hello"),
      user("u2", "more"),
      assistant("a2", "hello", " and more"),
    ];
    const result = stripReplayPrefix(trimmed);
    expect(result).not.toBe(trimmed);
    expect(result).toHaveLength(4);
    expect(textOf(result[3])).toBe(" and more");
  });

  it("gives the same result as a fresh call while the newest reply streams", () => {
    const history = [
      user("u1", "hi"),
      assistant("a1", "hello"),
      user("u2", "again"),
      assistant("a2", "hello and more"),
      user("u3", "go"),
    ];
    const reply = "hello and more, then something new";
    for (let n = 1; n <= reply.length; n++) {
      const msgs = [...history, assistant("a3", reply.slice(0, n))];
      const result = stripReplayPrefix(msgs);
      const fresh = reply.slice(0, n);
      if ("hello and more".startsWith(fresh)) {
        expect(result.map((m) => m.id)).toEqual(["u1", "a1", "u2", "a2", "u3"]);
      } else {
        expect(textOf(result[5])).toBe(fresh.slice("hello and more".length));
      }
    }
  });

  it("does not carry a replay decision over to a conversation whose earlier text differs", () => {
    const replayed = [
      user("u1", "hi"),
      assistant("a1", "hello"),
      user("u2", "more"),
      assistant("a2", "hello there"),
    ];
    const notReplayed = [
      user("u1", "hi"),
      assistant("a1", "goodbye"),
      user("u2", "more"),
      assistant("a2", "hello there"),
    ];

    for (let round = 0; round < 2; round++) {
      expect(textOf(stripReplayPrefix(replayed)[3])).toBe(" there");
      expect(stripReplayPrefix(notReplayed)).toBe(notReplayed);
    }
  });

  it("re-checks every later message when an earlier one changes", () => {
    const before = [
      user("u1", "hi"),
      assistant("a1", "plan"),
      user("u2", "next"),
      assistant("a2", "plan, step two"),
    ];
    expect(textOf(stripReplayPrefix(before)[3])).toBe(", step two");

    const edited = [
      ...before.slice(0, 1),
      assistant("a1", "other"),
      ...before.slice(2),
    ];
    expect(stripReplayPrefix(edited)).toBe(edited);
  });

  it("returns a new array when it removes a duplicate or a seeded copy", () => {
    const duplicated = [user("u1", "hi"), user("u1", "hi")];
    expect(stripReplayPrefix(duplicated)).toEqual([duplicated[0]]);

    const seeded = assistant("a1", "Saved.");
    const resumed = {
      ...assistant("a2", "Saved."),
      parts: [
        ...structuredClone(seeded.parts),
        { type: "step-start" as const },
      ],
    };
    const result = stripReplayPrefix([user("u1", "go"), seeded, resumed]);
    expect(result.map((m) => m.id)).toEqual(["u1", "a1", "a2"]);
    expect(result[2].parts).toEqual([{ type: "step-start" }]);
  });

  it("drops the copy of the bubble before it that a resumed turn opens with, cards included", () => {
    // `useChat` seeds a GET-resumed turn from the assistant message the list
    // ends on, so the turn's message starts with all of that bubble's parts.
    const paused = {
      id: "a1",
      role: "assistant",
      parts: [
        { type: "text", text: "Halfway there.", state: "done" },
        { type: "data-turn-paused", data: { reason: "steps", steps: 450 } },
        { type: "text", text: "Keep going.", state: "done" },
        { type: "data-status", data: { message: "Picking it back up…" } },
      ],
    } as UIMessage;
    const continued = {
      id: "a2",
      role: "assistant",
      parts: [
        ...structuredClone(paused.parts),
        { type: "step-start" },
        { type: "text", text: "Back on it.", state: "streaming" },
      ],
    } as UIMessage;

    const result = stripReplayPrefix([user("u1", "go"), paused, continued]);

    expect(result.map((m) => m.id)).toEqual(["u1", "a1", "a2"]);
    expect(result[1]).toBe(paused);
    expect(result[2].parts).toEqual([
      { type: "step-start" },
      { type: "text", text: "Back on it.", state: "streaming" },
    ]);
  });

  it("keeps a bubble that only resembles the one before it", () => {
    const next = {
      id: "a2",
      role: "assistant",
      parts: [
        { type: "data-turn-paused", data: { reason: "task" } },
        { type: "text", text: "Next.", state: "done" },
      ],
    } as UIMessage;

    const result = stripReplayPrefix([
      user("u1", "go"),
      assistant("a1", "Saved."),
      next,
    ]);

    expect(result[2]).toBe(next);
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
