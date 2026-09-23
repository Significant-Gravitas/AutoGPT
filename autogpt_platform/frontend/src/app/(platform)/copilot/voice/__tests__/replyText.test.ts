import { describe, expect, it } from "vitest";

import { createReplyTextReader } from "../replyText";

describe("createReplyTextReader", () => {
  it("returns only what is new on each read", () => {
    const reader = createReplyTextReader();
    expect(reader.read("m1", "Hello ")).toBe("Hello ");
    expect(reader.read("m1", "Hello there")).toBe("there");
  });

  it("drops fenced code as it streams", () => {
    const reader = createReplyTextReader();
    expect(reader.read("m1", "Here you go.\n")).toBe("Here you go.\n");
    expect(reader.read("m1", "Here you go.\n```python\n")).toBe("");
    expect(reader.read("m1", "Here you go.\n```python\nprint(1)\n")).toBe("");
    expect(
      reader.read("m1", "Here you go.\n```python\nprint(1)\n```\nDone."),
    ).toBe("Done.");
  });

  it("holds a partial line that could still open a fence", () => {
    const reader = createReplyTextReader();
    expect(reader.read("m1", "``")).toBe("");
    expect(reader.read("m1", "```js\nconst a = 1\n")).toBe("");
  });

  it("never emits an unterminated code block on flush", () => {
    const reader = createReplyTextReader();
    reader.read("m1", "Text.\n```\nhalf a program");
    expect(reader.flush()).toBe("");
  });

  it("flushes the trailing prose line", () => {
    const reader = createReplyTextReader();
    reader.read("m1", "Done");
    expect(reader.flush()).toBe("");
    reader.reset();
    reader.read("m1", "  ");
    expect(reader.flush()).toBe("  ");
  });

  it("never re-emits text it has already given out", () => {
    // The stream end swaps the streamed text for the server's copy, which
    // differs in whitespace. Emitting again there read the reply twice; but
    // text the server's copy has and the stream never delivered is owed.
    const reader = createReplyTextReader();
    expect(reader.read("m1", "Here is the answer. ")).toBe(
      "Here is the answer. ",
    );
    expect(reader.read("m1", "Here is the answer.  ")).toBe("");
    expect(
      reader.read("m1", "Here is the answer. Rewritten by the server. ").trim(),
    ).toBe("Rewritten by the server.");
  });

  it("speaks a turn replayed under a fresh id once, plus only what it adds", () => {
    // A reconnect replays the running turn from its start as a new message.
    // Keyed on the id, that was one full re-read per reconnect.
    const reader = createReplyTextReader();
    expect(reader.read("m1", "First sentence. Second sentence. ")).toBe(
      "First sentence. Second sentence. ",
    );
    expect(reader.read("m2", "First ")).toBe("");
    expect(reader.read("m2", "First sentence. Second ")).toBe("");
    expect(reader.read("m2", "First sentence. Second sentence. ")).toBe("");
    expect(
      reader
        .read("m2", "First sentence. Second sentence. Third sentence. ")
        .trim(),
    ).toBe("Third sentence.");
  });

  it("speaks a message restarted from empty under the same id once", () => {
    const reader = createReplyTextReader();
    expect(reader.read("m1", "Hello there. ")).toBe("Hello there. ");
    expect(reader.read("m1", "")).toBe("");
    expect(reader.read("m1", "Hello ")).toBe("");
    expect(reader.read("m1", "Hello there. Again. ").trim()).toBe("Again.");
  });

  it("matches a replay by text, not by whitespace", () => {
    const reader = createReplyTextReader();
    reader.read("m1", "One  two.\nThree. ");
    expect(reader.read("m2", "One two. Three. ")).toBe("");
    expect(reader.read("m2", "One two. Three. Four. ").trim()).toBe("Four.");
  });

  it("replays each tool-round passage once, in order", () => {
    const reader = createReplyTextReader();
    expect(reader.read("m1", "Let me check. ")).toBe("Let me check. ");
    expect(reader.read("m2", "Found it. ")).toBe("Found it. ");
    // The replay re-delivers both passages under new ids.
    expect(reader.read("m3", "Let me check. ")).toBe("");
    expect(reader.read("m4", "Found it. ")).toBe("");
    expect(reader.read("m4", "Found it. Done. ").trim()).toBe("Done.");
  });

  it("continues a restarted message from where it left off", () => {
    // Whatever was handed out before the restart — a whole line, or a word
    // still waiting for its full stop — is not handed out again.
    const reader = createReplyTextReader();
    expect(reader.read("m1", "First.\nDone")).toBe("First.\nDone");
    expect(reader.read("m1", "")).toBe("");
    expect(reader.read("m1", "First.\nDone")).toBe("");
    expect(reader.read("m1", "First.\nDone.\n")).toBe(".\n");
  });

  it("treats a new assistant message as new output, not a rewrite", () => {
    // A tool round starts a fresh message. Reading its text as a rewrite of
    // the previous one silently drops the entire answer.
    const reader = createReplyTextReader();
    expect(reader.read("m1", "Let me check that. ")).toBe(
      "Let me check that. ",
    );
    expect(reader.read("m2", "Here is what I found. ")).toBe(
      "Here is what I found. ",
    );
  });

  it("starts the next reply from scratch once reset", () => {
    const reader = createReplyTextReader();
    reader.read("m1", "First reply. ");
    reader.reset();
    expect(reader.read("m1", "A different reply. ")).toBe(
      "A different reply. ",
    );
  });
});
