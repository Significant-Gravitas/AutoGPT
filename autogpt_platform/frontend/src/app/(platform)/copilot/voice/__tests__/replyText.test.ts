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
    // The stream end swaps the streamed text for the server's copy. Emitting
    // again there is what read the whole reply aloud a second time.
    const reader = createReplyTextReader();
    expect(reader.read("m1", "Here is the answer. ")).toBe(
      "Here is the answer. ",
    );
    expect(reader.read("m1", "Here is the answer.  ")).toBe("");
    expect(
      reader.read("m1", "Here is the answer. Rewritten by the server. "),
    ).toBe("");
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
