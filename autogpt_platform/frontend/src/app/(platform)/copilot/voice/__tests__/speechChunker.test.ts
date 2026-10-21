import { describe, expect, it } from "vitest";

import { MAX_CHUNK_CHARS, takeSpeakableChunks } from "../speechChunker";

describe("takeSpeakableChunks", () => {
  it("emits a sentence as soon as it is terminated", () => {
    const { chunks, rest } = takeSpeakableChunks("Sure. Let me look");
    expect(chunks).toEqual(["Sure."]);
    expect(rest).toBe("Let me look");
  });

  it("holds an unterminated sentence back", () => {
    expect(takeSpeakableChunks("Still writing").chunks).toEqual([]);
  });

  it("holds a terminator at the very end of the buffer", () => {
    // The next delta may turn "3." into "3.5".
    expect(takeSpeakableChunks("It costs 3.").chunks).toEqual([]);
    expect(takeSpeakableChunks("It costs 3.", true).chunks).toEqual([
      "It costs 3.",
    ]);
  });

  it("does not split a decimal mid-stream", () => {
    const { chunks } = takeSpeakableChunks("It costs 3.50 dollars today");
    expect(chunks).toEqual([]);
  });

  it("splits several sentences at once", () => {
    const { chunks } = takeSpeakableChunks("One. Two! Three? ");
    expect(chunks).toEqual(["One.", "Two!", "Three?"]);
  });

  it("breaks on newlines so list items are spoken separately", () => {
    const { chunks, rest } = takeSpeakableChunks("First item\nSecond");
    expect(chunks).toEqual(["First item"]);
    expect(rest).toBe("Second");
  });

  it("hard-splits prose that never ends a sentence", () => {
    const long = "word ".repeat(80);
    const { chunks } = takeSpeakableChunks(long);
    expect(chunks.length).toBeGreaterThan(0);
    expect(chunks[0].length).toBeLessThanOrEqual(MAX_CHUNK_CHARS);
    expect(chunks[0].endsWith("word")).toBe(true);
  });

  it("emits the trailing partial sentence on flush", () => {
    const { chunks, rest } = takeSpeakableChunks("Done. Almost", true);
    expect(chunks).toEqual(["Done.", "Almost"]);
    expect(rest).toBe("");
  });

  it("drops whitespace-only remainders", () => {
    expect(takeSpeakableChunks("   \n  ", true).chunks).toEqual([]);
  });
});
