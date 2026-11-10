import { describe, expect, it } from "vitest";

import { findSegmentCutPoints, splitTextSegments } from "../text-segments";

const LONE_SURROGATE =
  /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/;

function piecesAt(text: string) {
  const cuts = findSegmentCutPoints(text);
  return cuts.map((cut, i) => text.slice(cuts[i - 1] ?? 0, cut));
}

describe("splitTextSegments", () => {
  it("keeps each word with the whitespace after it", () => {
    expect(splitTextSegments("Hello brave\n\nnew world")).toEqual([
      "Hello ",
      "brave\n\n",
      "new ",
      "world",
    ]);
  });

  it("splits Japanese written without spaces into words", () => {
    const text = "これは日本語の文章です。";
    const pieces = splitTextSegments(text);
    expect(pieces.join("")).toBe(text);
    expect(pieces.length).toBeGreaterThan(4);
    expect(pieces).toContain("日本語");
  });

  it("splits Chinese written without spaces into words", () => {
    const text = "我们今天去公园散步吧。";
    const pieces = splitTextSegments(text);
    expect(pieces.join("")).toBe(text);
    expect(pieces.length).toBeGreaterThan(4);
    expect(pieces).toContain("公园");
  });

  it("keeps the whitespace around an unspaced run on its edge pieces", () => {
    expect(splitTextSegments("\n使用React框架 ok")).toEqual([
      "\n使用",
      "React",
      "框架 ",
      "ok",
    ]);
  });

  it("never splits a surrogate pair or a combining sequence", () => {
    const text = "𠮷野家で𠮷を食べる。がくせい été ";
    const pieces = splitTextSegments(text);
    expect(pieces.join("")).toBe(text);
    for (const piece of pieces) {
      expect(piece).not.toMatch(LONE_SURROGATE);
      expect(piece).not.toMatch(/^\p{M}/u);
    }
  });

  it("does not cut between a space and the mark it carries", () => {
    const pieces = splitTextSegments("a ́b c");
    expect(pieces.join("")).toBe("a ́b c");
    for (const piece of pieces) expect(piece).not.toMatch(/^\p{M}/u);
  });
});

describe("findSegmentCutPoints", () => {
  it("holds back a trailing partial word", () => {
    expect(piecesAt("Hello wor")).toEqual(["Hello "]);
  });

  it("holds back whitespace that has no word after it yet", () => {
    expect(findSegmentCutPoints("\n\n")).toEqual([]);
  });

  it("cuts Japanese text into many pieces instead of holding it all", () => {
    const pieces = piecesAt("これは日本語の文章です。次の");
    expect(pieces.length).toBeGreaterThan(4);
    // The last word may still grow, so it waits.
    expect(pieces.join("")).not.toContain("次の");
  });

  it("cuts Chinese text into many pieces instead of holding it all", () => {
    const pieces = piecesAt("我们今天去公园散步吧。然后");
    expect(pieces.length).toBeGreaterThan(4);
    expect(pieces.join("")).toBe("我们今天去公园散步吧。");
  });
});
