import { describe, expect, it } from "vitest";

import { stripMarkdownForSpeech } from "../stripMarkdownForSpeech";

describe("stripMarkdownForSpeech", () => {
  it("drops the markers a reader would not say out loud", () => {
    expect(stripMarkdownForSpeech("## Heading\n**bold** and `code`")).toBe(
      "Heading\nbold and code",
    );
  });

  it("reads a numbered list as prose, not as its numerals", () => {
    expect(stripMarkdownForSpeech("1. Dutch Translator\n2. Summariser")).toBe(
      "Dutch Translator\nSummariser",
    );
  });

  it("keeps link text and drops the URL", () => {
    expect(stripMarkdownForSpeech("See [the docs](https://example.com).")).toBe(
      "See the docs.",
    );
  });

  it("leaves no tag behind, however they are nested", () => {
    for (const html of [
      "<div><p>hi</p></div>",
      "a<<b>b>c",
      "<scr<script>ipt>x",
    ]) {
      expect(stripMarkdownForSpeech(html)).not.toMatch(/<[^>]*>/);
    }
  });

  it("removes emoji", () => {
    expect(stripMarkdownForSpeech("Done 🎉")).toBe("Done");
  });
});
