import { splitMarkdownBlocks } from "../markdownBlocks";

describe("splitMarkdownBlocks", () => {
  it("splits paragraphs and headings at blank lines and loses nothing", () => {
    const text = "# Title\n\nFirst paragraph.\n\nSecond paragraph.";
    const blocks = splitMarkdownBlocks(text);
    expect(blocks).toHaveLength(3);
    expect(blocks.join("\n")).toBe(text);
  });

  it("keeps a code fence with blank lines inside whole", () => {
    const text = "Intro\n\n```ts\nconst a = 1;\n\nconst b = 2;\n```\n\nAfter";
    const blocks = splitMarkdownBlocks(text);
    expect(blocks).toHaveLength(3);
    expect(blocks[1]).toContain("const a = 1;\n\nconst b = 2;");
  });

  it("keeps a loose list together so its numbering holds", () => {
    const text = "1. one\n\n2. two\n\n   more about two\n\n3. three";
    expect(splitMarkdownBlocks(text)).toEqual([text]);
  });

  it("keeps an unfinished fence open while it streams", () => {
    const text = "Intro\n\n```\ncode\n\nmore";
    expect(splitMarkdownBlocks(text)).toHaveLength(2);
  });
});
