import { describe, expect, it } from "vitest";

import { cleanEntryMarkdown, parseChangelogIndex } from "../helpers";

// gitbook README table format: | [date](slug.md) | highlights |
const INDEX_MD = `
# Changelog

| Date | Highlights |
| ---- | ---------- |
| [May 7 – June 10](may-7-june-10-2026.md) | Copilot upgrades and new blocks |
| [Apr 1 – May 6](april-10-may-1-2026.md) | Marketplace redesign |
`;

describe("parseChangelogIndex", () => {
  it("parses each release row, routing the markdown through our proxy", () => {
    const entries = parseChangelogIndex(INDEX_MD);

    expect(entries).toHaveLength(2);
    expect(entries[0]).toEqual({
      slug: "may-7-june-10-2026",
      dateRange: "May 7 – June 10",
      highlights: "Copilot upgrades and new blocks",
      url: "https://agpt.co/docs/platform/changelog/changelog/may-7-june-10-2026",
      mdUrl: "/api/changelog?slug=may-7-june-10-2026",
    });
    expect(entries[1].slug).toBe("april-10-may-1-2026");
  });

  it("ignores the header/divider rows", () => {
    expect(
      parseChangelogIndex("| Date | Highlights |\n| ---- | ---------- |"),
    ).toEqual([]);
  });
});

describe("cleanEntryMarkdown", () => {
  it("routes gitbook <figure> images through the cached image proxy", () => {
    const raw = `<figure><img src="../.gitbook/assets/hero.png" alt="A hero"><figcaption><p>The caption</p></figcaption></figure>`;
    const cleaned = cleanEntryMarkdown(raw);

    expect(cleaned).toContain("![A hero](/api/changelog/image?file=hero.png)");
    expect(cleaned).toContain("*The caption*");
    expect(cleaned).not.toContain("<figure>");
  });

  it("strips GitBook liquid tags and converts details/summary", () => {
    const raw = `{% hint style="info" %}\n<details><summary>More</summary>\nx\n</details>`;
    const cleaned = cleanEntryMarkdown(raw);

    expect(cleaned).not.toContain("{%");
    expect(cleaned).toContain("### More");
    expect(cleaned).toContain("---");
  });

  it("drops the GitBook export boilerplate blockquote", () => {
    const raw =
      "> For the complete documentation index, see llms.txt.\n\n# Title";
    const cleaned = cleanEntryMarkdown(raw);

    expect(cleaned).not.toContain("complete documentation index");
    expect(cleaned).toContain("# Title");
  });
});
