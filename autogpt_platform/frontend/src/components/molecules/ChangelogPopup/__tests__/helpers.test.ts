import { describe, expect, it } from "vitest";

import { cleanEntryMarkdown, parseChangelogIndex } from "../helpers";

const INDEX_MD = `
# Changelog

| Release | Highlights |
| --- | --- |
| [May 7 – June 10, 2026](https://agpt.co/docs/platform/changelog/changelog/v0-6-63) | Copilot upgrades and new blocks |
| [Apr 1 – May 6, 2026](https://agpt.co/docs/platform/changelog/changelog/v0-6-58) | Marketplace redesign |
`;

describe("parseChangelogIndex", () => {
  it("parses each release row into a structured entry", () => {
    const entries = parseChangelogIndex(INDEX_MD);

    expect(entries).toHaveLength(2);
    expect(entries[0]).toEqual({
      slug: "v0-6-63",
      dateRange: "May 7 – June 10, 2026",
      highlights: "Copilot upgrades and new blocks",
      url: "https://agpt.co/docs/platform/changelog/changelog/v0-6-63",
      mdUrl: "https://agpt.co/docs/platform/changelog/changelog/v0-6-63.md",
    });
    expect(entries[1].slug).toBe("v0-6-58");
  });

  it("returns an empty array when there are no release rows", () => {
    expect(parseChangelogIndex("no table here")).toEqual([]);
  });
});

describe("cleanEntryMarkdown", () => {
  it("strips GitBook liquid tags and figure wrappers", () => {
    const raw = `{% hint style="info" %}\n<figure><img src="x" /><figcaption>caption text</figcaption></figure>\nBody`;
    const cleaned = cleanEntryMarkdown(raw);

    expect(cleaned).not.toContain("{%");
    expect(cleaned).not.toContain("<figure>");
    expect(cleaned).not.toContain("figcaption");
    expect(cleaned).toContain("Body");
  });

  it("converts details/summary blocks into headings and rules", () => {
    const raw = `<details><summary>More details</summary>\nhidden\n</details>`;
    const cleaned = cleanEntryMarkdown(raw);

    expect(cleaned).toContain("---");
    expect(cleaned).toContain("### More details");
    expect(cleaned).not.toContain("<summary>");
  });
});
