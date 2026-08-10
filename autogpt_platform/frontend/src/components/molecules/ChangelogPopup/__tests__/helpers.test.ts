import { describe, expect, it } from "vitest";

import { parseChangelogIndex } from "../helpers";

// gitbook README table format: | [date](slug.md) | highlights |
const INDEX_MD = `
# Changelog

| Date | Highlights |
| ---- | ---------- |
| [May 7 – June 10](may-7-june-10-2026.md) | Copilot upgrades, new blocks, faster runs |
| [Apr 1 – May 6](april-10-may-1-2026.md) | Marketplace redesign |
`;

describe("parseChangelogIndex", () => {
  it("parses each release row into a summary that links to the docs page", () => {
    const entries = parseChangelogIndex(INDEX_MD);

    expect(entries).toHaveLength(2);
    expect(entries[0]).toEqual({
      slug: "may-7-june-10-2026",
      dateRange: "May 7 – June 10",
      highlights: "Copilot upgrades, new blocks, faster runs",
      url: "https://agpt.co/docs/platform/changelog/changelog/may-7-june-10-2026",
    });
    expect(entries[1].slug).toBe("april-10-may-1-2026");
  });

  it("ignores the header/divider rows", () => {
    expect(
      parseChangelogIndex("| Date | Highlights |\n| ---- | ---------- |"),
    ).toEqual([]);
  });
});
