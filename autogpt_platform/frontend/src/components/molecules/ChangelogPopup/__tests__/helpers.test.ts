import { describe, expect, it } from "vitest";

import { parseChangelogIndex } from "../helpers";

// The docs index links to relative `.md` release pages.
const INDEX_MD = `
# Changelog

| Release | Highlights |
| --- | --- |
| [May 7 – June 10, 2026](/docs/platform/changelog/changelog/may-7-june-10-2026.md) | Copilot upgrades and new blocks |
| [Apr 1 – May 6, 2026](/docs/platform/changelog/changelog/april-10-may-1-2026.md) | Marketplace redesign |
`;

describe("parseChangelogIndex", () => {
  it("parses each release row into a summary that links to the docs page", () => {
    const entries = parseChangelogIndex(INDEX_MD);

    expect(entries).toHaveLength(2);
    expect(entries[0]).toEqual({
      slug: "may-7-june-10-2026",
      dateRange: "May 7 – June 10, 2026",
      highlights: "Copilot upgrades and new blocks",
      url: "https://agpt.co/docs/platform/changelog/changelog/may-7-june-10-2026",
    });
    expect(entries[1].slug).toBe("april-10-may-1-2026");
  });

  it("also parses absolute links without a .md suffix (older docs format)", () => {
    const md =
      "| [Old](https://agpt.co/docs/platform/changelog/changelog/v0-6-58) | notes |";
    const entries = parseChangelogIndex(md);
    expect(entries[0].slug).toBe("v0-6-58");
    expect(entries[0].url).toBe(
      "https://agpt.co/docs/platform/changelog/changelog/v0-6-58",
    );
  });

  it("returns an empty array when there are no release rows", () => {
    expect(parseChangelogIndex("no table here")).toEqual([]);
  });
});
