import { describe, expect, it } from "vitest";

import { parseChangelogIndex } from "../helpers";

const INDEX_MD = `
# What's New

| Date | Highlights |
| ---- | ---------- |
| [June 18 – July 9](/docs/platform/changelog/changelog/june-18-july-9-2026.md) | File folders, shareable chats, Discord bot |
| [May 7 – June 10](/docs/platform/changelog/changelog/may-7-june-10-2026.md) | AutoPilot upgrades, redesigned login, live subscriptions, rebuilt settings |
`;

describe("parseChangelogIndex", () => {
  it("parses published release rows with site-relative links", () => {
    const entries = parseChangelogIndex(INDEX_MD);

    expect(entries).toHaveLength(2);
    expect(entries[0]).toEqual({
      slug: "june-18-july-9-2026",
      dateRange: "June 18 – July 9",
      highlights: "File folders, shareable chats, Discord bot",
      url: "https://agpt.co/docs/platform/changelog/changelog/june-18-july-9-2026",
    });
    expect(entries[1].slug).toBe("may-7-june-10-2026");
  });

  it("ignores the header/divider rows", () => {
    expect(
      parseChangelogIndex("| Date | Highlights |\n| ---- | ---------- |"),
    ).toEqual([]);
  });
});
