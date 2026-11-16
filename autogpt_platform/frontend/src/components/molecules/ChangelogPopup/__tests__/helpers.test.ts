import { describe, expect, it } from "vitest";

import { isReleaseBeforeAccount, parseChangelogIndex } from "../helpers";

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

describe("isReleaseBeforeAccount", () => {
  const release = (slug: string) =>
    parseChangelogIndex(
      `| [x](/docs/platform/changelog/changelog/${slug}.md) | y |`,
    )[0];

  it.each([
    ["october-8-october-14-2026", "2026-10-15T00:00:00Z", true],
    ["october-8-october-14-2026", "2026-10-14T23:00:00Z", false],
    ["october-8-october-14-2026", "2026-10-01T00:00:00Z", false],
    ["september-19-2026", "2026-09-20T09:00:00Z", true],
    ["december-18-2025-january-8-2026", "2026-01-05T00:00:00Z", false],
    ["december-18-2025-january-8-2026", "2026-01-09T00:00:00Z", true],
    ["v0-6-63", "2027-01-01T00:00:00Z", false],
    ["february-30-2026", "2026-03-05T00:00:00Z", false],
    ["october-8-october-14-2026", undefined, false],
  ])("%s for an account created %s: %s", (slug, createdAt, expected) => {
    expect(isReleaseBeforeAccount(release(slug), createdAt)).toBe(expected);
  });
});
