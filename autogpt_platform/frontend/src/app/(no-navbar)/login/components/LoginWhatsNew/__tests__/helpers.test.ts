import { describe, expect, test } from "vitest";
import { splitHighlights, toPlainText, toRelease } from "../helpers";

describe("toPlainText", () => {
  test.each([
    ["**bold** and __strong__", "bold and strong"],
    ["[docs](https://agpt.co/docs) link", "docs link"],
    ["`code` and ~~gone~~", "code and gone"],
    ["line<br/>break <em>tag</em>", "line break tag"],
    ["cut off <script", "cut off"],
    ["Clay <3 Rock", "Clay <3 Rock"],
    ["kept <!-- hidden, note --> text", "kept text"],
    ["cut <!-- never closed", "cut"],
    ["[wiki](https://x.org/a_(b)) page", "wiki page"],
    ["Clay &amp; Rock &lt;3", "Clay & Rock <3"],
    ["keeps snake_case_names", "keeps snake_case_names"],
    ["  spaced \n  out  ", "spaced out"],
    ["emoji 🎉 stays", "emoji 🎉 stays"],
  ])("%j reads as %j", (markdown, text) => {
    expect(toPlainText(markdown)).toBe(text);
  });
});

describe("splitHighlights", () => {
  test("keeps commas inside brackets with their highlight", () => {
    expect(splitHighlights("New models (Claude, GPT), faster runs,,")).toEqual([
      "New models (Claude, GPT)",
      "faster runs",
    ]);
  });
});

describe("toRelease", () => {
  test("heads the release with its first highlight", () => {
    expect(
      toRelease({
        slug: "september-19-2026",
        dateRange: "September 19",
        highlights: "meet your Experts, routines, Skills marketplace",
        url: "https://agpt.co/docs/platform/changelog/changelog/september-19-2026",
      }),
    ).toEqual({
      slug: "september-19-2026",
      url: "https://agpt.co/docs/platform/changelog/changelog/september-19-2026",
      dateRange: "September 19",
      headline: "Meet your Experts",
      details: "Routines, Skills marketplace",
    });
  });

  test("keeps a comma in link text and drops HTML comments", () => {
    const release = toRelease({
      slug: "s",
      dateRange: "May 1",
      highlights:
        "[Clay, Rock](https://agpt.co/x) avatars <!-- internal, unpublished -->, faster runs",
      url: "u",
    });
    expect(release.headline).toBe("Clay, Rock avatars");
    expect(release.details).toBe("Faster runs");
  });

  test("keeps emphasis that spans highlights from leaving stray markers", () => {
    const release = toRelease({
      slug: "s",
      dateRange: "May 1",
      highlights: "**new agents, faster runs**, [docs, guides](https://x.org)",
      url: "u",
    });
    expect(release.headline).toBe("New agents");
    expect(release.details).toBe("Faster runs, docs, guides");
  });

  test.each([
    ["iOS deep links", "iOS deep links"],
    ["macOS app", "macOS app"],
    ["e-mail digests", "e-mail digests"],
    ["new models", "New models"],
  ])("capitalises %j as %j", (highlights, headline) => {
    const release = toRelease({
      slug: "s",
      dateRange: "",
      highlights,
      url: "u",
    });
    expect(release.headline).toBe(headline);
  });

  test("names a release with no readable highlights", () => {
    const release = toRelease({
      slug: "s",
      dateRange: "May 1",
      highlights: "  ,  ",
      url: "u",
    });
    expect(release.headline).toBe("Release notes");
    expect(release.details).toBe("");
  });
});
