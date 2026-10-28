import { describe, expect, test } from "vitest";
import { splitHighlights, toPlainText, toRelease } from "../helpers";

describe("toPlainText", () => {
  test.each([
    ["**bold** and __strong__", "bold and strong"],
    ["[docs](https://agpt.co/docs) link", "docs link"],
    ["`code` and ~~gone~~", "code and gone"],
    ["line<br/>break <em>tag</em>", "line break tag"],
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
