import { render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, it } from "vitest";

import { ChangelogMarkdownContent } from "../components/ChangelogMarkdownContent";

describe("ChangelogMarkdownContent", () => {
  it("renders markdown headings, emphasis, links, and images", () => {
    render(
      <ChangelogMarkdownContent
        markdown={
          "# Release notes\n\nSome **bold** text and a [docs link](https://agpt.co/docs).\n\n![a screenshot](https://agpt.co/img.png)"
        }
      />,
    );

    expect(
      screen.getByRole("heading", { name: "Release notes" }),
    ).toBeDefined();
    expect(screen.getByText("bold")).toBeDefined();

    const link = screen.getByRole("link", { name: "docs link" });
    expect(link.getAttribute("href")).toBe("https://agpt.co/docs");
    expect(link.getAttribute("target")).toBe("_blank");
    expect(link.getAttribute("rel")).toContain("noopener");

    expect(screen.getByAltText("a screenshot")).toBeDefined();
  });

  it("resolves relative links against the base URL but leaves the image src (already a same-origin proxy URL) untouched", () => {
    render(
      <ChangelogMarkdownContent
        markdown={
          "A [relative link](../other) and ![img](/api/changelog/image?file=pic.png)."
        }
        baseUrl="https://agpt.co/docs/platform/changelog/changelog/v0-6-63"
      />,
    );

    const link = screen.getByRole("link", { name: "relative link" });
    expect(link.getAttribute("href")).toBe(
      "https://agpt.co/docs/platform/changelog/other",
    );
    // Images are rewritten to our proxy in cleanEntryMarkdown, so the renderer
    // must NOT re-resolve them against the docs origin.
    expect(screen.getByAltText("img").getAttribute("src")).toBe(
      "/api/changelog/image?file=pic.png",
    );
  });
});
