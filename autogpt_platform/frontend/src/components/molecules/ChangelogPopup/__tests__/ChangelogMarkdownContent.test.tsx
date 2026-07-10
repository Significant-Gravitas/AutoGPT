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

  it("resolves docs-relative links and images against the base URL", () => {
    render(
      <ChangelogMarkdownContent
        markdown={"A [relative link](../other) and ![img](assets/pic.png)."}
        baseUrl="https://agpt.co/docs/platform/changelog/changelog/v0-6-63"
      />,
    );

    const link = screen.getByRole("link", { name: "relative link" });
    expect(link.getAttribute("href")).toBe(
      "https://agpt.co/docs/platform/changelog/other",
    );
    expect(screen.getByAltText("img").getAttribute("src")).toBe(
      "https://agpt.co/docs/platform/changelog/changelog/assets/pic.png",
    );
  });
});
