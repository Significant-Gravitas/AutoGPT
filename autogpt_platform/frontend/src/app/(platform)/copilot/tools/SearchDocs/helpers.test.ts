import { describe, expect, it } from "vitest";
import { toDocsUrl } from "./helpers";

describe("toDocsUrl", () => {
  it("builds extension-less rendered-page URLs on the live host", () => {
    expect(toDocsUrl("platform/block-sdk-guide.md")).toBe(
      "https://agpt.co/docs/platform/block-sdk-guide",
    );
    expect(toDocsUrl("a/b.mdx")).toBe("https://agpt.co/docs/a/b");
  });

  it("canonicalizes underscores to hyphens like the site's edge worker", () => {
    expect(toDocsUrl("platform/new_blocks.md")).toBe(
      "https://agpt.co/docs/platform/new-blocks",
    );
  });

  it("only strips markdown extensions, not dots inside names", () => {
    expect(toDocsUrl("platform/v1.2-guide.md")).toBe(
      "https://agpt.co/docs/platform/v1.2-guide",
    );
  });
});
