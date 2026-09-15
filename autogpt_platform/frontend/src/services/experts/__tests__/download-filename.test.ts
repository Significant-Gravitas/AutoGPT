import { filenameFromContentDisposition } from "@/lib/download-file";
import { describe, expect, test } from "vitest";
import { expertPackageFilename } from "../helpers";

describe("filenameFromContentDisposition", () => {
  test.each([
    ['attachment; filename="maria-ops.expert.zip"', "maria-ops.expert.zip"],
    ["attachment; filename=maria.expert.zip", "maria.expert.zip"],
    // RFC 5987 wins over the ascii name beside it: it is the form that
    // survives a non-ascii expert name.
    [
      "attachment; filename=\"zoe.expert.zip\"; filename*=UTF-8''Zo%C3%AB.expert.zip",
      "Zoë.expert.zip",
    ],
    ["attachment", "fallback.zip"],
    // A malformed escape falls back rather than throwing mid-download.
    ["attachment; filename*=UTF-8''%E0%A4%A", "fallback.zip"],
  ])("reads %s", (header, expected) => {
    const headers = new Headers({ "Content-Disposition": header });

    expect(filenameFromContentDisposition(headers, "fallback.zip")).toBe(
      expected,
    );
  });

  test("falls back when the header is absent", () => {
    expect(filenameFromContentDisposition(new Headers(), "fallback.zip")).toBe(
      "fallback.zip",
    );
  });
});

describe("expertPackageFilename", () => {
  test.each([
    ["Maria Ops", "maria-ops.expert.zip"],
    ["  Ada  ", "ada.expert.zip"],
    ["Zoë's #1 Analyst", "zo-s-1-analyst.expert.zip"],
    ["!!!", "expert.expert.zip"],
    // Capped at 60 characters, and never left ending on the dash the cap cut.
    [`${"a".repeat(59)} tail`, `${"a".repeat(59)}.expert.zip`],
  ])("names %s", (name, expected) => {
    expect(expertPackageFilename(name)).toBe(expected);
  });
});
