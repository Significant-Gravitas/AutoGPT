import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { describe, expect, test } from "vitest";
import { getDownloadErrorDescription, getPackageFilename } from "./helpers";

describe("getPackageFilename", () => {
  test("uses the name the server sent", () => {
    expect(
      getPackageFilename('attachment; filename="maria.expert.zip"', "Maria"),
    ).toBe("maria.expert.zip");
  });

  test("falls back to an ASCII slug of the expert's name", () => {
    expect(getPackageFilename(null, "Zoë O'Brien — SEO")).toBe(
      "zoe-obrien-seo.expert.zip",
    );
  });

  test("falls back to expert.expert.zip when the name has nothing to slug", () => {
    expect(getPackageFilename("attachment", "研究者")).toBe(
      "expert.expert.zip",
    );
  });
});

describe("getDownloadErrorDescription", () => {
  test("explains the statuses the route documents", () => {
    expect(getDownloadErrorDescription(new ApiError("x", 404, null))).toBe(
      "This expert is no longer on the marketplace.",
    );
    expect(getDownloadErrorDescription(new ApiError("x", 413, null))).toBe(
      "This expert is too large to download as a file.",
    );
  });

  test("never shows the raw server message", () => {
    expect(
      getDownloadErrorDescription(new ApiError("Traceback ...", 500, null)),
    ).toBe("Something went wrong. Try again in a moment.");
    expect(getDownloadErrorDescription(new TypeError("Failed to fetch"))).toBe(
      "Something went wrong. Try again in a moment.",
    );
  });
});
