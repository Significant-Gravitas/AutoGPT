import { describe, expect, test } from "vitest";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { describeFolderMoveError } from "./useArtifactsFolders";

describe("describeFolderMoveError", () => {
  test("names the folder on a clash at the destination", () => {
    expect(
      describeFolderMoveError(new ApiError("conflict", 409, null), "Q3"),
    ).toBe("A folder named “Q3” is already there");
  });

  test("explains a refused cycle", () => {
    expect(
      describeFolderMoveError(new ApiError("bad request", 400, null), "Q3"),
    ).toBe("A folder can't be moved into itself");
  });

  test("falls back for anything else, including a non-API failure", () => {
    expect(describeFolderMoveError(new ApiError("boom", 500, null), "Q3")).toBe(
      "Failed to move folder",
    );
    expect(describeFolderMoveError(new Error("offline"), "Q3")).toBe(
      "Failed to move folder",
    );
  });
});
