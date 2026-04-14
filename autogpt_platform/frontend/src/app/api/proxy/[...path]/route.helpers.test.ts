import { describe, expect, it } from "vitest";
import {
  isWorkspaceDownloadRequest,
  isRedirectStatus,
  isTransientWorkspaceDownloadStatus,
  getWorkspaceDownloadErrorMessage,
} from "./route.helpers";

describe("isWorkspaceDownloadRequest", () => {
  it("matches api/workspace/files/{id}/download pattern", () => {
    expect(
      isWorkspaceDownloadRequest([
        "api",
        "workspace",
        "files",
        "abc-123",
        "download",
      ]),
    ).toBe(true);
  });

  it("rejects paths with wrong segment count", () => {
    expect(
      isWorkspaceDownloadRequest(["api", "workspace", "files", "download"]),
    ).toBe(false);
    expect(
      isWorkspaceDownloadRequest([
        "api",
        "workspace",
        "files",
        "id",
        "download",
        "extra",
      ]),
    ).toBe(false);
  });

  it("rejects paths with wrong prefix", () => {
    expect(
      isWorkspaceDownloadRequest([
        "v1",
        "workspace",
        "files",
        "id",
        "download",
      ]),
    ).toBe(false);
  });

  it("rejects paths not ending with download", () => {
    expect(
      isWorkspaceDownloadRequest([
        "api",
        "workspace",
        "files",
        "id",
        "metadata",
      ]),
    ).toBe(false);
  });
});

describe("isRedirectStatus", () => {
  it.each([301, 302, 303, 307, 308])("returns true for %d", (status) => {
    expect(isRedirectStatus(status)).toBe(true);
  });

  it.each([200, 304, 400, 404, 500])("returns false for %d", (status) => {
    expect(isRedirectStatus(status)).toBe(false);
  });
});

describe("isTransientWorkspaceDownloadStatus", () => {
  it.each([408, 429, 500, 502, 503, 504])(
    "returns true for transient %d",
    (status) => {
      expect(isTransientWorkspaceDownloadStatus(status)).toBe(true);
    },
  );

  it.each([400, 401, 403, 404, 405])(
    "returns false for non-transient %d",
    (status) => {
      expect(isTransientWorkspaceDownloadStatus(status)).toBe(false);
    },
  );
});

describe("getWorkspaceDownloadErrorMessage", () => {
  it("extracts detail string from object", () => {
    expect(getWorkspaceDownloadErrorMessage({ detail: "Not found" })).toBe(
      "Not found",
    );
  });

  it("extracts error string from object", () => {
    expect(getWorkspaceDownloadErrorMessage({ error: "Server error" })).toBe(
      "Server error",
    );
  });

  it("extracts nested detail.message", () => {
    expect(
      getWorkspaceDownloadErrorMessage({
        detail: { message: "Nested error" },
      }),
    ).toBe("Nested error");
  });

  it("returns trimmed string body", () => {
    expect(getWorkspaceDownloadErrorMessage("  error text  ")).toBe(
      "error text",
    );
  });

  it("returns null for empty string", () => {
    expect(getWorkspaceDownloadErrorMessage("")).toBeNull();
  });

  it("returns null for whitespace-only string", () => {
    expect(getWorkspaceDownloadErrorMessage("   ")).toBeNull();
  });

  it("returns null for null/undefined", () => {
    expect(getWorkspaceDownloadErrorMessage(null)).toBeNull();
    expect(getWorkspaceDownloadErrorMessage(undefined)).toBeNull();
  });

  it("returns null for object with empty detail", () => {
    expect(getWorkspaceDownloadErrorMessage({ detail: "" })).toBeNull();
  });

  it("returns null for object with no recognized keys", () => {
    expect(getWorkspaceDownloadErrorMessage({ foo: "bar" })).toBeNull();
  });

  it("prefers detail over error", () => {
    expect(
      getWorkspaceDownloadErrorMessage({
        detail: "detail msg",
        error: "error msg",
      }),
    ).toBe("detail msg");
  });
});
