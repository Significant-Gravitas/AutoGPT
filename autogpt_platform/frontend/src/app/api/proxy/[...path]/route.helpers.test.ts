import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import {
  isWorkspaceDownloadRequest,
  isRedirectStatus,
  isTransientWorkspaceDownloadStatus,
  getWorkspaceDownloadErrorMessage,
  fetchWorkspaceDownloadOnce,
  fetchWorkspaceDownloadWithRetry,
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

describe("fetchWorkspaceDownloadOnce", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("returns response directly for non-redirect status", async () => {
    const mockResponse = { ok: true, status: 200, headers: new Headers() };
    vi.mocked(fetch).mockResolvedValue(mockResponse as unknown as Response);

    const result = await fetchWorkspaceDownloadOnce("https://backend/file", {});
    expect(result).toBe(mockResponse);
    expect(fetch).toHaveBeenCalledOnce();
  });

  it("follows redirect when Location header is present", async () => {
    const redirectResponse = {
      ok: false,
      status: 302,
      headers: new Headers({ Location: "https://storage.example.com/file" }),
    };
    const finalResponse = { ok: true, status: 200, headers: new Headers() };
    vi.mocked(fetch)
      .mockResolvedValueOnce(redirectResponse as unknown as Response)
      .mockResolvedValueOnce(finalResponse as unknown as Response);

    const result = await fetchWorkspaceDownloadOnce("https://backend/file", {
      Authorization: "Bearer token",
    });
    expect(result).toBe(finalResponse);
    expect(fetch).toHaveBeenCalledTimes(2);
    expect(fetch).toHaveBeenNthCalledWith(
      2,
      "https://storage.example.com/file",
      { method: "GET", redirect: "follow" },
    );
  });

  it("returns redirect response when Location header is missing", async () => {
    const redirectResponse = {
      ok: false,
      status: 307,
      headers: new Headers(),
    };
    vi.mocked(fetch).mockResolvedValue(redirectResponse as unknown as Response);

    const result = await fetchWorkspaceDownloadOnce("https://backend/file", {});
    expect(result).toBe(redirectResponse);
    expect(fetch).toHaveBeenCalledOnce();
  });
});

describe("fetchWorkspaceDownloadWithRetry", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("returns immediately on success", async () => {
    const okResponse = { ok: true, status: 200, headers: new Headers() };
    vi.mocked(fetch).mockResolvedValue(okResponse as unknown as Response);

    const result = await fetchWorkspaceDownloadWithRetry(
      "https://backend/file",
      {},
      2,
      0,
    );
    expect(result).toBe(okResponse);
    expect(fetch).toHaveBeenCalledOnce();
  });

  it("returns immediately on non-transient error without retrying", async () => {
    const notFound = { ok: false, status: 404, headers: new Headers() };
    vi.mocked(fetch).mockResolvedValue(notFound as unknown as Response);

    const result = await fetchWorkspaceDownloadWithRetry(
      "https://backend/file",
      {},
      2,
      0,
    );
    expect(result.status).toBe(404);
    expect(fetch).toHaveBeenCalledOnce();
  });

  it("retries on transient 502 and succeeds", async () => {
    const bad = { ok: false, status: 502, headers: new Headers() };
    const ok = { ok: true, status: 200, headers: new Headers() };
    vi.mocked(fetch)
      .mockResolvedValueOnce(bad as unknown as Response)
      .mockResolvedValueOnce(ok as unknown as Response);

    const result = await fetchWorkspaceDownloadWithRetry(
      "https://backend/file",
      {},
      2,
      0,
    );
    expect(result).toBe(ok);
    expect(fetch).toHaveBeenCalledTimes(2);
  });

  it("returns last transient response after exhausting retries", async () => {
    const bad = { ok: false, status: 503, headers: new Headers() };
    vi.mocked(fetch).mockResolvedValue(bad as unknown as Response);

    const result = await fetchWorkspaceDownloadWithRetry(
      "https://backend/file",
      {},
      2,
      0,
    );
    expect(result.status).toBe(503);
    expect(fetch).toHaveBeenCalledTimes(3);
  });

  it("retries on network error and throws after exhausting retries", async () => {
    vi.mocked(fetch).mockRejectedValue(new Error("Connection reset"));

    await expect(
      fetchWorkspaceDownloadWithRetry("https://backend/file", {}, 1, 0),
    ).rejects.toThrow("Connection reset");
    expect(fetch).toHaveBeenCalledTimes(2);
  });
});
