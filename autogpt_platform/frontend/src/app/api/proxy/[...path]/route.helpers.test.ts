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
  const VALID_UUID = "550e8400-e29b-41d4-a716-446655440000";
  const VALID_UUID_2 = "6ba7b810-9dad-11d1-80b4-00c04fd430c8";

  it("matches api/workspace/files/{uuid}/download pattern", () => {
    expect(
      isWorkspaceDownloadRequest([
        "api",
        "workspace",
        "files",
        VALID_UUID,
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
        VALID_UUID,
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
        VALID_UUID,
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
        VALID_UUID,
        "metadata",
      ]),
    ).toBe(false);
  });

  it("matches api/public/shared/{uuid}/files/{uuid}/download pattern", () => {
    expect(
      isWorkspaceDownloadRequest([
        "api",
        "public",
        "shared",
        VALID_UUID,
        "files",
        VALID_UUID_2,
        "download",
      ]),
    ).toBe(true);
  });

  it("rejects public shared paths not ending with download", () => {
    expect(
      isWorkspaceDownloadRequest([
        "api",
        "public",
        "shared",
        VALID_UUID,
        "files",
        VALID_UUID_2,
        "metadata",
      ]),
    ).toBe(false);
  });

  it("rejects non-UUID file ID in workspace path", () => {
    expect(
      isWorkspaceDownloadRequest([
        "api",
        "workspace",
        "files",
        "not-a-uuid",
        "download",
      ]),
    ).toBe(false);
  });

  it("rejects non-UUID token in public share path", () => {
    expect(
      isWorkspaceDownloadRequest([
        "api",
        "public",
        "shared",
        "not-a-uuid",
        "files",
        VALID_UUID,
        "download",
      ]),
    ).toBe(false);
  });

  it("rejects non-UUID file ID in public share path", () => {
    expect(
      isWorkspaceDownloadRequest([
        "api",
        "public",
        "shared",
        VALID_UUID,
        "files",
        "not-a-uuid",
        "download",
      ]),
    ).toBe(false);
  });

  it("accepts uppercase hex in UUIDs", () => {
    expect(
      isWorkspaceDownloadRequest([
        "api",
        "workspace",
        "files",
        "550E8400-E29B-41D4-A716-446655440000",
        "download",
      ]),
    ).toBe(true);
  });

  describe("adversarial inputs", () => {
    it("rejects empty path", () => {
      expect(isWorkspaceDownloadRequest([])).toBe(false);
    });

    it("rejects single-segment path", () => {
      expect(isWorkspaceDownloadRequest(["download"])).toBe(false);
    });

    it("rejects path traversal in file ID segment", () => {
      expect(
        isWorkspaceDownloadRequest([
          "api",
          "workspace",
          "files",
          "../../etc/passwd",
          "download",
        ]),
      ).toBe(false);
    });

    it("rejects path traversal in token segment", () => {
      expect(
        isWorkspaceDownloadRequest([
          "api",
          "public",
          "shared",
          "../../etc/passwd",
          "files",
          VALID_UUID,
          "download",
        ]),
      ).toBe(false);
    });

    it("rejects path traversal replacing fixed segments", () => {
      expect(
        isWorkspaceDownloadRequest([
          "api",
          "..",
          "files",
          VALID_UUID,
          "download",
        ]),
      ).toBe(false);
      expect(
        isWorkspaceDownloadRequest([
          "..",
          "workspace",
          "files",
          VALID_UUID,
          "download",
        ]),
      ).toBe(false);
    });

    it("rejects swapped workspace/public segments to confuse routing", () => {
      expect(
        isWorkspaceDownloadRequest([
          "api",
          "public",
          "files",
          VALID_UUID,
          "download",
        ]),
      ).toBe(false);
      expect(
        isWorkspaceDownloadRequest([
          "api",
          "workspace",
          "shared",
          VALID_UUID,
          "files",
          VALID_UUID_2,
          "download",
        ]),
      ).toBe(false);
    });

    it("rejects case variations on fixed segments", () => {
      expect(
        isWorkspaceDownloadRequest([
          "API",
          "workspace",
          "files",
          VALID_UUID,
          "download",
        ]),
      ).toBe(false);
      expect(
        isWorkspaceDownloadRequest([
          "api",
          "Workspace",
          "files",
          VALID_UUID,
          "download",
        ]),
      ).toBe(false);
      expect(
        isWorkspaceDownloadRequest([
          "api",
          "workspace",
          "files",
          VALID_UUID,
          "DOWNLOAD",
        ]),
      ).toBe(false);
      expect(
        isWorkspaceDownloadRequest([
          "api",
          "PUBLIC",
          "shared",
          VALID_UUID,
          "files",
          VALID_UUID_2,
          "download",
        ]),
      ).toBe(false);
      expect(
        isWorkspaceDownloadRequest([
          "api",
          "public",
          "SHARED",
          VALID_UUID,
          "files",
          VALID_UUID_2,
          "download",
        ]),
      ).toBe(false);
    });

    it("rejects empty string in fixed segments", () => {
      expect(
        isWorkspaceDownloadRequest([
          "api",
          "",
          "files",
          VALID_UUID,
          "download",
        ]),
      ).toBe(false);
    });

    it("rejects empty token in public share path", () => {
      expect(
        isWorkspaceDownloadRequest([
          "api",
          "public",
          "shared",
          "",
          "files",
          VALID_UUID,
          "download",
        ]),
      ).toBe(false);
    });

    it("rejects empty file ID in public share path", () => {
      expect(
        isWorkspaceDownloadRequest([
          "api",
          "public",
          "shared",
          VALID_UUID,
          "files",
          "",
          "download",
        ]),
      ).toBe(false);
    });

    it("rejects empty file ID in workspace path", () => {
      expect(
        isWorkspaceDownloadRequest([
          "api",
          "workspace",
          "files",
          "",
          "download",
        ]),
      ).toBe(false);
    });

    it("rejects UUID with null bytes injected", () => {
      expect(
        isWorkspaceDownloadRequest([
          "api",
          "workspace",
          "files",
          VALID_UUID + "\x00.jpg",
          "download",
        ]),
      ).toBe(false);
    });

    it("rejects UUID with trailing garbage", () => {
      expect(
        isWorkspaceDownloadRequest([
          "api",
          "workspace",
          "files",
          VALID_UUID + "-extra",
          "download",
        ]),
      ).toBe(false);
    });

    it("rejects UUID with leading garbage", () => {
      expect(
        isWorkspaceDownloadRequest([
          "api",
          "workspace",
          "files",
          "prefix-" + VALID_UUID,
          "download",
        ]),
      ).toBe(false);
    });

    it("rejects truncated UUIDs", () => {
      expect(
        isWorkspaceDownloadRequest([
          "api",
          "workspace",
          "files",
          "550e8400-e29b-41d4",
          "download",
        ]),
      ).toBe(false);
    });

    it("rejects UUID-length strings with wrong format", () => {
      // Right length (36 chars) but missing hyphens
      expect(
        isWorkspaceDownloadRequest([
          "api",
          "workspace",
          "files",
          "550e8400e29b41d4a716446655440000xxxx",
          "download",
        ]),
      ).toBe(false);
      // Hyphens in wrong positions
      expect(
        isWorkspaceDownloadRequest([
          "api",
          "workspace",
          "files",
          "550e-8400e29b-41d4a716-44665544-0000",
          "download",
        ]),
      ).toBe(false);
    });

    it("rejects UUID with non-hex characters", () => {
      expect(
        isWorkspaceDownloadRequest([
          "api",
          "workspace",
          "files",
          "550e8400-e29b-41d4-a716-44665544000g",
          "download",
        ]),
      ).toBe(false);
      expect(
        isWorkspaceDownloadRequest([
          "api",
          "workspace",
          "files",
          "550e8400-e29b-41d4-a716-44665544000!",
          "download",
        ]),
      ).toBe(false);
    });

    it("rejects SQL injection via ID segment", () => {
      expect(
        isWorkspaceDownloadRequest([
          "api",
          "workspace",
          "files",
          "'; DROP TABLE files;--",
          "download",
        ]),
      ).toBe(false);
    });

    it("rejects padded segments with whitespace", () => {
      expect(
        isWorkspaceDownloadRequest([
          "api",
          " workspace",
          "files",
          VALID_UUID,
          "download",
        ]),
      ).toBe(false);
      expect(
        isWorkspaceDownloadRequest([
          "api",
          "workspace ",
          "files",
          VALID_UUID,
          "download",
        ]),
      ).toBe(false);
      expect(
        isWorkspaceDownloadRequest([
          "api",
          "workspace",
          "files",
          " " + VALID_UUID,
          "download",
        ]),
      ).toBe(false);
    });

    it("rejects extra trailing segments after download", () => {
      expect(
        isWorkspaceDownloadRequest([
          "api",
          "workspace",
          "files",
          VALID_UUID,
          "download",
          "",
        ]),
      ).toBe(false);
      expect(
        isWorkspaceDownloadRequest([
          "api",
          "public",
          "shared",
          VALID_UUID,
          "files",
          VALID_UUID_2,
          "download",
          "extra",
        ]),
      ).toBe(false);
    });

    it("rejects extra leading segments before api", () => {
      expect(
        isWorkspaceDownloadRequest([
          "prefix",
          "api",
          "workspace",
          "files",
          VALID_UUID,
          "download",
        ]),
      ).toBe(false);
      expect(
        isWorkspaceDownloadRequest([
          "",
          "api",
          "public",
          "shared",
          VALID_UUID,
          "files",
          VALID_UUID_2,
          "download",
        ]),
      ).toBe(false);
    });

    it("rejects URL-encoded segment lookalikes", () => {
      expect(
        isWorkspaceDownloadRequest([
          "api",
          "workspace%2Ffiles",
          VALID_UUID,
          "download",
        ]),
      ).toBe(false);
      expect(
        isWorkspaceDownloadRequest([
          "api",
          "public%2Fshared",
          VALID_UUID,
          "files",
          VALID_UUID_2,
          "download",
        ]),
      ).toBe(false);
    });

    it("rejects unicode homoglyph substitutions in fixed segments", () => {
      // Cyrillic 'а' (U+0430) instead of Latin 'a'
      expect(
        isWorkspaceDownloadRequest([
          "\u0430pi",
          "workspace",
          "files",
          VALID_UUID,
          "download",
        ]),
      ).toBe(false);
      // Fullwidth 'ａ' (U+FF41)
      expect(
        isWorkspaceDownloadRequest([
          "\uff41pi",
          "workspace",
          "files",
          VALID_UUID,
          "download",
        ]),
      ).toBe(false);
    });

    it("rejects hybrid path mixing workspace and public patterns", () => {
      // 5-segment but with public prefix
      expect(
        isWorkspaceDownloadRequest([
          "api",
          "public",
          "shared",
          VALID_UUID,
          "download",
        ]),
      ).toBe(false);
      // 7-segment but with workspace prefix
      expect(
        isWorkspaceDownloadRequest([
          "api",
          "workspace",
          "files",
          VALID_UUID,
          "files",
          VALID_UUID_2,
          "download",
        ]),
      ).toBe(false);
    });

    it("rejects download appearing in non-terminal position", () => {
      expect(
        isWorkspaceDownloadRequest([
          "api",
          "workspace",
          "download",
          "files",
          VALID_UUID,
        ]),
      ).toBe(false);
      expect(
        isWorkspaceDownloadRequest([
          "api",
          "public",
          "shared",
          "download",
          "files",
          VALID_UUID,
          "extra",
        ]),
      ).toBe(false);
    });

    it("rejects prototype pollution segment names as IDs", () => {
      expect(
        isWorkspaceDownloadRequest([
          "api",
          "workspace",
          "files",
          "__proto__",
          "download",
        ]),
      ).toBe(false);
      expect(
        isWorkspaceDownloadRequest([
          "api",
          "workspace",
          "files",
          "constructor",
          "download",
        ]),
      ).toBe(false);
    });

    it("rejects very long path segments (DoS vector)", () => {
      const longId = "a".repeat(10000);
      expect(
        isWorkspaceDownloadRequest([
          "api",
          "workspace",
          "files",
          longId,
          "download",
        ]),
      ).toBe(false);
    });

    it("rejects UUID with embedded path separators", () => {
      expect(
        isWorkspaceDownloadRequest([
          "api",
          "workspace",
          "files",
          "550e8400/e29b-41d4-a716-446655440000",
          "download",
        ]),
      ).toBe(false);
    });

    it("rejects UUID-shaped strings with unicode hyphens", () => {
      // EN DASH (U+2013) instead of HYPHEN-MINUS
      expect(
        isWorkspaceDownloadRequest([
          "api",
          "workspace",
          "files",
          "550e8400\u2013e29b\u201341d4\u2013a716\u2013446655440000",
          "download",
        ]),
      ).toBe(false);
    });

    it("rejects SSRF-style payloads in ID position", () => {
      expect(
        isWorkspaceDownloadRequest([
          "api",
          "workspace",
          "files",
          "http://169.254.169.254",
          "download",
        ]),
      ).toBe(false);
    });
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
