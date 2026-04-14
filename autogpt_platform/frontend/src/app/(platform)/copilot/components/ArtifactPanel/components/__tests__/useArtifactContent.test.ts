import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { renderHook, waitFor, act } from "@testing-library/react";
import {
  useArtifactContent,
  getCachedArtifactContent,
  clearContentCache,
} from "../useArtifactContent";
import type { ArtifactRef } from "../../../../store";
import type { ArtifactClassification } from "../../helpers";

function makeArtifact(overrides?: Partial<ArtifactRef>): ArtifactRef {
  return {
    id: "file-001",
    title: "test.txt",
    mimeType: "text/plain",
    sourceUrl: "/api/proxy/api/workspace/files/file-001/download",
    origin: "agent",
    ...overrides,
  };
}

function makeClassification(
  overrides?: Partial<ArtifactClassification>,
): ArtifactClassification {
  return {
    type: "text",
    icon: vi.fn() as any,
    label: "Text",
    openable: true,
    hasSourceToggle: false,
    ...overrides,
  };
}

describe("useArtifactContent", () => {
  beforeEach(() => {
    clearContentCache();
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        text: () => Promise.resolve("file content here"),
        blob: () => Promise.resolve(new Blob(["pdf bytes"])),
      }),
    );
  });

  afterEach(() => {
    clearContentCache();
    vi.restoreAllMocks();
  });

  it("fetches text content for text artifacts", async () => {
    const artifact = makeArtifact();
    const classification = makeClassification({ type: "text" });

    const { result } = renderHook(() =>
      useArtifactContent(artifact, classification),
    );

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.content).toBe("file content here");
    expect(result.current.error).toBeNull();
  });

  it("skips fetch for image artifacts", async () => {
    const artifact = makeArtifact({ mimeType: "image/png" });
    const classification = makeClassification({ type: "image" });

    const { result } = renderHook(() =>
      useArtifactContent(artifact, classification),
    );

    expect(result.current.isLoading).toBe(false);
    expect(result.current.content).toBeNull();
    expect(fetch).not.toHaveBeenCalled();
  });

  it("creates blob URL for PDF artifacts", async () => {
    const artifact = makeArtifact({ mimeType: "application/pdf" });
    const classification = makeClassification({ type: "pdf" });

    const { result } = renderHook(() =>
      useArtifactContent(artifact, classification),
    );

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.pdfUrl).toMatch(/^blob:/);
  });

  it("sets error on fetch failure", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 404,
        text: () => Promise.resolve("Not found"),
      }),
    );

    // Use a unique ID to avoid hitting the module-level content cache
    const artifact = makeArtifact({ id: "error-test-unique" });
    const classification = makeClassification({ type: "text" });

    const { result } = renderHook(() =>
      useArtifactContent(artifact, classification),
    );

    await waitFor(
      () => {
        expect(result.current.error).toBeTruthy();
      },
      { timeout: 2500 },
    );

    expect(result.current.error).toContain("404");
    expect(result.current.content).toBeNull();
  });

  it("caches fetched content and exposes via getCachedArtifactContent", async () => {
    const artifact = makeArtifact({ id: "cache-test" });
    const classification = makeClassification({ type: "text" });

    const { result } = renderHook(() =>
      useArtifactContent(artifact, classification),
    );

    await waitFor(() => {
      expect(result.current.content).toBe("file content here");
    });

    expect(getCachedArtifactContent("cache-test")).toBe("file content here");
  });

  it("sets error on fetch failure for HTML artifacts (stale artifact)", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 404,
        text: () => Promise.resolve("Not found"),
      }),
    );

    const artifact = makeArtifact({ id: "stale-html-artifact" });
    const classification = makeClassification({ type: "html" });

    const { result } = renderHook(() =>
      useArtifactContent(artifact, classification),
    );

    await waitFor(
      () => {
        expect(result.current.error).toBeTruthy();
      },
      { timeout: 2500 },
    );

    expect(result.current.error).toContain("404");
    expect(result.current.content).toBeNull();
  });

  it("sets error on network failure", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockRejectedValue(new Error("Network error")),
    );

    const artifact = makeArtifact({ id: "network-error-artifact" });
    const classification = makeClassification({ type: "html" });

    const { result } = renderHook(() =>
      useArtifactContent(artifact, classification),
    );

    await waitFor(
      () => {
        expect(result.current.error).toBeTruthy();
      },
      { timeout: 2500 },
    );

    expect(result.current.error).toContain("Network error");
    expect(result.current.content).toBeNull();
  });

  it("retries transient HTML fetch failures before surfacing an error", async () => {
    let callCount = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation(() => {
        callCount++;
        if (callCount < 3) {
          return Promise.resolve({
            ok: false,
            status: 503,
            headers: {
              get: () => "application/json",
            },
            json: () => Promise.resolve({ detail: "temporary upstream error" }),
          });
        }

        return Promise.resolve({
          ok: true,
          text: () => Promise.resolve("<html>ok now</html>"),
        });
      }),
    );

    const artifact = makeArtifact({ id: "transient-html-retry" });
    const classification = makeClassification({ type: "html" });

    const { result } = renderHook(() =>
      useArtifactContent(artifact, classification),
    );

    await waitFor(
      () => {
        expect(result.current.content).toBe("<html>ok now</html>");
      },
      { timeout: 2500 },
    );

    expect(callCount).toBe(3);
    expect(result.current.error).toBeNull();
  });

  it("surfaces backend error detail from JSON responses", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 404,
        headers: {
          get: () => "application/json",
        },
        json: () => Promise.resolve({ detail: "File not found" }),
      }),
    );

    const artifact = makeArtifact({ id: "json-error-detail" });
    const classification = makeClassification({ type: "html" });

    const { result } = renderHook(() =>
      useArtifactContent(artifact, classification),
    );

    await waitFor(
      () => {
        expect(result.current.error).toBeTruthy();
      },
      { timeout: 2500 },
    );

    expect(result.current.error).toContain("404");
    expect(result.current.error).toContain("File not found");
  });

  it("retry after 404 on HTML artifact clears cache and re-fetches", async () => {
    let callCount = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation(() => {
        callCount++;
        if (callCount === 1) {
          return Promise.resolve({
            ok: false,
            status: 404,
            text: () => Promise.resolve("Not found"),
          });
        }
        return Promise.resolve({
          ok: true,
          text: () => Promise.resolve("<html>recovered</html>"),
        });
      }),
    );

    const artifact = makeArtifact({ id: "retry-html-artifact" });
    const classification = makeClassification({ type: "html" });

    const { result } = renderHook(() =>
      useArtifactContent(artifact, classification),
    );

    await waitFor(() => {
      expect(result.current.error).toBeTruthy();
    });

    act(() => {
      result.current.retry();
    });

    await waitFor(
      () => {
        expect(result.current.content).toBe("<html>recovered</html>");
      },
      { timeout: 2500 },
    );

    expect(result.current.error).toBeNull();
  });

  it("retry clears cache and re-fetches", async () => {
    let callCount = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation(() => {
        callCount++;
        return Promise.resolve({
          ok: true,
          text: () => Promise.resolve(`response ${callCount}`),
        });
      }),
    );

    const artifact = makeArtifact({ id: "retry-test" });
    const classification = makeClassification({ type: "text" });

    const { result } = renderHook(() =>
      useArtifactContent(artifact, classification),
    );

    await waitFor(() => {
      expect(result.current.content).toBe("response 1");
    });

    act(() => {
      result.current.retry();
    });

    await waitFor(() => {
      expect(result.current.content).toBe("response 2");
    });
  });

  // ── Non-transient errors ──────────────────────────────────────────

  it("rejects immediately on 403 without retrying", async () => {
    let callCount = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation(() => {
        callCount++;
        return Promise.resolve({
          ok: false,
          status: 403,
          text: () => Promise.resolve("Forbidden"),
        });
      }),
    );

    const artifact = makeArtifact({ id: "forbidden-no-retry" });
    const classification = makeClassification({ type: "text" });

    const { result } = renderHook(() =>
      useArtifactContent(artifact, classification),
    );

    await waitFor(
      () => {
        expect(result.current.error).toBeTruthy();
      },
      { timeout: 2500 },
    );

    expect(callCount).toBe(1);
    expect(result.current.error).toContain("403");
  });

  // ── Video skip-fetch ──────────────────────────────────────────────

  it("skips fetch for video artifacts (like image)", async () => {
    const artifact = makeArtifact({
      id: "video-skip",
      mimeType: "video/mp4",
    });
    const classification = makeClassification({ type: "video" });

    const { result } = renderHook(() =>
      useArtifactContent(artifact, classification),
    );

    expect(result.current.isLoading).toBe(false);
    expect(result.current.content).toBeNull();
    expect(result.current.pdfUrl).toBeNull();
    expect(fetch).not.toHaveBeenCalled();
  });

  // ── PDF error paths ───────────────────────────────────────────────

  it("sets error on PDF fetch failure (non-2xx)", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 500,
        text: () => Promise.resolve("Server Error"),
      }),
    );

    const artifact = makeArtifact({ id: "pdf-error" });
    const classification = makeClassification({ type: "pdf" });

    const { result } = renderHook(() =>
      useArtifactContent(artifact, classification),
    );

    await waitFor(
      () => {
        expect(result.current.error).toBeTruthy();
      },
      { timeout: 2500 },
    );

    expect(result.current.error).toContain("500");
    expect(result.current.pdfUrl).toBeNull();
  });

  it("sets error on PDF network failure", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockRejectedValue(new Error("PDF network failure")),
    );

    const artifact = makeArtifact({ id: "pdf-network-error" });
    const classification = makeClassification({ type: "pdf" });

    const { result } = renderHook(() =>
      useArtifactContent(artifact, classification),
    );

    await waitFor(
      () => {
        expect(result.current.error).toBeTruthy();
      },
      { timeout: 2500 },
    );

    expect(result.current.error).toContain("PDF network failure");
    expect(result.current.pdfUrl).toBeNull();
  });

  // ── LRU cache eviction ────────────────────────────────────────────

  it("evicts oldest entry when cache exceeds 12 items", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation((url: string) => {
        const fileId = url.match(/files\/([^/]+)\/download/)?.[1] ?? "unknown";
        return Promise.resolve({
          ok: true,
          text: () => Promise.resolve(`content-${fileId}`),
        });
      }),
    );

    const classification = makeClassification({ type: "text" });

    // Fill the cache with 12 entries (cache max = 12)
    for (let i = 0; i < 12; i++) {
      const artifact = makeArtifact({
        id: `lru-${i}`,
        sourceUrl: `/api/proxy/api/workspace/files/lru-${i}/download`,
      });
      const { result } = renderHook(() =>
        useArtifactContent(artifact, classification),
      );
      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });
    }

    // All 12 should be cached
    expect(getCachedArtifactContent("lru-0")).toBe("content-lru-0");
    expect(getCachedArtifactContent("lru-11")).toBe("content-lru-11");

    // Adding a 13th should evict lru-0 (the oldest)
    const artifact13 = makeArtifact({
      id: "lru-12",
      sourceUrl: "/api/proxy/api/workspace/files/lru-12/download",
    });
    const { result: result13 } = renderHook(() =>
      useArtifactContent(artifact13, classification),
    );
    await waitFor(() => {
      expect(result13.current.isLoading).toBe(false);
    });

    expect(getCachedArtifactContent("lru-0")).toBeUndefined();
    expect(getCachedArtifactContent("lru-1")).toBe("content-lru-1");
    expect(getCachedArtifactContent("lru-12")).toBe("content-lru-12");
  });
});
