import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { renderHook, waitFor, act } from "@testing-library/react";
import {
  useArtifactContent,
  getCachedArtifactContent,
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

    await waitFor(() => {
      expect(result.current.error).toBeTruthy();
    });

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
});
