import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import React from "react";
import {
  getCachedArtifactContent,
  clearContentCache,
} from "../useArtifactContent";
import { ArtifactContent } from "../ArtifactContent";
import type { ArtifactRef } from "../../../../store";
import type { ArtifactClassification } from "../../helpers";

vi.mock("@/components/contextual/OutputRenderers", () => ({
  globalRegistry: {
    getRenderer: vi.fn().mockReturnValue({
      render: vi.fn(() => React.createElement("div", null, "rendered")),
    }),
  },
}));

vi.mock(
  "@/components/contextual/OutputRenderers/renderers/CodeRenderer",
  () => ({
    codeRenderer: {
      render: vi.fn(() => React.createElement("div", null, "code")),
    },
  }),
);

vi.mock("../ArtifactReactPreview", () => ({
  ArtifactReactPreview: vi.fn(() =>
    React.createElement("div", { "data-testid": "react-preview" }),
  ),
}));

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
    icon: vi.fn(() => null) as unknown as ArtifactClassification["icon"],
    label: "Text",
    openable: true,
    hasSourceToggle: false,
    ...overrides,
  };
}

describe("useArtifactContent helpers (module-level cache)", () => {
  beforeEach(() => {
    clearContentCache();
  });

  afterEach(() => {
    cleanup();
    clearContentCache();
    vi.unstubAllGlobals();
  });

  it("populates the module cache so getCachedArtifactContent can read it from outside the panel", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        text: () => Promise.resolve("file content here"),
      }),
    );

    const artifact = makeArtifact({ id: "cache-test" });

    render(
      React.createElement(ArtifactContent, {
        artifact,
        isSourceView: false,
        classification: makeClassification({ type: "text" }),
      }),
    );

    await screen.findByText("rendered");
    expect(getCachedArtifactContent("cache-test")).toBe("file content here");
  });

  it("evicts the oldest cache entry once more than 12 artifacts have been loaded (LRU)", async () => {
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

    for (let i = 0; i < 12; i++) {
      const artifact = makeArtifact({
        id: `lru-${i}`,
        sourceUrl: `/api/proxy/api/workspace/files/lru-${i}/download`,
      });
      const { unmount } = render(
        React.createElement(ArtifactContent, {
          artifact,
          isSourceView: false,
          classification,
        }),
      );
      await screen.findByText("rendered");
      unmount();
    }

    expect(getCachedArtifactContent("lru-0")).toBe("content-lru-0");
    expect(getCachedArtifactContent("lru-11")).toBe("content-lru-11");

    const artifact13 = makeArtifact({
      id: "lru-12",
      sourceUrl: "/api/proxy/api/workspace/files/lru-12/download",
    });
    const { unmount } = render(
      React.createElement(ArtifactContent, {
        artifact: artifact13,
        isSourceView: false,
        classification,
      }),
    );
    await screen.findByText("rendered");
    unmount();

    expect(getCachedArtifactContent("lru-0")).toBeUndefined();
    expect(getCachedArtifactContent("lru-1")).toBe("content-lru-1");
    expect(getCachedArtifactContent("lru-12")).toBe("content-lru-12");
  });
});
