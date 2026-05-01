import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { ArtifactContent } from "../ArtifactContent";
import type { ArtifactRef } from "../../../../store";
import { classifyArtifact, type ArtifactClassification } from "../../helpers";
import { globalRegistry } from "@/components/contextual/OutputRenderers";
import { codeRenderer } from "@/components/contextual/OutputRenderers/renderers/CodeRenderer";
import { ArtifactReactPreview } from "../ArtifactReactPreview";

// Mock the renderers so we don't pull in the full renderer dependency tree
vi.mock("@/components/contextual/OutputRenderers", () => ({
  globalRegistry: {
    getRenderer: vi.fn().mockReturnValue({
      render: vi.fn((_val: unknown, meta: Record<string, unknown>) => (
        <div data-testid="global-renderer">
          rendered:{String(meta?.mimeType ?? "unknown")}
        </div>
      )),
    }),
  },
}));

vi.mock(
  "@/components/contextual/OutputRenderers/renderers/CodeRenderer",
  () => ({
    codeRenderer: {
      render: vi.fn((content: string) => (
        <div data-testid="code-renderer">{content}</div>
      )),
    },
  }),
);

vi.mock("../ArtifactReactPreview", () => ({
  ArtifactReactPreview: vi.fn(
    ({ source, title }: { source: string; title: string }) => (
      <div data-testid="react-preview" data-title={title}>
        {source}
      </div>
    ),
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

describe("ArtifactContent", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        text: () => Promise.resolve("file content here"),
        blob: () => Promise.resolve(new Blob(["content"])),
      }),
    );
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  // ── Image ─────────────────────────────────────────────────────────

  it("renders image artifact as img tag with loading skeleton", () => {
    const artifact = makeArtifact({
      id: "img-001",
      title: "photo.png",
      mimeType: "image/png",
      sourceUrl: "/api/proxy/api/workspace/files/img-001/download",
    });
    const classification = makeClassification({ type: "image" });

    const { container } = render(
      <ArtifactContent
        artifact={artifact}
        isSourceView={false}
        classification={classification}
      />,
    );

    const img = container.querySelector("img");
    expect(img).toBeTruthy();
    expect(img?.getAttribute("src")).toBe(
      "/api/proxy/api/workspace/files/img-001/download",
    );
    expect(fetch).not.toHaveBeenCalled();
  });

  it("image artifact shows loading skeleton before image loads", () => {
    const artifact = makeArtifact({
      id: "img-skeleton",
      title: "photo.png",
      mimeType: "image/png",
      sourceUrl: "/api/proxy/api/workspace/files/img-skeleton/download",
    });
    const classification = makeClassification({ type: "image" });

    const { container } = render(
      <ArtifactContent
        artifact={artifact}
        isSourceView={false}
        classification={classification}
      />,
    );

    // Skeleton uses animate-pulse class
    const skeleton = container.querySelector('[class*="animate-pulse"]');
    expect(skeleton).toBeTruthy();
  });

  it("image artifact shows error state when image fails to load", () => {
    const artifact = makeArtifact({
      id: "img-error",
      title: "broken.png",
      mimeType: "image/png",
      sourceUrl: "/api/proxy/api/workspace/files/img-error/download",
    });
    const classification = makeClassification({ type: "image" });

    const { container } = render(
      <ArtifactContent
        artifact={artifact}
        isSourceView={false}
        classification={classification}
      />,
    );

    const img = container.querySelector("img");
    expect(img).toBeTruthy();
    fireEvent.error(img!);

    const errorAlert = screen.queryByRole("alert");
    expect(errorAlert).toBeTruthy();
    expect(screen.queryByText("Failed to load image")).toBeTruthy();
  });

  it("image retry resets error and re-shows img", async () => {
    const artifact = makeArtifact({
      id: "img-retry",
      title: "retry.png",
      mimeType: "image/png",
      sourceUrl: "/api/proxy/api/workspace/files/img-retry/download",
    });
    const classification = makeClassification({ type: "image" });

    const { container } = render(
      <ArtifactContent
        artifact={artifact}
        isSourceView={false}
        classification={classification}
      />,
    );

    const img = container.querySelector("img");
    fireEvent.error(img!);

    // Should show error state
    await waitFor(() => {
      expect(screen.queryByText("Failed to load image")).toBeTruthy();
    });

    // Click "Try again"
    fireEvent.click(screen.getByRole("button", { name: /try again/i }));

    // Error should be cleared, img should reappear
    await waitFor(() => {
      expect(screen.queryByText("Failed to load image")).toBeNull();
      expect(container.querySelector("img")).toBeTruthy();
    });
  });

  // SECRT-2221 integration: the classification-level fix (hi-res PNGs stop
  // being size-gated) only matters if the end-to-end rendering pipeline
  // actually reaches the <img> path. Pass in the real classifyArtifact
  // result for a 25 MB .png and assert the panel renders an img element
  // rather than routing to the download-only surface.
  it("renders a 25 MB PNG through the <img> path, not download-only (SECRT-2221)", () => {
    const artifact = makeArtifact({
      id: "hires-png-001",
      title: "poster.png",
      mimeType: "image/png",
      sourceUrl: "/api/proxy/api/workspace/files/hires-png-001/download",
      sizeBytes: 25 * 1024 * 1024,
    });
    const classification = classifyArtifact(
      artifact.mimeType,
      artifact.title,
      artifact.sizeBytes,
    );
    expect(classification.type).toBe("image");
    expect(classification.openable).toBe(true);

    const { container } = render(
      <ArtifactContent
        artifact={artifact}
        isSourceView={false}
        classification={classification}
      />,
    );

    const img = container.querySelector("img");
    expect(img).toBeTruthy();
    expect(img?.getAttribute("src")).toBe(artifact.sourceUrl);
  });

  // SECRT-2221: image retry appends a cache-busting query so the browser
  // can't reuse a previously-failed response. Without this, a transient
  // 5xx that gets negative-cached keeps showing "Failed to load image" no
  // matter how many times the user clicks Try again.
  it("image retry appends a cache-busting query so the browser re-fetches (SECRT-2221)", async () => {
    const artifact = makeArtifact({
      id: "img-cachebust",
      title: "hires.png",
      mimeType: "image/png",
      sourceUrl: "/api/proxy/api/workspace/files/img-cachebust/download",
    });
    const classification = makeClassification({ type: "image" });

    const { container } = render(
      <ArtifactContent
        artifact={artifact}
        isSourceView={false}
        classification={classification}
      />,
    );

    const firstImg = container.querySelector("img");
    const firstSrc = firstImg?.getAttribute("src");
    expect(firstSrc).toBe(artifact.sourceUrl);

    fireEvent.error(firstImg!);
    await waitFor(() => {
      expect(screen.queryByText("Failed to load image")).toBeTruthy();
    });
    fireEvent.click(screen.getByRole("button", { name: /try again/i }));

    await waitFor(() => {
      const nextImg = container.querySelector("img");
      const nextSrc = nextImg?.getAttribute("src") ?? "";
      expect(nextSrc).not.toBe(firstSrc);
      expect(nextSrc.startsWith(artifact.sourceUrl)).toBe(true);
      // Assert the specific cache-bust contract, not just that the URL
      // changed — guards against accidental rewrites that drop the key.
      expect(nextSrc).toContain("_retry=");
    });
  });

  // ── Video ─────────────────────────────────────────────────────────

  it("renders video artifact with video tag and controls", () => {
    const artifact = makeArtifact({
      id: "vid-001",
      title: "clip.mp4",
      mimeType: "video/mp4",
      sourceUrl: "/api/proxy/api/workspace/files/vid-001/download",
    });
    const classification = makeClassification({ type: "video" });

    const { container } = render(
      <ArtifactContent
        artifact={artifact}
        isSourceView={false}
        classification={classification}
      />,
    );

    const video = container.querySelector("video");
    expect(video).toBeTruthy();
    expect(video?.hasAttribute("controls")).toBe(true);
    expect(video?.getAttribute("src")).toBe(
      "/api/proxy/api/workspace/files/vid-001/download",
    );
    expect(fetch).not.toHaveBeenCalled();
  });

  it("video shows loading skeleton before metadata loads", () => {
    const artifact = makeArtifact({
      id: "vid-skel",
      title: "clip.mp4",
      mimeType: "video/mp4",
      sourceUrl: "/api/proxy/api/workspace/files/vid-skel/download",
    });
    const classification = makeClassification({ type: "video" });

    const { container } = render(
      <ArtifactContent
        artifact={artifact}
        isSourceView={false}
        classification={classification}
      />,
    );

    const skeleton = container.querySelector('[class*="animate-pulse"]');
    expect(skeleton).toBeTruthy();

    // After metadata loads, skeleton should disappear
    const video = container.querySelector("video");
    fireEvent.loadedMetadata(video!);

    expect(container.querySelector('[class*="animate-pulse"]')).toBeNull();
  });

  it("video shows error state when video fails to load", () => {
    const artifact = makeArtifact({
      id: "vid-error",
      title: "broken.mp4",
      mimeType: "video/mp4",
      sourceUrl: "/api/proxy/api/workspace/files/vid-error/download",
    });
    const classification = makeClassification({ type: "video" });

    const { container } = render(
      <ArtifactContent
        artifact={artifact}
        isSourceView={false}
        classification={classification}
      />,
    );

    const video = container.querySelector("video");
    expect(video).toBeTruthy();
    fireEvent.error(video!);

    const errorAlert = screen.queryByRole("alert");
    expect(errorAlert).toBeTruthy();
    expect(screen.queryByText("Failed to load video")).toBeTruthy();
  });

  it("video retry resets error and re-shows video", async () => {
    const artifact = makeArtifact({
      id: "vid-retry",
      title: "retry.mp4",
      mimeType: "video/mp4",
      sourceUrl: "/api/proxy/api/workspace/files/vid-retry/download",
    });
    const classification = makeClassification({ type: "video" });

    const { container } = render(
      <ArtifactContent
        artifact={artifact}
        isSourceView={false}
        classification={classification}
      />,
    );

    const video = container.querySelector("video");
    fireEvent.error(video!);

    await waitFor(() => {
      expect(screen.queryByText("Failed to load video")).toBeTruthy();
    });

    fireEvent.click(screen.getByRole("button", { name: /try again/i }));

    await waitFor(() => {
      expect(screen.queryByText("Failed to load video")).toBeNull();
      expect(container.querySelector("video")).toBeTruthy();
    });
  });

  // ── PDF ───────────────────────────────────────────────────────────

  it("renders PDF artifact in unsandboxed iframe with blob URL", async () => {
    const blobUrl = "blob:http://localhost/fake-pdf-blob";
    vi.stubGlobal(
      "URL",
      Object.assign(URL, {
        createObjectURL: vi.fn().mockReturnValue(blobUrl),
        revokeObjectURL: vi.fn(),
      }),
    );

    const artifact = makeArtifact({
      id: "pdf-render",
      title: "report.pdf",
      mimeType: "application/pdf",
      sourceUrl: "/api/proxy/api/workspace/files/pdf-render/download",
    });
    const classification = makeClassification({ type: "pdf" });

    const { container } = render(
      <ArtifactContent
        artifact={artifact}
        isSourceView={false}
        classification={classification}
      />,
    );

    await waitFor(() => {
      const iframe = container.querySelector("iframe");
      expect(iframe).toBeTruthy();
      expect(iframe?.getAttribute("src")).toBe(blobUrl);
      // No sandbox attribute — Chrome blocks PDF in sandboxed iframes
      expect(iframe?.hasAttribute("sandbox")).toBe(false);
    });
  });

  // ── Fetch error ───────────────────────────────────────────────────

  it("shows error state with retry button on fetch failure", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 404,
        text: () => Promise.resolve("Not found"),
      }),
    );

    const artifact = makeArtifact({ id: "error-content-test" });
    const classification = makeClassification({ type: "html" });

    render(
      <ArtifactContent
        artifact={artifact}
        isSourceView={false}
        classification={classification}
      />,
    );

    const errorText = await screen.findByText("Failed to load content");
    expect(errorText).toBeTruthy();

    const retryButtons = screen.getAllByRole("button", { name: /try again/i });
    expect(retryButtons.length).toBeGreaterThan(0);
  });

  // SECRT-2224: "try again doesn't do anything". The retry itself works — the
  // user's complaint is that there's no visible feedback when the same error
  // returns (e.g. a 404 for a deleted file). Clicking Try Again must flip the
  // UI into the loading skeleton immediately so the user can tell their click
  // registered, instead of the error UI re-flashing in place.
  it("clicking Try Again shows the loading skeleton before the next fetch settles (SECRT-2224)", async () => {
    let resolveSecond: (value: unknown) => void = () => {};
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
        return new Promise((resolve) => {
          resolveSecond = resolve;
        });
      }),
    );

    const artifact = makeArtifact({
      id: "retry-skeleton-001",
      title: "flaky.html",
      mimeType: "text/html",
    });
    const classification = makeClassification({ type: "html" });

    const { container } = render(
      <ArtifactContent
        artifact={artifact}
        isSourceView={false}
        classification={classification}
      />,
    );

    await screen.findByText("Failed to load content");
    fireEvent.click(screen.getByRole("button", { name: /try again/i }));

    // Before the second fetch resolves, the error must be gone and a skeleton
    // visible (animate-pulse is the Skeleton component's signature class).
    await waitFor(() => {
      expect(screen.queryByText("Failed to load content")).toBeNull();
      expect(container.querySelector('[class*="animate-pulse"]')).toBeTruthy();
    });

    // Let the second fetch complete and wait for the recovered render so
    // pending React updates can't leak into the next test.
    resolveSecond({
      ok: true,
      text: () => Promise.resolve("<html><body>ok</body></html>"),
    });
    await screen.findByTitle("flaky.html");
  });

  // SECRT-2224 end-to-end: Try Again actually recovers when the next fetch
  // succeeds. Covers the full click → re-fetch → iframe-render loop.
  it("clicking Try Again re-fetches and renders recovered HTML content (SECRT-2224)", async () => {
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
          text: () =>
            Promise.resolve(
              "<html><body><h1 id='ok'>recovered</h1></body></html>",
            ),
        });
      }),
    );

    const artifact = makeArtifact({
      id: "retry-recover-001",
      title: "flaky.html",
      mimeType: "text/html",
    });
    const classification = makeClassification({ type: "html" });

    const { container } = render(
      <ArtifactContent
        artifact={artifact}
        isSourceView={false}
        classification={classification}
      />,
    );

    await screen.findByText("Failed to load content");
    fireEvent.click(screen.getByRole("button", { name: /try again/i }));

    await waitFor(() => {
      const iframe = container.querySelector("iframe");
      expect(iframe).toBeTruthy();
      expect(iframe?.getAttribute("srcdoc")).toContain("recovered");
    });
    expect(screen.queryByText("Failed to load content")).toBeNull();
    expect(callCount).toBeGreaterThanOrEqual(2);
  });

  // ── HTML ──────────────────────────────────────────────────────────

  it("renders HTML content in sandboxed iframe", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        text: () =>
          Promise.resolve("<html><body><h1>Hello World</h1></body></html>"),
      }),
    );

    const artifact = makeArtifact({
      id: "html-001",
      title: "page.html",
      mimeType: "text/html",
    });
    const classification = makeClassification({ type: "html" });

    const { container } = render(
      <ArtifactContent
        artifact={artifact}
        isSourceView={false}
        classification={classification}
      />,
    );

    await screen.findByTitle("page.html");
    const iframe = container.querySelector("iframe");
    expect(iframe).toBeTruthy();
    expect(iframe?.getAttribute("sandbox")).toBe("allow-scripts");
  });

  it("injects the fragment-link interceptor into HTML artifact iframes (regression)", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        text: () =>
          Promise.resolve(
            '<html><head></head><body><a href="#x">x</a><div id="x">x</div></body></html>',
          ),
      }),
    );

    const { container } = render(
      <ArtifactContent
        artifact={makeArtifact({
          id: "html-frag",
          title: "page.html",
          mimeType: "text/html",
        })}
        isSourceView={false}
        classification={makeClassification({ type: "html" })}
      />,
    );

    await screen.findByTitle("page.html");
    const srcdoc = container.querySelector("iframe")?.getAttribute("srcdoc");
    expect(srcdoc).toBeTruthy();
    // Markers unique to FRAGMENT_LINK_INTERCEPTOR_SCRIPT — if any of these
    // disappear, the interceptor is no longer being injected and fragment
    // links will navigate the parent URL again.
    expect(srcdoc).toContain("__fragmentLinkInterceptor");
    expect(srcdoc).toContain('a[href^="#"]');
    expect(srcdoc).toContain("scrollIntoView");
  });

  // ── Source view ───────────────────────────────────────────────────

  it("renders source view as pre tag", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        text: () => Promise.resolve("source code here"),
      }),
    );

    const artifact = makeArtifact({ id: "source-view-test" });
    const classification = makeClassification({
      type: "html",
      hasSourceToggle: true,
    });

    const { container } = render(
      <ArtifactContent
        artifact={artifact}
        isSourceView={true}
        classification={classification}
      />,
    );

    await screen.findByText("source code here");
    const pre = container.querySelector("pre");
    expect(pre).toBeTruthy();
    expect(pre?.textContent).toBe("source code here");
  });

  // ── React ─────────────────────────────────────────────────────────

  it("renders react artifacts via ArtifactReactPreview", async () => {
    const jsxSource = "export default function App() { return <div>Hi</div>; }";
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        text: () => Promise.resolve(jsxSource),
      }),
    );

    const artifact = makeArtifact({
      id: "react-001",
      title: "App.tsx",
      mimeType: "text/tsx",
    });
    const classification = makeClassification({ type: "react" });

    render(
      <ArtifactContent
        artifact={artifact}
        isSourceView={false}
        classification={classification}
      />,
    );

    const preview = await screen.findByTestId("react-preview");
    expect(preview).toBeTruthy();
    expect(preview.textContent).toContain(jsxSource);
    expect(preview.getAttribute("data-title")).toBe("App.tsx");
  });

  it("routes a concrete props-based TSX artifact into ArtifactReactPreview", async () => {
    const jsxSource = `
      import React, { FC, useState } from "react";

      interface ArtifactFile {
        id: string;
        name: string;
        mimeType: string;
        url: string;
        sizeBytes: number;
      }

      interface Props {
        files: ArtifactFile[];
        onSelect: (file: ArtifactFile) => void;
      }

      export const previewProps: Props = {
        files: [
          {
            id: "1",
            name: "report.png",
            mimeType: "image/png",
            url: "/report.png",
            sizeBytes: 2048,
          },
        ],
        onSelect: () => {},
      };

      const ArtifactList: FC<Props> = ({ files, onSelect }) => {
        const [selected, setSelected] = useState<string | null>(null);

        const handleClick = (file: ArtifactFile) => {
          setSelected(file.id);
          onSelect(file);
        };

        return (
          <ul>
            {files.map((file) => (
              <li key={file.id} onClick={() => handleClick(file)}>
                <span>{selected === file.id ? "selected" : file.name}</span>
              </li>
            ))}
          </ul>
        );
      };

      export default ArtifactList;
    `;

    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        text: () => Promise.resolve(jsxSource),
      }),
    );

    const artifact = makeArtifact({
      id: "react-props-001",
      title: "ArtifactList.tsx",
      mimeType: "text/tsx",
    });
    const classification = classifyArtifact(artifact.mimeType, artifact.title);

    render(
      <ArtifactContent
        artifact={artifact}
        isSourceView={false}
        classification={classification}
      />,
    );

    const preview = await screen.findByTestId("react-preview");
    expect(preview.textContent).toContain("previewProps");
    expect(preview.getAttribute("data-title")).toBe("ArtifactList.tsx");
    expect(vi.mocked(ArtifactReactPreview).mock.calls[0]?.[0]).toEqual(
      expect.objectContaining({
        source: expect.stringContaining("export const previewProps"),
        title: "ArtifactList.tsx",
      }),
    );
  });

  // ── Code ──────────────────────────────────────────────────────────

  it("renders code artifacts via codeRenderer", async () => {
    const code = 'def hello():\n    print("hi")';
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        text: () => Promise.resolve(code),
      }),
    );

    const artifact = makeArtifact({
      id: "code-render-001",
      title: "script.py",
      mimeType: "text/x-python",
    });
    const classification = makeClassification({ type: "code" });

    render(
      <ArtifactContent
        artifact={artifact}
        isSourceView={false}
        classification={classification}
      />,
    );

    const rendered = await screen.findByTestId("code-renderer");
    expect(rendered).toBeTruthy();
    expect(rendered.textContent).toContain(code);
  });

  it.each([
    {
      filename: "events.jsonl",
      mimeType: "application/x-ndjson",
      content: '{"event":"start"}\n{"event":"finish"}',
    },
    {
      filename: ".env.local",
      mimeType: "text/plain",
      content: "OPENAI_API_KEY=test\nDEBUG=true",
    },
    {
      filename: "Dockerfile",
      mimeType: "text/plain",
      content: "FROM node:20\nRUN pnpm install",
    },
    {
      filename: "schema.graphql",
      mimeType: "text/plain",
      content: "type Query { viewer: User }",
    },
  ])(
    "renders concrete code artifact $filename through codeRenderer",
    async ({ filename, mimeType, content }) => {
      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue({
          ok: true,
          text: () => Promise.resolve(content),
        }),
      );

      const artifact = makeArtifact({
        id: `code-${filename}`,
        title: filename,
        mimeType,
      });
      const classification = classifyArtifact(
        artifact.mimeType,
        artifact.title,
      );

      render(
        <ArtifactContent
          artifact={artifact}
          isSourceView={false}
          classification={classification}
        />,
      );

      await screen.findByTestId("code-renderer");

      expect(classification.type).toBe("code");
      expect(vi.mocked(codeRenderer.render)).toHaveBeenCalledWith(
        content,
        expect.objectContaining({
          filename,
          mimeType,
          type: "code",
        }),
      );
    },
  );

  // ── JSON ──────────────────────────────────────────────────────────

  it("renders valid JSON via globalRegistry", async () => {
    const jsonContent = JSON.stringify({ key: "value" }, null, 2);
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        text: () => Promise.resolve(jsonContent),
      }),
    );

    const artifact = makeArtifact({
      id: "json-render-001",
      title: "data.json",
      mimeType: "application/json",
    });
    const classification = makeClassification({ type: "json" });

    render(
      <ArtifactContent
        artifact={artifact}
        isSourceView={false}
        classification={classification}
      />,
    );

    const rendered = await screen.findByTestId("global-renderer");
    expect(rendered).toBeTruthy();
    expect(rendered.textContent).toContain("application/json");
  });

  it("renders invalid JSON as fallback pre tag", async () => {
    const { globalRegistry } = await import(
      "@/components/contextual/OutputRenderers"
    );
    const originalImpl = vi
      .mocked(globalRegistry.getRenderer)
      .getMockImplementation();

    // For invalid JSON, JSON.parse throws, then the registry fallback
    // also returns null → falls through to <pre>
    vi.mocked(globalRegistry.getRenderer).mockReturnValue(null);

    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        text: () => Promise.resolve("{invalid json!!!"),
      }),
    );

    const artifact = makeArtifact({
      id: "json-invalid-001",
      title: "bad.json",
      mimeType: "application/json",
    });
    const classification = makeClassification({ type: "json" });

    const { container } = render(
      <ArtifactContent
        artifact={artifact}
        isSourceView={false}
        classification={classification}
      />,
    );

    await waitFor(() => {
      const pre = container.querySelector("pre");
      expect(pre).toBeTruthy();
      expect(pre?.textContent).toBe("{invalid json!!!");
    });

    // Restore
    if (originalImpl) {
      vi.mocked(globalRegistry.getRenderer).mockImplementation(originalImpl);
    }
  });

  // ── CSV ───────────────────────────────────────────────────────────

  it("renders CSV via globalRegistry with text/csv metadata", async () => {
    const csvContent = "Name,Age\nAlice,30\nBob,25";
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        text: () => Promise.resolve(csvContent),
      }),
    );

    const artifact = makeArtifact({
      id: "csv-render-001",
      title: "data.csv",
      mimeType: "text/csv",
    });
    const classification = makeClassification({
      type: "csv",
      hasSourceToggle: true,
    });

    render(
      <ArtifactContent
        artifact={artifact}
        isSourceView={false}
        classification={classification}
      />,
    );

    const rendered = await screen.findByTestId("global-renderer");
    expect(rendered).toBeTruthy();
    expect(rendered.textContent).toContain("text/csv");
  });

  it("renders TSV via globalRegistry with tab-separated metadata", async () => {
    const tsvContent = "Name\tAge\nAlice\t30\nBob\t25";
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        text: () => Promise.resolve(tsvContent),
      }),
    );

    const artifact = makeArtifact({
      id: "tsv-render-001",
      title: "data.tsv",
      mimeType: "text/tab-separated-values",
    });
    const classification = makeClassification({
      type: "csv",
      hasSourceToggle: true,
    });

    render(
      <ArtifactContent
        artifact={artifact}
        isSourceView={false}
        classification={classification}
      />,
    );

    const rendered = await screen.findByTestId("global-renderer");
    expect(rendered).toBeTruthy();
    expect(rendered.textContent).toContain("text/tab-separated-values");
  });

  // ── Markdown ──────────────────────────────────────────────────────

  it("renders markdown via globalRegistry", async () => {
    const mdContent = "# Hello\n\nThis is **markdown**.";
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        text: () => Promise.resolve(mdContent),
      }),
    );

    const artifact = makeArtifact({
      id: "md-render-001",
      title: "README.md",
      mimeType: "text/markdown",
    });
    const classification = makeClassification({
      type: "markdown",
      hasSourceToggle: true,
    });

    render(
      <ArtifactContent
        artifact={artifact}
        isSourceView={false}
        classification={classification}
      />,
    );

    const rendered = await screen.findByTestId("global-renderer");
    expect(rendered).toBeTruthy();
    expect(rendered.textContent).toContain("text/markdown");
  });

  // ── Text fallback ─────────────────────────────────────────────────

  it("renders text artifacts via globalRegistry fallback", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        text: () => Promise.resolve("plain text content"),
      }),
    );

    const artifact = makeArtifact({
      id: "text-render-001",
      title: "notes.txt",
      mimeType: "text/plain",
    });
    const classification = makeClassification({ type: "text" });

    render(
      <ArtifactContent
        artifact={artifact}
        isSourceView={false}
        classification={classification}
      />,
    );

    const rendered = await screen.findByTestId("global-renderer");
    expect(rendered).toBeTruthy();
  });

  it.each([
    {
      filename: "calendar.ics",
      mimeType: "text/calendar",
      content: "BEGIN:VCALENDAR\nVERSION:2.0\nEND:VCALENDAR",
    },
    {
      filename: "contact.vcf",
      mimeType: "text/vcard",
      content: "BEGIN:VCARD\nVERSION:4.0\nFN:Alice Example\nEND:VCARD",
    },
  ])(
    "renders concrete text artifact $filename through the global renderer path",
    async ({ filename, mimeType, content }) => {
      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue({
          ok: true,
          text: () => Promise.resolve(content),
        }),
      );

      const artifact = makeArtifact({
        id: `text-${filename}`,
        title: filename,
        mimeType,
      });
      const classification = classifyArtifact(
        artifact.mimeType,
        artifact.title,
      );

      render(
        <ArtifactContent
          artifact={artifact}
          isSourceView={false}
          classification={classification}
        />,
      );

      await screen.findByTestId("global-renderer");

      expect(classification.type).toBe("text");
      expect(vi.mocked(globalRegistry.getRenderer)).toHaveBeenCalledWith(
        content,
        expect.objectContaining({
          filename,
          mimeType,
        }),
      );
    },
  );

  // ── Error boundary ────────────────────────────────────────────────

  it("shows a visible error instead of crashing when the renderer throws", async () => {
    const consoleErr = vi.spyOn(console, "error").mockImplementation(() => {});
    const originalImpl = vi
      .mocked(ArtifactReactPreview)
      .getMockImplementation();
    vi.mocked(ArtifactReactPreview).mockImplementation(() => {
      throw new Error("boom in renderer");
    });

    try {
      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue({
          ok: true,
          text: () => Promise.resolve("source"),
        }),
      );

      const artifact = makeArtifact({
        id: "crash-001",
        title: "broken.tsx",
        mimeType: "text/tsx",
      });
      const classification = makeClassification({ type: "react" });

      render(
        <ArtifactContent
          artifact={artifact}
          isSourceView={false}
          classification={classification}
        />,
      );

      expect(
        await screen.findByText(/This artifact couldn't be rendered/i),
      ).toBeTruthy();
      expect(screen.getByText(/boom in renderer/)).toBeTruthy();
      expect(
        screen.getByRole("button", { name: /copy error details/i }),
      ).toBeTruthy();
    } finally {
      if (originalImpl) {
        vi.mocked(ArtifactReactPreview).mockImplementation(originalImpl);
      }
      consoleErr.mockRestore();
    }
  });

  it("copies artifact title, type, and error to the clipboard", async () => {
    const consoleErr = vi.spyOn(console, "error").mockImplementation(() => {});
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText },
      writable: true,
      configurable: true,
    });

    const originalImpl = vi
      .mocked(ArtifactReactPreview)
      .getMockImplementation();
    vi.mocked(ArtifactReactPreview).mockImplementation(() => {
      throw new Error("jsx parse failed at line 42");
    });

    try {
      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue({
          ok: true,
          text: () => Promise.resolve("source"),
        }),
      );

      render(
        <ArtifactContent
          artifact={makeArtifact({
            id: "crash-002",
            title: "report.tsx",
            mimeType: "text/tsx",
          })}
          isSourceView={false}
          classification={makeClassification({ type: "react" })}
        />,
      );

      fireEvent.click(
        await screen.findByRole("button", { name: /copy error details/i }),
      );

      await waitFor(() => {
        expect(writeText).toHaveBeenCalled();
      });
      const payload = writeText.mock.calls[0]![0] as string;
      expect(payload).toContain("report.tsx");
      expect(payload).toContain("crash-002");
      expect(payload).toContain("react");
      expect(payload).toContain("jsx parse failed at line 42");
    } finally {
      if (originalImpl) {
        vi.mocked(ArtifactReactPreview).mockImplementation(originalImpl);
      }
      consoleErr.mockRestore();
    }
  });

  // Regression: two different artifacts can share the same title+type (e.g.
  // two "App.tsx" files from different sessions). The boundary must reset
  // when artifact.id changes, not only on title/type changes, otherwise
  // opening a second artifact after a crash stays stuck on the first's error.
  it("resets the error fallback when the artifact id changes (same title/type)", async () => {
    const consoleErr = vi.spyOn(console, "error").mockImplementation(() => {});
    const originalImpl = vi
      .mocked(ArtifactReactPreview)
      .getMockImplementation();

    // First render: throws.
    vi.mocked(ArtifactReactPreview).mockImplementation(() => {
      throw new Error("first render boom");
    });

    try {
      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue({
          ok: true,
          text: () => Promise.resolve("source"),
        }),
      );
      const classification = makeClassification({ type: "react" });

      const { rerender } = render(
        <ArtifactContent
          artifact={makeArtifact({
            id: "id-one",
            title: "App.tsx",
            mimeType: "text/tsx",
          })}
          isSourceView={false}
          classification={classification}
        />,
      );

      await screen.findByText(/This artifact couldn't be rendered/i);

      // Swap in a working renderer and a different artifact id (same title/type).
      if (originalImpl) {
        vi.mocked(ArtifactReactPreview).mockImplementation(originalImpl);
      }

      rerender(
        <ArtifactContent
          artifact={makeArtifact({
            id: "id-two",
            title: "App.tsx",
            mimeType: "text/tsx",
          })}
          isSourceView={false}
          classification={classification}
        />,
      );

      await waitFor(() => {
        expect(
          screen.queryByText(/This artifact couldn't be rendered/i),
        ).toBeNull();
        expect(screen.getByTestId("react-preview")).toBeTruthy();
      });
    } finally {
      if (originalImpl) {
        vi.mocked(ArtifactReactPreview).mockImplementation(originalImpl);
      }
      consoleErr.mockRestore();
    }
  });

  it("renders the user-reported plotly HTML artifact into a sandboxed iframe", async () => {
    const html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AutoGPT Beta Launch Interactive Report</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  :root { --bg: #f8f9fa; --primary: #6c5ce7; }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; }
</style>
</head>
<body>
<header><h1>\u{1F4CA} AutoGPT Beta Launch Interactive Report</h1></header>
<div class="chart-container" id="globalActivationChart"></div>
<script>
  function showTab(tabId, groupId) {
    const group = document.getElementById(groupId);
    group.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    document.getElementById(tabId).classList.add('active');
  }
  Plotly.newPlot('globalActivationChart', [{ type: 'pie', values: [1, 2] }], {});
</script>
</body>
</html>`;

    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        text: () => Promise.resolve(html),
      }),
    );

    const artifact = makeArtifact({
      id: "html-big-report",
      title: "report.html",
      mimeType: "text/html",
    });

    const { container } = render(
      <ArtifactContent
        artifact={artifact}
        isSourceView={false}
        classification={makeClassification({ type: "html" })}
      />,
    );

    await screen.findByTitle("report.html");
    const iframe = container.querySelector("iframe");
    expect(iframe).toBeTruthy();
    expect(iframe?.getAttribute("sandbox")).toBe("allow-scripts");
    expect(screen.queryByText(/couldn't be rendered/i)).toBeNull();
  });

  it("falls back to pre tag when no renderer matches", async () => {
    const { globalRegistry } = await import(
      "@/components/contextual/OutputRenderers"
    );
    const originalImpl = vi
      .mocked(globalRegistry.getRenderer)
      .getMockImplementation();

    vi.mocked(globalRegistry.getRenderer).mockReturnValue(null);

    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        text: () => Promise.resolve("raw content fallback"),
      }),
    );

    const artifact = makeArtifact({
      id: "fallback-pre-001",
      title: "unknown.txt",
      mimeType: "text/plain",
    });
    const classification = makeClassification({ type: "text" });

    const { container } = render(
      <ArtifactContent
        artifact={artifact}
        isSourceView={false}
        classification={classification}
      />,
    );

    await waitFor(() => {
      const pre = container.querySelector("pre");
      expect(pre).toBeTruthy();
      expect(pre?.textContent).toBe("raw content fallback");
    });

    // Restore
    if (originalImpl) {
      vi.mocked(globalRegistry.getRenderer).mockImplementation(originalImpl);
    }
  });
});
