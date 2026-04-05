import { afterEach, describe, expect, it, vi } from "vitest";
import type { ArtifactRef } from "../../store";
import { downloadArtifact } from "./downloadArtifact";

function makeArtifact(title: string): ArtifactRef {
  return {
    id: "abc",
    title,
    mimeType: "text/plain",
    sourceUrl: "/api/proxy/api/workspace/files/abc/download",
    origin: "agent",
  };
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("downloadArtifact filename sanitization", () => {
  it("strips path separators and control characters", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      blob: () => Promise.resolve(new Blob(["x"])),
    });
    const clicks: HTMLAnchorElement[] = [];
    const originalCreate = document.createElement.bind(document);
    vi.spyOn(document, "createElement").mockImplementation((tag: string) => {
      const el = originalCreate(tag);
      if (tag === "a") {
        clicks.push(el as HTMLAnchorElement);
        // Prevent actual navigation in test env.
        (el as HTMLAnchorElement).click = () => {};
      }
      return el;
    });
    global.URL.createObjectURL = vi.fn(() => "blob:mock");
    global.URL.revokeObjectURL = vi.fn();

    await downloadArtifact(makeArtifact("../../etc/passwd"));
    // ..→_ then /→_ gives ____etc_passwd (no leading ..)
    expect(clicks[0]?.download).toBe("____etc_passwd");
  });

  it("replaces Windows-reserved characters", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      blob: () => Promise.resolve(new Blob(["x"])),
    });
    const clicks: HTMLAnchorElement[] = [];
    const originalCreate = document.createElement.bind(document);
    vi.spyOn(document, "createElement").mockImplementation((tag: string) => {
      const el = originalCreate(tag);
      if (tag === "a") {
        clicks.push(el as HTMLAnchorElement);
        (el as HTMLAnchorElement).click = () => {};
      }
      return el;
    });
    global.URL.createObjectURL = vi.fn(() => "blob:mock");
    global.URL.revokeObjectURL = vi.fn();

    await downloadArtifact(makeArtifact('a<b>c:"d*e?f|g'));
    expect(clicks[0]?.download).toBe("a_b_c__d_e_f_g");
  });

  it("falls back to 'download' when title is empty after sanitization", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      blob: () => Promise.resolve(new Blob(["x"])),
    });
    const clicks: HTMLAnchorElement[] = [];
    const originalCreate = document.createElement.bind(document);
    vi.spyOn(document, "createElement").mockImplementation((tag: string) => {
      const el = originalCreate(tag);
      if (tag === "a") {
        clicks.push(el as HTMLAnchorElement);
        (el as HTMLAnchorElement).click = () => {};
      }
      return el;
    });
    global.URL.createObjectURL = vi.fn(() => "blob:mock");
    global.URL.revokeObjectURL = vi.fn();

    await downloadArtifact(makeArtifact(""));
    expect(clicks[0]?.download).toBe("download");
  });

  it("keeps normal filenames intact", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      blob: () => Promise.resolve(new Blob(["x"])),
    });
    const clicks: HTMLAnchorElement[] = [];
    const originalCreate = document.createElement.bind(document);
    vi.spyOn(document, "createElement").mockImplementation((tag: string) => {
      const el = originalCreate(tag);
      if (tag === "a") {
        clicks.push(el as HTMLAnchorElement);
        (el as HTMLAnchorElement).click = () => {};
      }
      return el;
    });
    global.URL.createObjectURL = vi.fn(() => "blob:mock");
    global.URL.revokeObjectURL = vi.fn();

    await downloadArtifact(makeArtifact("report-2024 (final).pdf"));
    expect(clicks[0]?.download).toBe("report-2024 (final).pdf");
  });

  it("rejects when fetch returns non-ok status", async () => {
    global.fetch = vi.fn().mockResolvedValue({ ok: false, status: 404 });
    await expect(downloadArtifact(makeArtifact("x.txt"))).rejects.toThrow(
      /Download failed: 404/,
    );
  });

  it("rejects when fetch itself throws", async () => {
    global.fetch = vi.fn().mockRejectedValue(new Error("network"));
    await expect(downloadArtifact(makeArtifact("x.txt"))).rejects.toThrow();
  });
});
