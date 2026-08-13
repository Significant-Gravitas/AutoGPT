import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { downloadArtifact } from "../downloadArtifact";
import type { ArtifactRef } from "../../../store";

function makeArtifact(overrides?: Partial<ArtifactRef>): ArtifactRef {
  return {
    id: "file-001",
    title: "report.pdf",
    mimeType: "application/pdf",
    sourceUrl: "/api/proxy/api/workspace/files/file-001/download",
    origin: "agent",
    ...overrides,
  };
}

describe("downloadArtifact", () => {
  let clickSpy: ReturnType<typeof vi.fn>;
  let removeSpy: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    clickSpy = vi.fn();
    removeSpy = vi.fn();

    vi.stubGlobal(
      "URL",
      Object.assign(URL, {
        createObjectURL: vi.fn().mockReturnValue("blob:fake-url"),
        revokeObjectURL: vi.fn(),
      }),
    );

    vi.spyOn(document, "createElement").mockReturnValue({
      href: "",
      download: "",
      click: clickSpy,
      remove: removeSpy,
    } as unknown as HTMLAnchorElement);

    vi.spyOn(document.body, "appendChild").mockImplementation(
      (node) => node as ChildNode,
    );
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("downloads file successfully on 200 response", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        blob: () => Promise.resolve(new Blob(["pdf content"])),
      }),
    );

    await downloadArtifact(makeArtifact());

    expect(fetch).toHaveBeenCalledWith(
      "/api/proxy/api/workspace/files/file-001/download",
    );
    expect(clickSpy).toHaveBeenCalled();
    expect(removeSpy).toHaveBeenCalled();
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:fake-url");
  });

  it("rejects on persistent server error after exhausting retries", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 500,
      }),
    );

    await expect(downloadArtifact(makeArtifact())).rejects.toThrow(
      "Download failed: 500",
    );
    expect(clickSpy).not.toHaveBeenCalled();
  });

  it("rejects on persistent network error after exhausting retries", async () => {
    let callCount = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation(() => {
        callCount++;
        return Promise.reject(new Error("Network error"));
      }),
    );

    await expect(downloadArtifact(makeArtifact())).rejects.toThrow(
      "Network error",
    );
    expect(callCount).toBe(3);
    expect(clickSpy).not.toHaveBeenCalled();
  });

  it("retries on transient network error and succeeds", async () => {
    let callCount = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation(() => {
        callCount++;
        if (callCount === 1) {
          return Promise.reject(new Error("Connection reset"));
        }
        return Promise.resolve({
          ok: true,
          blob: () => Promise.resolve(new Blob(["content"])),
        });
      }),
    );

    await downloadArtifact(makeArtifact());
    expect(callCount).toBe(2);
    expect(clickSpy).toHaveBeenCalled();
  });

  it("retries on transient 500 and succeeds", async () => {
    let callCount = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation(() => {
        callCount++;
        if (callCount === 1) {
          return Promise.resolve({ ok: false, status: 500 });
        }
        return Promise.resolve({
          ok: true,
          blob: () => Promise.resolve(new Blob(["content"])),
        });
      }),
    );

    // Should succeed on second attempt
    await downloadArtifact(makeArtifact());
    expect(callCount).toBe(2);
    expect(clickSpy).toHaveBeenCalled();
  });

  it("sanitizes dangerous filenames", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        blob: () => Promise.resolve(new Blob(["content"])),
      }),
    );

    const anchor = {
      href: "",
      download: "",
      click: clickSpy,
      remove: removeSpy,
    };
    vi.spyOn(document, "createElement").mockReturnValue(
      anchor as unknown as HTMLAnchorElement,
    );

    await downloadArtifact(makeArtifact({ title: "../../../etc/passwd" }));

    expect(anchor.download).not.toContain("..");
    expect(anchor.download).not.toContain("/");
  });

  // ── Transient retry codes ─────────────────────────────────────────

  it("retries on 408 (Request Timeout) and succeeds", async () => {
    let callCount = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation(() => {
        callCount++;
        if (callCount === 1) {
          return Promise.resolve({ ok: false, status: 408 });
        }
        return Promise.resolve({
          ok: true,
          blob: () => Promise.resolve(new Blob(["content"])),
        });
      }),
    );

    await downloadArtifact(makeArtifact());
    expect(callCount).toBe(2);
    expect(clickSpy).toHaveBeenCalled();
  });

  it("retries on 429 (Too Many Requests) and succeeds", async () => {
    let callCount = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation(() => {
        callCount++;
        if (callCount === 1) {
          return Promise.resolve({ ok: false, status: 429 });
        }
        return Promise.resolve({
          ok: true,
          blob: () => Promise.resolve(new Blob(["content"])),
        });
      }),
    );

    await downloadArtifact(makeArtifact());
    expect(callCount).toBe(2);
    expect(clickSpy).toHaveBeenCalled();
  });

  // ── Non-transient errors ──────────────────────────────────────────

  it("rejects immediately on 403 (non-transient) without retry", async () => {
    let callCount = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation(() => {
        callCount++;
        return Promise.resolve({ ok: false, status: 403 });
      }),
    );

    await expect(downloadArtifact(makeArtifact())).rejects.toThrow(
      "Download failed: 403",
    );
    expect(callCount).toBe(1);
    expect(clickSpy).not.toHaveBeenCalled();
  });

  it("rejects immediately on 404 without retry", async () => {
    let callCount = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation(() => {
        callCount++;
        return Promise.resolve({ ok: false, status: 404 });
      }),
    );

    await expect(downloadArtifact(makeArtifact())).rejects.toThrow(
      "Download failed: 404",
    );
    expect(callCount).toBe(1);
  });

  // ── Exhausted retries ─────────────────────────────────────────────

  it("rejects after exhausting all retries on persistent 500", async () => {
    let callCount = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation(() => {
        callCount++;
        return Promise.resolve({ ok: false, status: 500 });
      }),
    );

    await expect(downloadArtifact(makeArtifact())).rejects.toThrow(
      "Download failed: 500",
    );
    // Initial attempt + 2 retries = 3 total
    expect(callCount).toBe(3);
    expect(clickSpy).not.toHaveBeenCalled();
  });

  // ── Filename edge cases ───────────────────────────────────────────

  it("falls back to 'download' when title is empty", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        blob: () => Promise.resolve(new Blob(["content"])),
      }),
    );

    const anchor = {
      href: "",
      download: "",
      click: clickSpy,
      remove: removeSpy,
    };
    vi.spyOn(document, "createElement").mockReturnValue(
      anchor as unknown as HTMLAnchorElement,
    );

    await downloadArtifact(makeArtifact({ title: "" }));
    expect(anchor.download).toBe("download");
  });

  it("falls back to 'download' when title is only dots", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        blob: () => Promise.resolve(new Blob(["content"])),
      }),
    );

    const anchor = {
      href: "",
      download: "",
      click: clickSpy,
      remove: removeSpy,
    };
    vi.spyOn(document, "createElement").mockReturnValue(
      anchor as unknown as HTMLAnchorElement,
    );

    // Dot-only names should not produce a hidden or empty filename.
    await downloadArtifact(makeArtifact({ title: "...." }));
    expect(anchor.download).toBe("download");
  });

  it("replaces special chars with underscores (not empty)", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        blob: () => Promise.resolve(new Blob(["content"])),
      }),
    );

    const anchor = {
      href: "",
      download: "",
      click: clickSpy,
      remove: removeSpy,
    };
    vi.spyOn(document, "createElement").mockReturnValue(
      anchor as unknown as HTMLAnchorElement,
    );

    await downloadArtifact(makeArtifact({ title: '***???"' }));
    // Special chars become underscores, not removed
    expect(anchor.download).toBe("_______");
  });

  it("strips leading dots from filename", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        blob: () => Promise.resolve(new Blob(["content"])),
      }),
    );

    const anchor = {
      href: "",
      download: "",
      click: clickSpy,
      remove: removeSpy,
    };
    vi.spyOn(document, "createElement").mockReturnValue(
      anchor as unknown as HTMLAnchorElement,
    );

    await downloadArtifact(makeArtifact({ title: "...hidden.txt" }));
    expect(anchor.download).not.toMatch(/^\./);
    expect(anchor.download).toContain("hidden.txt");
  });

  it("replaces Windows-reserved characters", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        blob: () => Promise.resolve(new Blob(["content"])),
      }),
    );

    const anchor = {
      href: "",
      download: "",
      click: clickSpy,
      remove: removeSpy,
    };
    vi.spyOn(document, "createElement").mockReturnValue(
      anchor as unknown as HTMLAnchorElement,
    );

    await downloadArtifact(
      makeArtifact({ title: "file<name>with:bad*chars?.txt" }),
    );
    expect(anchor.download).not.toMatch(/[<>:*?]/);
  });

  it("replaces control characters in filename", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        blob: () => Promise.resolve(new Blob(["content"])),
      }),
    );

    const anchor = {
      href: "",
      download: "",
      click: clickSpy,
      remove: removeSpy,
    };
    vi.spyOn(document, "createElement").mockReturnValue(
      anchor as unknown as HTMLAnchorElement,
    );

    await downloadArtifact(
      makeArtifact({ title: "file\x00with\x1fcontrol.txt" }),
    );
    expect(anchor.download).not.toMatch(/[\x00-\x1f]/);
  });
});
