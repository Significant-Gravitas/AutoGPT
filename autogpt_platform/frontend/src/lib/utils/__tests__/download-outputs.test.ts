import { describe, expect, it, vi, beforeEach } from "vitest";
import {
  sanitizeFilename,
  getUniqueFilename,
  downloadOutputs,
} from "../download-outputs";
import type { DownloadItem } from "../download-outputs";

describe("sanitizeFilename", () => {
  it("strips forward slashes", () => {
    expect(sanitizeFilename("path/to/file.txt")).toBe("path_to_file.txt");
  });

  it("strips backslashes", () => {
    expect(sanitizeFilename("path\\to\\file.txt")).toBe("path_to_file.txt");
  });

  it("replaces parent directory traversal", () => {
    const result = sanitizeFilename("../../etc/passwd");
    expect(result).not.toContain("/");
    expect(result).not.toContain("\\");
    expect(result).not.toContain("..");
    expect(result).not.toMatch(/^\./);
  });

  it("strips leading dots", () => {
    expect(sanitizeFilename(".gitignore")).toBe("gitignore");
    expect(sanitizeFilename("..hidden")).toBe("hidden");
    expect(sanitizeFilename("...triple")).toBe("triple");
  });

  it("returns 'file' for empty results", () => {
    expect(sanitizeFilename("")).toBe("file");
    expect(sanitizeFilename("...")).toBe("file");
    expect(sanitizeFilename(".")).toBe("file");
  });

  it("leaves safe filenames unchanged", () => {
    expect(sanitizeFilename("report.pdf")).toBe("report.pdf");
    expect(sanitizeFilename("image_001.png")).toBe("image_001.png");
  });
});

describe("getUniqueFilename", () => {
  it("returns the filename when not already used", () => {
    const used = new Set<string>();
    expect(getUniqueFilename("file.txt", used)).toBe("file.txt");
    expect(used.has("file.txt")).toBe(true);
  });

  it("appends a counter on collision", () => {
    const used = new Set<string>(["file.txt"]);
    expect(getUniqueFilename("file.txt", used)).toBe("file_1.txt");
    expect(used.has("file_1.txt")).toBe(true);
  });

  it("increments counter until unique", () => {
    const used = new Set<string>(["file.txt", "file_1.txt", "file_2.txt"]);
    expect(getUniqueFilename("file.txt", used)).toBe("file_3.txt");
  });

  it("handles filenames without extensions", () => {
    const used = new Set<string>(["README"]);
    expect(getUniqueFilename("README", used)).toBe("README_1");
  });

  it("sanitizes the filename before deduplication", () => {
    const used = new Set<string>();
    expect(getUniqueFilename("../evil.txt", used)).toBe("_evil.txt");
  });

  it("handles dotfiles by stripping leading dots first", () => {
    const used = new Set<string>();
    expect(getUniqueFilename(".gitignore", used)).toBe("gitignore");
  });
});

const mockZipFile = vi.fn();
const mockGenerateAsync = vi.fn();
let mockZipFiles: Record<string, { async: () => Promise<Blob> }> = {};

vi.mock("jszip", () => ({
  default: class MockJSZip {
    files = mockZipFiles;
    file = (...args: unknown[]) => {
      if (typeof args[0] === "string" && args[1] !== undefined) {
        const content = args[1];
        mockZipFiles[args[0] as string] = {
          async: () =>
            Promise.resolve(
              content instanceof Blob ? content : new Blob([String(content)]),
            ),
        };
      }
      mockZipFile(...args);
    };
    generateAsync = mockGenerateAsync;
  },
}));

function makeRenderer(overrides: {
  isConcatenable?: boolean;
  copyData?: string;
  downloadData?: Blob | string;
  downloadFilename?: string;
}) {
  return {
    value: "test",
    metadata: undefined,
    renderer: {
      name: "test",
      priority: 1,
      canRender: () => true,
      render: () => null,
      isConcatenable: () => overrides.isConcatenable ?? false,
      getCopyContent: () =>
        overrides.copyData
          ? { mimeType: "text/plain", data: overrides.copyData }
          : null,
      getDownloadContent: () =>
        overrides.downloadData
          ? {
              data: overrides.downloadData,
              filename: overrides.downloadFilename ?? "file.bin",
              mimeType: "application/octet-stream",
            }
          : null,
    },
  } satisfies DownloadItem;
}

describe("downloadOutputs", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockZipFiles = {};
    mockGenerateAsync.mockResolvedValue(new Blob(["zip-content"]));
    vi.stubGlobal(
      "URL",
      Object.assign(URL, {
        createObjectURL: vi.fn(() => "blob:mock-url"),
        revokeObjectURL: vi.fn(),
      }),
    );
  });

  it("creates a zip with concatenable text outputs", async () => {
    const items = [
      makeRenderer({ isConcatenable: true, copyData: "Hello" }),
      makeRenderer({ isConcatenable: true, copyData: "World" }),
    ];

    await downloadOutputs(items);

    expect(mockZipFile).toHaveBeenCalledWith(
      "combined_output.txt",
      "Hello\n\n---\n\nWorld",
    );
    // Single file in zip → downloaded directly, no zip generation
    expect(mockGenerateAsync).not.toHaveBeenCalled();
  });

  it("includes direct blob data in the zip", async () => {
    const blob = new Blob(["binary data"]);
    const items = [
      makeRenderer({ downloadData: blob, downloadFilename: "image.png" }),
    ];

    await downloadOutputs(items);

    expect(mockZipFile).toHaveBeenCalledWith("image.png", blob);
  });

  it("skips blobs exceeding size limit", async () => {
    const consoleSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const bigBlob = new Blob(["x".repeat(100)]);
    Object.defineProperty(bigBlob, "size", { value: 60 * 1024 * 1024 });

    const items = [
      makeRenderer({ downloadData: bigBlob, downloadFilename: "huge.bin" }),
    ];

    await downloadOutputs(items);

    expect(consoleSpy).toHaveBeenCalledWith(
      expect.stringContaining("blob too large"),
    );
    expect(mockZipFile).not.toHaveBeenCalledWith("huge.bin", expect.anything());
    consoleSpy.mockRestore();
  });

  it("fetches http URLs and adds to zip", async () => {
    const mockBlob = new Blob(["fetched"]);
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        headers: new Headers({ "content-length": "7" }),
        blob: () => Promise.resolve(mockBlob),
      }),
    );

    const items = [
      makeRenderer({
        downloadData: "https://example.com/file.pdf",
        downloadFilename: "report.pdf",
      }),
    ];

    await downloadOutputs(items);

    expect(fetch).toHaveBeenCalledWith("https://example.com/file.pdf", {
      mode: "cors",
    });
    expect(mockZipFile).toHaveBeenCalledWith("report.pdf", mockBlob);
  });

  it("handles fetch failures gracefully and records unfetchable URLs", async () => {
    const consoleSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("CORS error")));

    const items = [
      makeRenderer({ isConcatenable: true, copyData: "some text" }),
      makeRenderer({
        downloadData: "https://cors-blocked.com/file.bin",
        downloadFilename: "blocked.bin",
      }),
    ];

    await downloadOutputs(items);

    expect(mockZipFile).toHaveBeenCalledWith(
      "unfetched_files.txt",
      expect.stringContaining("cors-blocked.com"),
    );
    consoleSpy.mockRestore();
  });

  it("handles malformed data URLs with try-catch", async () => {
    const consoleSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    vi.stubGlobal(
      "fetch",
      vi.fn().mockRejectedValue(new Error("Invalid data URL")),
    );

    const items = [
      makeRenderer({
        downloadData: "data:invalid",
        downloadFilename: "broken.bin",
      }),
    ];

    await downloadOutputs(items);

    expect(consoleSpy).toHaveBeenCalledWith(
      expect.stringContaining("malformed or unsupported format"),
    );
    consoleSpy.mockRestore();
  });

  it("skips unsupported URL formats with a warning", async () => {
    const consoleSpy = vi.spyOn(console, "warn").mockImplementation(() => {});

    const items = [
      makeRenderer({
        downloadData: "ftp://server/file.dat",
        downloadFilename: "file.dat",
      }),
    ];

    await downloadOutputs(items);

    expect(consoleSpy).toHaveBeenCalledWith(
      expect.stringContaining("unsupported URL format"),
    );
    consoleSpy.mockRestore();
  });

  it("does nothing when items array is empty", async () => {
    await downloadOutputs([]);

    expect(mockZipFile).not.toHaveBeenCalled();
    expect(mockGenerateAsync).not.toHaveBeenCalled();
  });

  it("fetches relative URLs (workspace files) and adds to zip", async () => {
    const mockBlob = new Blob(["image-data"]);
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        headers: new Headers({ "content-length": "10" }),
        blob: () => Promise.resolve(mockBlob),
      }),
    );

    const items = [
      makeRenderer({
        downloadData: "/api/proxy/api/workspace/files/abc-123/download",
        downloadFilename: "photo.png",
      }),
    ];

    await downloadOutputs(items);

    expect(fetch).toHaveBeenCalledWith(
      "/api/proxy/api/workspace/files/abc-123/download",
      { mode: "cors" },
    );
    expect(mockZipFile).toHaveBeenCalledWith("photo.png", mockBlob);
  });

  it("includes workspace images that renderers return as relative URLs", async () => {
    const mockBlob = new Blob(["img"]);
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        headers: new Headers({ "content-length": "3" }),
        blob: () => Promise.resolve(mockBlob),
      }),
    );

    const items = [
      makeRenderer({
        downloadData: "/api/proxy/api/workspace/files/file-1/download",
        downloadFilename: "image1.png",
      }),
      makeRenderer({
        downloadData: "/api/proxy/api/workspace/files/file-2/download",
        downloadFilename: "image2.jpg",
      }),
    ];

    await downloadOutputs(items);

    expect(mockZipFile).toHaveBeenCalledWith("image1.png", mockBlob);
    expect(mockZipFile).toHaveBeenCalledWith("image2.jpg", mockBlob);
  });

  it("fetches public share endpoint URLs for workspace files", async () => {
    const mockBlob = new Blob(["shared-image-data"]);
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        headers: new Headers({ "content-length": "17" }),
        blob: () => Promise.resolve(mockBlob),
      }),
    );

    const items = [
      makeRenderer({
        downloadData:
          "/api/proxy/api/public/shared/abc-token/files/file-123/download",
        downloadFilename: "shared-image.png",
      }),
    ];

    await downloadOutputs(items);

    expect(fetch).toHaveBeenCalledWith(
      "/api/proxy/api/public/shared/abc-token/files/file-123/download",
      { mode: "cors" },
    );
    expect(mockZipFile).toHaveBeenCalledWith("shared-image.png", mockBlob);
  });

  it("rejects files over content-length before buffering", async () => {
    const consoleSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const blobFn = vi.fn();
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        headers: new Headers({
          "content-length": String(60 * 1024 * 1024),
        }),
        blob: blobFn,
      }),
    );

    const items = [
      makeRenderer({
        downloadData: "https://example.com/huge.zip",
        downloadFilename: "huge.zip",
      }),
    ];

    await downloadOutputs(items);

    expect(blobFn).not.toHaveBeenCalled();
    expect(consoleSpy).toHaveBeenCalledWith(
      expect.stringContaining("file too large"),
    );
    consoleSpy.mockRestore();
  });

  it("downloads single file directly without zip wrapping", async () => {
    const blob = new Blob(["single file"]);
    const items = [
      makeRenderer({ downloadData: blob, downloadFilename: "photo.png" }),
    ];

    await downloadOutputs(items);

    expect(mockZipFile).toHaveBeenCalledWith("photo.png", blob);
    expect(mockGenerateAsync).not.toHaveBeenCalled();
    expect(URL.createObjectURL).toHaveBeenCalled();
  });

  it("uses zip when multiple files are present", async () => {
    const blob1 = new Blob(["file1"]);
    const blob2 = new Blob(["file2"]);
    const items = [
      makeRenderer({ downloadData: blob1, downloadFilename: "a.png" }),
      makeRenderer({ downloadData: blob2, downloadFilename: "b.png" }),
    ];

    await downloadOutputs(items);

    expect(mockGenerateAsync).toHaveBeenCalledWith({ type: "blob" });
  });
});
