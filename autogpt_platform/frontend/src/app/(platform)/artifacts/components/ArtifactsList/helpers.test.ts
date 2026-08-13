import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import {
  deriveFileOrigin,
  downloadFileBlob,
  formatFileSize,
  formatRelativeDate,
  getFileDownloadUrl,
  getFilePreviewUrl,
  getFileTypeIcon,
  getFileTypeLabel,
  getPreviewKind,
  isCodeFile,
} from "./helpers";

describe("deriveFileOrigin", () => {
  test("extracts session id from /sessions/<id>/...", () => {
    const origin = deriveFileOrigin("/sessions/abc-123/notes.txt");
    expect(origin.kind).toBe("session");
    if (origin.kind === "session") {
      expect(origin.sessionId).toBe("abc-123");
      expect(origin.href).toBe("/copilot?sessionId=abc-123");
    }
  });

  test("url-encodes session ids that contain special characters", () => {
    const origin = deriveFileOrigin("/sessions/has space/x");
    expect(origin.kind).toBe("session");
    if (origin.kind === "session") {
      expect(origin.href).toBe("/copilot?sessionId=has%20space");
    }
  });

  test("falls back to builder for non-session paths", () => {
    expect(deriveFileOrigin("/uploads/file.txt")).toEqual({
      kind: "builder",
      href: "/build",
    });
  });

  test("handles undefined safely", () => {
    expect(deriveFileOrigin(undefined)).toEqual({
      kind: "builder",
      href: "/build",
    });
  });

  test("a literal /sessions path without a child segment is not a session", () => {
    expect(deriveFileOrigin("/sessions/")).toEqual({
      kind: "builder",
      href: "/build",
    });
  });
});

describe("getFileDownloadUrl", () => {
  test("url-encodes the file id", () => {
    expect(getFileDownloadUrl("a/b c")).toBe(
      "/api/proxy/api/workspace/files/a%2Fb%20c/download",
    );
  });
});

describe("getFilePreviewUrl", () => {
  test("returns the base url with no query when no opts are given", () => {
    expect(getFilePreviewUrl("f1", {})).toBe(
      "/api/proxy/api/workspace/files/f1/preview",
    );
  });

  test("adds the width param", () => {
    expect(getFilePreviewUrl("f1", { width: 400 })).toBe(
      "/api/proxy/api/workspace/files/f1/preview?w=400",
    );
  });

  test("adds the bytes param", () => {
    expect(getFilePreviewUrl("f1", { bytes: 4096 })).toBe(
      "/api/proxy/api/workspace/files/f1/preview?bytes=4096",
    );
  });

  test("combines width and bytes and url-encodes the id", () => {
    expect(getFilePreviewUrl("a b", { width: 400, bytes: 4096 })).toBe(
      "/api/proxy/api/workspace/files/a%20b/preview?w=400&bytes=4096",
    );
  });
});

describe("downloadFileBlob", () => {
  const realCreate = URL.createObjectURL;
  const realRevoke = URL.revokeObjectURL;

  afterEach(() => {
    vi.restoreAllMocks();
    URL.createObjectURL = realCreate;
    URL.revokeObjectURL = realRevoke;
  });

  test("fetches the download url and triggers an anchor click", async () => {
    URL.createObjectURL = vi.fn(() => "blob:mock");
    URL.revokeObjectURL = vi.fn();
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(new Response(new Blob(["hi"]), { status: 200 }));
    const clickSpy = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => {});

    await downloadFileBlob("file 1", "out.txt");

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/proxy/api/workspace/files/file%201/download",
    );
    expect(clickSpy).toHaveBeenCalled();
    expect(URL.createObjectURL).toHaveBeenCalled();
  });

  test("throws when the response is not ok", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response("nope", { status: 404 }),
    );

    await expect(downloadFileBlob("f", "n")).rejects.toThrow(/404/);
  });
});

describe("formatFileSize", () => {
  test.each([
    [0, "0 B"],
    [-100, "0 B"],
    [Number.NaN, "0 B"],
    [Number.POSITIVE_INFINITY, "0 B"],
    [500, "500 B"],
    [1024, "1.0 KB"],
    [1024 * 1024, "1.0 MB"],
    [1024 * 1024 * 1024, "1.0 GB"],
    [1024 * 1024 * 1024 * 1024, "1.0 TB"],
  ])("formatFileSize(%s) === %s", (bytes, expected) => {
    expect(formatFileSize(bytes)).toBe(expected);
  });
});

describe("formatRelativeDate", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-05-29T12:00:00Z"));
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  test("returns dash for invalid dates", () => {
    expect(formatRelativeDate("not-a-date")).toBe("—");
  });

  test("under a minute reads as 'just now'", () => {
    expect(formatRelativeDate("2026-05-29T11:59:30Z")).toBe("just now");
  });

  test("between 1 and 60 minutes reads in minutes", () => {
    expect(formatRelativeDate("2026-05-29T11:55:00Z")).toBe("5m ago");
  });

  test("between 1 and 24 hours reads in hours", () => {
    expect(formatRelativeDate("2026-05-29T08:00:00Z")).toBe("4h ago");
  });

  test("between 1 and 7 days reads in days", () => {
    expect(formatRelativeDate("2026-05-27T12:00:00Z")).toBe("2d ago");
  });

  test("older than 7 days reads as a date with the current year hidden", () => {
    const out = formatRelativeDate("2026-04-01T12:00:00Z");
    expect(out).toMatch(/Apr 1/);
    expect(out).not.toMatch(/2026/);
  });

  test("older than 7 days from a previous year includes the year", () => {
    const out = formatRelativeDate("2024-04-01T12:00:00Z");
    expect(out).toMatch(/2024/);
  });

  test("accepts Date instances directly", () => {
    expect(formatRelativeDate(new Date("2026-05-29T11:55:00Z"))).toBe("5m ago");
  });
});

describe("getFileTypeLabel + getFileTypeIcon", () => {
  test.each([
    ["image/png", "Image"],
    ["video/mp4", "Video"],
    ["audio/mpeg", "Audio"],
    ["application/pdf", "PDF document"],
    ["text/html", "Web page"],
    ["application/xhtml+xml", "Web page"],
    ["text/csv", "Spreadsheet"],
    [
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
      "Spreadsheet",
    ],
    ["application/json", "JSON data"],
    ["text/markdown", "Markdown"],
    ["text/plain", "Document"],
    ["application/octet-stream", "Generated file"],
    [undefined, "Generated file"],
  ])("getFileTypeLabel(%s) === %s", (mime, expected) => {
    expect(getFileTypeLabel(mime)).toBe(expected);
  });

  test("returns a callable icon component for every known mime family", () => {
    for (const mime of [
      "image/png",
      "video/mp4",
      "application/pdf",
      "text/html",
      "text/csv",
      "application/json",
      "text/markdown",
      "application/octet-stream",
      undefined,
    ]) {
      const Icon = getFileTypeIcon(mime);
      expect(typeof Icon).toBe("object");
    }
  });
});

describe("getPreviewKind", () => {
  test("images under the 10MB cap preview as 'image'", () => {
    expect(getPreviewKind("image/png", 1_000)).toBe("image");
  });

  test("images over the cap suppress the preview", () => {
    expect(getPreviewKind("image/png", 20_000_000)).toBe("none");
  });

  test("videos under the cap preview as 'video'", () => {
    expect(getPreviewKind("video/mp4", 1_000_000)).toBe("video");
  });

  test("videos over the cap suppress the preview", () => {
    expect(getPreviewKind("video/mp4", 800_000_000)).toBe("none");
  });

  test.each([
    "text/plain",
    "text/html",
    "application/xml",
    "application/javascript",
    "application/typescript",
    "application/yaml",
  ])("text-like mime '%s' under cap previews as 'text'", (mt) => {
    expect(getPreviewKind(mt, 1_000)).toBe("text");
  });

  test("markdown previews as 'markdown' so the card renders its content", () => {
    expect(getPreviewKind("text/markdown", 1_000)).toBe("markdown");
    expect(getPreviewKind("text/plain", 1_000, "README.md")).toBe("markdown");
    expect(getPreviewKind("application/octet-stream", 1_000, "notes.mdx")).toBe(
      "markdown",
    );
  });

  test("csv previews as 'csv'", () => {
    expect(getPreviewKind("text/csv", 1_000)).toBe("csv");
    expect(getPreviewKind("text/plain", 1_000, "data.csv")).toBe("csv");
  });

  test("json previews as 'json'", () => {
    expect(getPreviewKind("application/json", 1_000)).toBe("json");
    expect(getPreviewKind("text/plain", 1_000, "data.json")).toBe("json");
  });

  test("pdf previews as 'pdf' by mime or extension", () => {
    expect(getPreviewKind("application/pdf", 1_000)).toBe("pdf");
    expect(getPreviewKind("application/octet-stream", 1_000, "x.pdf")).toBe(
      "pdf",
    );
  });

  test("office openxml docs preview as 'office'", () => {
    expect(
      getPreviewKind(
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        1_000,
      ),
    ).toBe("office");
    expect(getPreviewKind("application/octet-stream", 1_000, "deck.pptx")).toBe(
      "office",
    );
  });

  test("ics/vcard preview as cards only under the 50KB cap", () => {
    expect(getPreviewKind("text/calendar", 1_000, "e.ics")).toBe("ics");
    expect(getPreviewKind("text/vcard", 1_000, "c.vcf")).toBe("vcard");
    expect(getPreviewKind("text/calendar", 200_000, "e.ics")).toBe("none");
  });

  test("svg images are treated as text, not raster image", () => {
    expect(getPreviewKind("image/svg+xml", 1_000)).toBe("text");
  });

  test("large text/csv still previews (fetch is byte-capped)", () => {
    expect(getPreviewKind("text/plain", 5_000_000)).toBe("text");
    expect(getPreviewKind("text/csv", 5_000_000)).toBe("csv");
  });

  test("text/csv over the 50MB backend ceiling suppresses the preview", () => {
    expect(getPreviewKind("text/plain", 60_000_000)).toBe("none");
    expect(getPreviewKind("text/csv", 60_000_000)).toBe("none");
  });

  test("images/pdf over the backend ceiling suppress the preview", () => {
    expect(getPreviewKind("image/png", 11_000_000)).toBe("none");
    expect(getPreviewKind("application/pdf", 60_000_000)).toBe("none");
  });

  test("unknown binary mime types return 'none'", () => {
    expect(getPreviewKind("application/octet-stream", 100)).toBe("none");
  });

  test("undefined mime returns 'none'", () => {
    expect(getPreviewKind(undefined, 100)).toBe("none");
  });
});

describe("source code files (`.ts` → video/mp2t MIME)", () => {
  test("isCodeFile matches code extensions, not media", () => {
    expect(isCodeFile("main.ts")).toBe(true);
    expect(isCodeFile("App.tsx")).toBe(true);
    expect(isCodeFile("script.py")).toBe(true);
    expect(isCodeFile("style.css")).toBe(true);
    expect(isCodeFile("clip.mp4")).toBe(false);
    expect(isCodeFile("photo.png")).toBe(false);
    expect(isCodeFile("notes.txt")).toBe(false);
    expect(isCodeFile(undefined)).toBe(false);
  });

  test("a .ts file with video/mp2t MIME previews as text, not video", () => {
    expect(getPreviewKind("video/mp2t", 1_000, "main.ts")).toBe("text");
  });

  test("a .ts file labels as Code, not Video", () => {
    expect(getFileTypeLabel("video/mp2t", "main.ts")).toBe("Code");
  });

  test("a .ts file uses the code icon, not the video-camera icon", () => {
    const codeIcon = getFileTypeIcon("video/mp2t", "main.ts");
    const videoIcon = getFileTypeIcon("video/mp4", "clip.mp4");
    expect(codeIcon).not.toBe(videoIcon);
  });
});
