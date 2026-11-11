import type { WorkspaceFile } from "@/lib/api/workspace";

import {
  chatFiles,
  fileTypeIcon,
  formatDayLabel,
  formatFileSize,
  hasImageThumbnail,
} from "../listHelpers";

function file(overrides: Partial<WorkspaceFile>): WorkspaceFile {
  return {
    id: "f1",
    name: "notes.txt",
    path: "/sessions/s1/notes.txt",
    mime_type: "text/plain",
    size_bytes: 12 * 1024,
    folder_id: null,
    metadata: null,
    origin: "generated",
    created_at: "2026-11-11T10:00:00Z",
    expert_id: null,
    is_internal: false,
    ...overrides,
  };
}

describe("formatFileSize", () => {
  it("shows nothing as 0 B", () => {
    expect(formatFileSize(0)).toBe("0 B");
    expect(formatFileSize(-5)).toBe("0 B");
    expect(formatFileSize(Number.NaN)).toBe("0 B");
  });

  it("shows bytes whole and larger units to one place", () => {
    expect(formatFileSize(512)).toBe("512 B");
    expect(formatFileSize(12 * 1024)).toBe("12.0 KB");
    expect(formatFileSize(1536)).toBe("1.5 KB");
    expect(formatFileSize(5 * 1024 * 1024)).toBe("5.0 MB");
  });
});

describe("formatDayLabel", () => {
  const now = new Date(2026, 10, 11, 18, 0);

  it("names today and yesterday", () => {
    expect(formatDayLabel(new Date(2026, 10, 11, 0, 5), now)).toBe("Today");
    expect(formatDayLabel(new Date(2026, 10, 10, 23, 55), now)).toBe("Yesterday");
  });

  it("gives an older day this year without the year", () => {
    const date = new Date(2026, 9, 2);
    expect(formatDayLabel(date, now)).toBe(
      date.toLocaleDateString(undefined, { month: "short", day: "numeric" }),
    );
  });

  it("adds the year for another year", () => {
    expect(formatDayLabel(new Date(2024, 4, 20), now)).toContain("2024");
  });

  it("shows a dash for a bad date", () => {
    expect(formatDayLabel("not a date", now)).toBe("—");
  });
});

describe("fileTypeIcon", () => {
  it("reads source code by extension, whatever its MIME type", () => {
    expect(fileTypeIcon({ name: "app.ts", mime_type: "video/mp2t" })).toBe("code");
  });

  it("reads office files by extension first", () => {
    expect(fileTypeIcon({ name: "deck.pptx", mime_type: "text/plain" })).toBe("slides");
    expect(fileTypeIcon({ name: "sheet.xlsx", mime_type: "" })).toBe("table");
  });

  it("falls back to the MIME type", () => {
    expect(fileTypeIcon({ name: "photo", mime_type: "image/png" })).toBe("image");
    expect(fileTypeIcon({ name: "report", mime_type: "application/pdf" })).toBe("pdf");
    expect(fileTypeIcon({ name: "data", mime_type: "text/csv" })).toBe("table");
    expect(fileTypeIcon({ name: "data", mime_type: "application/json" })).toBe("json");
    expect(fileTypeIcon({ name: "page", mime_type: "text/html" })).toBe("code");
    expect(fileTypeIcon({ name: "notes", mime_type: "text/markdown" })).toBe("document");
    expect(fileTypeIcon({ name: "blob", mime_type: "application/zip" })).toBe("other");
  });
});

describe("hasImageThumbnail", () => {
  it("loads a thumbnail for an ordinary image", () => {
    expect(hasImageThumbnail(file({ name: "a.png", mime_type: "image/png" }))).toBe(true);
  });

  it("skips SVGs, oversized images and other files", () => {
    expect(hasImageThumbnail(file({ name: "a.svg", mime_type: "image/svg+xml" }))).toBe(false);
    expect(
      hasImageThumbnail(file({ name: "a.png", mime_type: "image/png", size_bytes: 10_000_001 })),
    ).toBe(false);
    expect(hasImageThumbnail(file({}))).toBe(false);
  });
});

describe("chatFiles", () => {
  const made = file({ id: "made", origin: "generated" });
  const attached = file({ id: "attached", origin: "uploaded" });

  it("lists files made in the chat before attached ones", () => {
    expect(chatFiles([attached, made], null).map((f) => f.id)).toEqual(["made", "attached"]);
  });

  it("leaves out what the expert's documents already list", () => {
    expect(chatFiles([made, attached], [made]).map((f) => f.id)).toEqual(["attached"]);
  });
});
