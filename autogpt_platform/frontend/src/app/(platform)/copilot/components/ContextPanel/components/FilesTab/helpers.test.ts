import { describe, expect, test } from "vitest";
import { downloadFilesAsZip } from "./helpers";
import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import {
  fileItemToArtifactRef,
  formatFileSize,
  formatFileTimestamp,
  isUploadedFile,
} from "./helpers";

const baseItem: WorkspaceFileItem = {
  id: "11111111-1111-1111-1111-111111111111",
  name: "report.pdf",
  path: "/sessions/s1/report.pdf",
  mime_type: "application/pdf",
  size_bytes: 2048,
  metadata: { origin: "user-upload" },
  origin: "uploaded",
  created_at: "2026-05-20T10:00:00Z",
};

describe("FilesTab helpers", () => {
  test("formatFileSize renders human units", () => {
    expect(formatFileSize(0)).toBe("0 B");
    expect(formatFileSize(2048)).toBe("2 KB");
    expect(formatFileSize(5 * 1024 * 1024)).toBe("5 MB");
  });
  test("isUploadedFile splits by origin", () => {
    expect(isUploadedFile(baseItem)).toBe(true);
    expect(isUploadedFile({ ...baseItem, origin: "generated" })).toBe(false);
  });
  test("fileItemToArtifactRef builds a proxy download URL + origin", () => {
    const ref = fileItemToArtifactRef(baseItem);
    expect(ref.id).toBe(baseItem.id);
    expect(ref.title).toBe("report.pdf");
    expect(ref.origin).toBe("user-upload");
    expect(ref.sourceUrl).toContain("/api/proxy");
    expect(ref.sourceUrl).toContain(baseItem.id);
    expect(ref.sizeBytes).toBe(2048);
  });
  test("formatFileTimestamp returns a non-empty string", () => {
    expect(formatFileTimestamp(baseItem.created_at).length).toBeGreaterThan(0);
  });
});

describe("downloadFilesAsZip", () => {
  test("fetches each file and produces a zip blob via the sink", async () => {
    const fetched: string[] = [];
    const fetchImpl = async (url: string) => {
      fetched.push(url);
      return new Response(new Blob([url]), { status: 200 });
    };
    let savedName = "";
    let savedBlob: Blob | null = null;
    await downloadFilesAsZip(
      [
        { id: "a1", name: "a.txt" },
        { id: "b2", name: "b.txt" },
      ],
      {
        fetchImpl,
        save: (blob, name) => {
          savedBlob = blob;
          savedName = name;
        },
      },
    );
    expect(fetched).toHaveLength(2);
    expect(savedName.endsWith(".zip")).toBe(true);
    expect(savedBlob).toBeInstanceOf(Blob);
  });
});
