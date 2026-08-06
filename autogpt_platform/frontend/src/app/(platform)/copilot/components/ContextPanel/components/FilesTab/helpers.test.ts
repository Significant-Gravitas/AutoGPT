import { describe, expect, test } from "vitest";
import { downloadFilesAsZip } from "./helpers";
import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import {
  fileItemToArtifactRef,
  formatFileSize,
  formatFileTimestamp,
  isInternalToolOutput,
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

describe("isInternalToolOutput", () => {
  function generated(name: string, path?: string): WorkspaceFileItem {
    return {
      ...baseItem,
      name,
      path: path ?? `/sessions/s1/${name}`,
      mime_type: "application/json",
      metadata: {},
      origin: "generated",
    };
  }

  test("matches SDK tool result files by name", () => {
    expect(isInternalToolOutput(generated("toolu_01ABCdef.json"))).toBe(true);
    expect(isInternalToolOutput(generated("mcp_a1b2-c3d4.json"))).toBe(true);
    expect(isInternalToolOutput(generated("TOOLU_01ABC.JSON"))).toBe(true);
  });

  test("matches files living under an SDK tool result directory", () => {
    expect(
      isInternalToolOutput(
        generated(
          "summary.json",
          "/root/.claude/projects/-workspace/abc-123/tool-results/summary.json",
        ),
      ),
    ).toBe(true);
    expect(
      isInternalToolOutput(
        generated(
          "summary.json",
          ".claude/projects/-workspace/abc-123/tool-outputs/summary.json",
        ),
      ),
    ).toBe(true);
  });

  test("leaves user-facing files alone", () => {
    expect(isInternalToolOutput(generated("report.json"))).toBe(false);
    expect(isInternalToolOutput(generated("result.csv"))).toBe(false);
    expect(isInternalToolOutput(generated("toolu_notes.txt"))).toBe(false);
    expect(isInternalToolOutput(generated("toolusage.json"))).toBe(false);
    expect(isInternalToolOutput(generated("mcpserver.json"))).toBe(false);
    expect(
      isInternalToolOutput(
        generated("notes.json", "/sessions/s1/my-tool-results-archive.json"),
      ),
    ).toBe(false);
  });

  // The backend classifier refuses to treat a bare `tool-outputs/` segment as
  // SDK-internal; a user deliverable that happens to live in such a folder must
  // still be eligible for auto-open.
  test("leaves user deliverables under a tool-outputs directory alone", () => {
    expect(
      isInternalToolOutput(
        generated(
          "data.json",
          "/sessions/s1/my-pipeline/tool-outputs/data.json",
        ),
      ),
    ).toBe(false);
    expect(
      isInternalToolOutput(
        generated("data.json", "/sessions/s1/tool-results/data.json"),
      ),
    ).toBe(false);
  });

  test("never hides files the user uploaded themselves", () => {
    expect(
      isInternalToolOutput({ ...baseItem, name: "toolu_01ABCdef.json" }),
    ).toBe(false);
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
