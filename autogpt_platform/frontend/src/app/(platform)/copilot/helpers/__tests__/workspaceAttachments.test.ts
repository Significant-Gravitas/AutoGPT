import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import { describe, expect, it } from "vitest";
import {
  type Attachment,
  attachmentName,
  buildWorkspaceFilePart,
  partitionAttachments,
  workspaceFileDownloadUrl,
  workspaceItemToAttachment,
} from "../workspaceAttachments";

function makeWorkspaceItem(
  overrides: Partial<WorkspaceFileItem> = {},
): WorkspaceFileItem {
  return {
    id: "file-1",
    name: "report.pdf",
    path: "/workspace/report.pdf",
    mime_type: "application/pdf",
    size_bytes: 1024,
    origin: "uploaded",
    created_at: "2026-01-01T00:00:00Z",
    ...overrides,
  };
}

describe("workspaceAttachments", () => {
  it("builds the workspace download URL from a file id", () => {
    expect(workspaceFileDownloadUrl("abc")).toBe(
      "/api/proxy/api/workspace/files/abc/download",
    );
  });

  it("maps a workspace file item to a workspace attachment", () => {
    const attachment = workspaceItemToAttachment(makeWorkspaceItem());
    expect(attachment).toEqual({
      kind: "workspace",
      fileId: "file-1",
      name: "report.pdf",
      mimeType: "application/pdf",
    });
  });

  it("resolves the display name for both attachment kinds", () => {
    const local: Attachment = {
      kind: "local",
      file: new File(["x"], "local.txt", { type: "text/plain" }),
    };
    const workspace = workspaceItemToAttachment(makeWorkspaceItem());
    expect(attachmentName(local)).toBe("local.txt");
    expect(attachmentName(workspace)).toBe("report.pdf");
  });

  it("builds a FileUIPart that points at the workspace download URL", () => {
    const part = buildWorkspaceFilePart({
      fileId: "file-9",
      name: "data.csv",
      mimeType: "text/csv",
    });
    expect(part).toEqual({
      type: "file",
      mediaType: "text/csv",
      filename: "data.csv",
      url: "/api/proxy/api/workspace/files/file-9/download",
    });
  });

  it("partitions mixed attachments into local files and workspace refs", () => {
    const localFile = new File(["x"], "local.txt", { type: "text/plain" });
    const attachments: Attachment[] = [
      { kind: "local", file: localFile },
      workspaceItemToAttachment(makeWorkspaceItem({ id: "ws-1" })),
    ];

    const { localFiles, workspaceFiles } = partitionAttachments(attachments);

    expect(localFiles).toEqual([localFile]);
    expect(workspaceFiles).toEqual([
      { fileId: "ws-1", name: "report.pdf", mimeType: "application/pdf" },
    ]);
  });
});
