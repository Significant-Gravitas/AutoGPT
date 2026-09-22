import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import { describe, expect, it } from "vitest";
import {
  type Attachment,
  type WorkspaceAttachment,
  attachmentName,
  appendWithinCap,
  buildStoredAttachmentParts,
  buildWorkspaceFilePart,
  buildWorkspaceFolderPart,
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

    const { localFiles, workspaceAttachments } =
      partitionAttachments(attachments);

    expect(localFiles).toEqual([localFile]);
    expect(workspaceAttachments).toEqual([
      {
        kind: "workspace",
        fileId: "ws-1",
        name: "report.pdf",
        mimeType: "application/pdf",
      },
    ]);
  });
});

function makeFolderAttachment(id: string, name = "Q3"): WorkspaceAttachment {
  return {
    kind: "folder",
    folderId: id,
    name,
    fileCount: 3,
    subfolderCount: 0,
  };
}

describe("folder attachments", () => {
  it("names a folder the same way a file is named", () => {
    expect(attachmentName(makeFolderAttachment("fld-1"))).toBe("Q3");
  });

  it("builds a data part the model reads as a folder id", () => {
    expect(
      buildWorkspaceFolderPart({
        folderId: "fld-1",
        name: "Q3",
        fileCount: 3,
        subfolderCount: 1,
      }),
    ).toEqual({
      type: "data-workspace-folder",
      data: { id: "fld-1", name: "Q3", fileCount: 3 },
    });
  });

  it("partitions a folder alongside the files", () => {
    const { localFiles, workspaceAttachments } = partitionAttachments([
      makeFolderAttachment("fld-1"),
      workspaceItemToAttachment(makeWorkspaceItem({ id: "ws-1" })),
    ]);
    expect(localFiles).toEqual([]);
    expect(workspaceAttachments).toHaveLength(2);
    expect(workspaceAttachments[0].kind).toBe("folder");
  });

  it("accepts folders up to the cap and refuses the rest", () => {
    const five = Array.from({ length: 5 }, (_, i) =>
      makeFolderAttachment(`fld-${i}`, `F${i}`),
    );
    const first = appendWithinCap([], five);
    expect(first.attachments).toHaveLength(5);
    expect(first.refusedFolders).toBe(0);

    const sixth = appendWithinCap(first.attachments, [
      makeFolderAttachment("fld-5", "F5"),
    ]);
    expect(sixth.attachments).toHaveLength(5);
    expect(sixth.refusedFolders).toBe(1);
  });

  it("skips a re-picked folder without counting it as refused", () => {
    const held = [makeFolderAttachment("fld-1")];
    const again = appendWithinCap(held, [makeFolderAttachment("fld-1")]);
    expect(again.attachments).toHaveLength(1);
    expect(again.refusedFolders).toBe(0);
  });

  it("lets files past the folder cap, which only counts folders", () => {
    const five = Array.from({ length: 5 }, (_, i) =>
      makeFolderAttachment(`fld-${i}`, `F${i}`),
    );
    const withFile = appendWithinCap(five, [
      workspaceItemToAttachment(makeWorkspaceItem({ id: "ws-9" })),
    ]);
    expect(withFile.attachments).toHaveLength(6);
    expect(withFile.refusedFolders).toBe(0);
  });

  it("builds one part per stored attachment, folder or file", () => {
    const parts = buildStoredAttachmentParts([
      makeFolderAttachment("fld-1"),
      {
        kind: "workspace",
        fileId: "ws-1",
        name: "a.csv",
        mimeType: "text/csv",
      },
    ]);
    expect(parts.map((p) => p.type)).toEqual(["data-workspace-folder", "file"]);
  });
});
