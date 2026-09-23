import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import { describe, expect, it } from "vitest";
import openapiSpec from "@/app/api/openapi.json";
import {
  type Attachment,
  appendWithinCap,
  attachmentName,
  buildWorkspaceFilePart,
  MAX_ATTACHMENTS,
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

describe("appendWithinCap", () => {
  it("keeps exactly MAX_ATTACHMENTS and reports the rest as refused", () => {
    const incoming = makeLocals(MAX_ATTACHMENTS + 3);

    const { next, refused } = appendWithinCap([], incoming);

    expect(next).toHaveLength(MAX_ATTACHMENTS);
    expect(refused).toBe(3);
    // The first ones in are the ones kept, so the user sees what they picked.
    expect(next.map(attachmentName)).toEqual(
      incoming.slice(0, MAX_ATTACHMENTS).map(attachmentName),
    );
  });

  it("counts what is already attached against the cap", () => {
    const prev = makeLocals(MAX_ATTACHMENTS - 2);

    const { next, refused } = appendWithinCap(prev, makeLocals(5, "late"));

    expect(next).toHaveLength(MAX_ATTACHMENTS);
    expect(refused).toBe(3);
  });

  it("refuses nothing when everything fits", () => {
    const { next, refused } = appendWithinCap(
      makeLocals(2),
      makeLocals(3, "b"),
    );

    expect(next).toHaveLength(5);
    expect(refused).toBe(0);
  });

  it("skips a workspace file already attached without counting it as refused", () => {
    const already = workspaceItemToAttachment(
      makeWorkspaceItem({ id: "ws-1" }),
    );

    const { next, refused } = appendWithinCap([already], [already]);

    expect(next).toEqual([already]);
    expect(refused).toBe(0);
  });

  it("does not let a duplicate consume a slot a new file could have used", () => {
    const already = workspaceItemToAttachment(
      makeWorkspaceItem({ id: "ws-1" }),
    );
    const fresh = workspaceItemToAttachment(makeWorkspaceItem({ id: "ws-2" }));

    const { next, refused } = appendWithinCap([already], [already, fresh]);

    expect(next).toEqual([already, fresh]);
    expect(refused).toBe(0);
  });

  it("leaves the caller's array alone", () => {
    const prev = makeLocals(1);

    appendWithinCap(prev, makeLocals(2, "b"));

    expect(prev).toHaveLength(1);
  });

  it("keeps the first free slots of a workspace batch and counts the rest", () => {
    const prev = makeLocals(MAX_ATTACHMENTS - 4);
    const picked = Array.from({ length: 9 }, (_, i) =>
      workspaceItemToAttachment(makeWorkspaceItem({ id: `ws-${i}` })),
    );

    const { next, refused } = appendWithinCap(prev, picked);

    expect(next).toHaveLength(MAX_ATTACHMENTS);
    expect(refused).toBe(5);
    expect(next.slice(MAX_ATTACHMENTS - 4)).toEqual(picked.slice(0, 4));
  });
});

describe("the composer cap against the backend's own", () => {
  // The workflow-import part is sent by an autosubmit carrying no composer
  // attachments, so the two never share a request and the caps can be equal.
  it.each(["StreamChatRequest", "QueuePendingMessageRequest"])(
    "equals %s.file_ids maxItems",
    (schemaName) => {
      expect(fileIdsMaxItems(schemaName)).toBe(MAX_ATTACHMENTS);
    },
  );
});

function makeLocals(count: number, prefix = "a"): Attachment[] {
  return Array.from({ length: count }, (_, i) => ({
    kind: "local" as const,
    file: new File(["x"], `${prefix}-${i}.txt`, { type: "text/plain" }),
  }));
}

function fileIdsMaxItems(schemaName: string): number {
  const schemas = (
    openapiSpec as unknown as {
      components: { schemas: Record<string, SchemaWithFileIds> };
    }
  ).components.schemas;
  const fileIds = schemas[schemaName]?.properties?.file_ids;
  const arrayBranch = (fileIds?.anyOf ?? []).find(
    (branch) => branch.type === "array",
  );
  expect(arrayBranch?.maxItems).toBeTypeOf("number");
  return arrayBranch!.maxItems!;
}

interface SchemaWithFileIds {
  properties?: {
    file_ids?: { anyOf?: { type?: string; maxItems?: number }[] };
  };
}
