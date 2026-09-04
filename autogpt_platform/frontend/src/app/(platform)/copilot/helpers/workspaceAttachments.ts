import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import type { FileUIPart } from "ai";

/**
 * A file already stored in the user's workspace. Unlike a freshly picked
 * local `File`, it already has an id on the backend, so attaching it to a
 * message needs no upload — we build a `FileUIPart` pointing straight at the
 * workspace download URL.
 */
export interface WorkspaceAttachment {
  fileId: string;
  name: string;
  mimeType: string;
}

/**
 * A chat-composer attachment: either a local file awaiting upload or a
 * reference to an existing workspace file.
 */
export type Attachment =
  | { kind: "local"; file: File }
  | ({ kind: "workspace" } & WorkspaceAttachment);

export function workspaceFileDownloadUrl(fileId: string): string {
  return `/api/proxy/api/workspace/files/${encodeURIComponent(fileId)}/download`;
}

export function workspaceItemToAttachment(item: WorkspaceFileItem): Attachment {
  return {
    kind: "workspace",
    fileId: item.id,
    name: item.name,
    mimeType: item.mime_type,
  };
}

export function attachmentName(attachment: Attachment): string {
  return attachment.kind === "local" ? attachment.file.name : attachment.name;
}

export function buildWorkspaceFilePart(
  attachment: WorkspaceAttachment,
): FileUIPart {
  return {
    type: "file",
    mediaType: attachment.mimeType,
    filename: attachment.name,
    url: workspaceFileDownloadUrl(attachment.fileId),
  };
}

export function partitionAttachments(attachments: Attachment[]): {
  localFiles: File[];
  workspaceFiles: WorkspaceAttachment[];
} {
  const localFiles: File[] = [];
  const workspaceFiles: WorkspaceAttachment[] = [];
  for (const attachment of attachments) {
    if (attachment.kind === "local") {
      localFiles.push(attachment.file);
    } else {
      workspaceFiles.push({
        fileId: attachment.fileId,
        name: attachment.name,
        mimeType: attachment.mimeType,
      });
    }
  }
  return { localFiles, workspaceFiles };
}
