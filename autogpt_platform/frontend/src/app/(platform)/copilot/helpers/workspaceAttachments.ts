import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import type { WorkspaceFolder } from "@/app/api/__generated__/models/workspaceFolder";
import type { FileUIPart } from "ai";

/**
 * A file already stored in the user's workspace. Unlike a freshly picked
 * local `File`, it already has an id on the backend, so attaching it to a
 * message needs no upload — we build a `FileUIPart` pointing straight at the
 * workspace download URL.
 */
export interface WorkspaceFileAttachment {
  fileId: string;
  name: string;
  mimeType: string;
}

/**
 * A folder named for the model to open with `list_workspace_files`. It is
 * never expanded into its files, so it costs one attachment whatever it holds.
 */
export interface WorkspaceFolderAttachment {
  folderId: string;
  name: string;
  fileCount: number;
  subfolderCount: number;
}

/** What crosses `onSend`: a stored file or a folder pointer. */
export type WorkspaceAttachment =
  | ({ kind: "workspace" } & WorkspaceFileAttachment)
  | ({ kind: "folder" } & WorkspaceFolderAttachment);

/**
 * A chat-composer attachment: a local file awaiting upload, a reference to an
 * existing workspace file, or a folder.
 */
export type Attachment = { kind: "local"; file: File } | WorkspaceAttachment;

/**
 * The backend caps `folder_ids` at 5 per message. Without the same ceiling
 * here the composer would build a message the send answers with a 422.
 * A folder also costs one of the `MAX_ATTACHMENTS`.
 */
export const MAX_FOLDER_ATTACHMENTS = 5;

export const WORKSPACE_FOLDER_PART_TYPE = "data-workspace-folder";

/** A message part for an attachment that needed no upload. */
export type StoredAttachmentPart =
  | FileUIPart
  | ReturnType<typeof buildWorkspaceFolderPart>;

export interface WorkspaceFolderPartData {
  id: string;
  name: string;
  fileCount: number;
}

// Must equal `file_ids` max_length on StreamChatRequest/QueuePendingMessageRequest
// (chat/routes.py): every attachment, uploaded or from the workspace, is one entry.
export const MAX_ATTACHMENTS = 20;

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

export function workspaceFolderToAttachment(
  folder: WorkspaceFolder,
  subfolderCount: number,
): Attachment {
  return {
    kind: "folder",
    folderId: folder.id,
    name: folder.name,
    fileCount: folder.file_count ?? 0,
    subfolderCount,
  };
}

export function attachmentName(attachment: Attachment): string {
  return attachment.kind === "local" ? attachment.file.name : attachment.name;
}

/** Stable identity for de-duplication and for `AnimatePresence` keys. */
export function attachmentKey(attachment: Attachment): string {
  if (attachment.kind === "workspace") return `ws-${attachment.fileId}`;
  if (attachment.kind === "folder") return `folder-${attachment.folderId}`;
  const { file } = attachment;
  return `local-${file.name}-${file.size}-${file.lastModified}`;
}

export function buildWorkspaceFilePart(
  attachment: WorkspaceFileAttachment,
): FileUIPart {
  return {
    type: "file",
    mediaType: attachment.mimeType,
    filename: attachment.name,
    url: workspaceFileDownloadUrl(attachment.fileId),
  };
}

export function buildWorkspaceFolderPart(
  attachment: WorkspaceFolderAttachment,
) {
  return {
    type: WORKSPACE_FOLDER_PART_TYPE,
    data: {
      id: attachment.folderId,
      name: attachment.name,
      fileCount: attachment.fileCount,
    },
  } as const;
}

/**
 * Adds attachments the composer does not already hold, refusing anything past
 * `MAX_ATTACHMENTS` and folders past `MAX_FOLDER_ATTACHMENTS`. A re-picked
 * attachment is skipped without being reported as refused: the user asked for
 * it and it is there.
 */
export function appendWithinCap(
  held: Attachment[],
  incoming: Attachment[],
): { next: Attachment[]; refused: number; refusedFolders: number } {
  const keys = new Set(held.map(attachmentKey));
  const next = [...held];
  let folders = held.filter((a) => a.kind === "folder").length;
  let refused = 0;
  let refusedFolders = 0;

  for (const attachment of incoming) {
    if (keys.has(attachmentKey(attachment))) continue;
    if (next.length >= MAX_ATTACHMENTS) {
      refused += 1;
      continue;
    }
    if (attachment.kind === "folder") {
      if (folders >= MAX_FOLDER_ATTACHMENTS) {
        refusedFolders += 1;
        continue;
      }
      folders += 1;
    }
    keys.add(attachmentKey(attachment));
    next.push(attachment);
  }
  return { next, refused, refusedFolders };
}

export function partitionAttachments(attachments: Attachment[]): {
  localFiles: File[];
  workspaceAttachments: WorkspaceAttachment[];
} {
  const localFiles: File[] = [];
  const workspaceAttachments: WorkspaceAttachment[] = [];
  for (const attachment of attachments) {
    if (attachment.kind === "local") localFiles.push(attachment.file);
    else workspaceAttachments.push(attachment);
  }
  return { localFiles, workspaceAttachments };
}

/** Message parts for the attachments that need no upload. */
export function buildStoredAttachmentParts(
  attachments: WorkspaceAttachment[],
): StoredAttachmentPart[] {
  return attachments.map((attachment) =>
    attachment.kind === "folder"
      ? buildWorkspaceFolderPart(attachment)
      : buildWorkspaceFilePart(attachment),
  );
}

export function isFolderPart(
  part: StoredAttachmentPart,
): part is ReturnType<typeof buildWorkspaceFolderPart> {
  return part.type === WORKSPACE_FOLDER_PART_TYPE;
}
