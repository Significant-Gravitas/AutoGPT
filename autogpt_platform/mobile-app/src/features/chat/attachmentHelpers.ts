// Relative import: jest has no `@/` alias.
import { PLAN_REQUIRED_MESSAGE } from "./chatErrors";

export const MAX_ATTACHMENTS = 10;
// The backend takes up to 100 MB (`max_file_size_mb`); a phone keeps to less.
export const MAX_ATTACHMENT_BYTES = 25 * 1024 * 1024;

export const TOO_LARGE_MESSAGE = "That file is too large to attach.";
export const TOO_MANY_MESSAGE = "You can attach up to 10 files.";
export const STORAGE_FULL_MESSAGE =
  "Your storage is full. Remove files on the web to attach more.";
export const REJECTED_MESSAGE = "That file can't be attached.";
export const UPLOAD_FAILED_MESSAGE = "Couldn't attach that file. Try again.";

export interface Attachment {
  id: string;
  name: string;
  isImage: boolean;
  status: "uploading" | "ready" | "failed";
  fileID?: string;
  error?: string;
}

export interface PickedFile {
  name: string;
  uri: string;
  mimeType: string;
  size?: number;
  isImage: boolean;
}

/** The picks that fit beside `count` attachments, and the line for any left out. */
export function admitPicks<T extends { size?: number }>(count: number, picks: T[]) {
  const accepted: T[] = [];
  let notice: string | null = null;
  for (const pick of picks) {
    if (pick.size !== undefined && pick.size > MAX_ATTACHMENT_BYTES) {
      notice ??= TOO_LARGE_MESSAGE;
    } else if (count + accepted.length >= MAX_ATTACHMENTS) {
      notice = TOO_MANY_MESSAGE;
    } else {
      accepted.push(pick);
    }
  }
  return { accepted, notice };
}

/**
 * The sentence for a refused upload. The upload route answers 413 for both a
 * file over its cap and a full storage quota, told apart by the detail; 400
 * is the virus scan.
 */
export function uploadErrorMessage(status: number | null, detail = "") {
  if (status === 413) {
    return detail.startsWith("Storage limit")
      ? STORAGE_FULL_MESSAGE
      : TOO_LARGE_MESSAGE;
  }
  if (status === 400) return REJECTED_MESSAGE;
  if (status === 402) return PLAN_REQUIRED_MESSAGE;
  return UPLOAD_FAILED_MESSAGE;
}

/** The last segment of a file URI, for a photo that came without a name. */
export function nameFromUri(uri: string, fallback: string) {
  const last = uri.split(/[/\\]/).pop() ?? "";
  try {
    return decodeURIComponent(last) || fallback;
  } catch {
    return last || fallback;
  }
}

/** True while the send button has to wait for an upload. */
export function isUploading(attachments: Attachment[]) {
  return attachments.some((item) => item.status === "uploading");
}

export function readyFiles(attachments: Attachment[]) {
  return attachments.flatMap((item) =>
    item.status === "ready" && item.fileID
      ? [{ id: item.fileID, name: item.name }]
      : [],
  );
}

export interface MessageFiles {
  fileIDs: string[];
  fileNames: string[];
}

export function messageFiles(metadata: unknown): MessageFiles {
  if (typeof metadata !== "object" || metadata === null) {
    return { fileIDs: [], fileNames: [] };
  }
  const { fileIDs, fileNames } = metadata as Record<string, unknown>;
  return { fileIDs: stringList(fileIDs), fileNames: stringList(fileNames) };
}

function stringList(value: unknown) {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is string => typeof item === "string");
}
