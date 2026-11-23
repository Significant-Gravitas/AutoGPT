import {
  decodeUtf8,
  FULL_TEXT_LIMIT,
  isPreviewed,
  PREVIEW_BYTES,
  trimPartialUtf8,
} from "@/features/files/helpers";
import { assertSameSession, clearAccessToken, sessionMark } from "@/lib/auth/token";

import { ApiError, apiRequest, apiUrl, authHeaders } from "./client";

export interface WorkspaceFile {
  id: string;
  name: string;
  path: string;
  mime_type: string;
  size_bytes: number;
  folder_id: string | null;
  metadata: Record<string, unknown> | null;
  origin: "uploaded" | "generated" | null;
  created_at: string;
  expert_id: string | null;
  is_internal: boolean;
}

interface FileList {
  files: WorkspaceFile[];
  offset: number;
  has_more: boolean;
}

export const workspaceKeys = {
  sessionFiles: (sessionID: string) =>
    ["workspace", "files", "session", sessionID] as const,
  expertFiles: (expertID: string) =>
    ["workspace", "files", "expert", expertID] as const,
  fileText: (fileID: string) => ["workspace", "file-text", fileID] as const,
  imageHeaders: ["workspace", "image-headers"] as const,
  // Apart from the viewer's, whose no-cache policy a shared entry would override.
  thumbnailHeaders: ["workspace", "thumbnail-headers"] as const,
  library: (filter: LibraryFilter) => ["workspace", "files", "library", filter] as const,
  folders: ["workspace", "folders"] as const,
  storage: ["workspace", "storage"] as const,
};

/** A folder in the Files page (`WorkspaceFolder`). */
export interface WorkspaceFolder {
  id: string;
  name: string;
  parent_id: string | null;
  file_count: number;
  updated_at: string;
}

export interface StorageUsage {
  used_bytes: number;
  limit_bytes: number;
}

/** What the Files page lists: a folder's files, or the whole workspace's when searching. */
export interface LibraryFilter {
  q: string | null;
  origin: "uploaded" | null;
  expertID: string | null;
  folderID: string | null;
}

const LIBRARY_PAGE_SIZE = 50;

/** One page of the Files page's list. Without a search, the top level holds only the files in no folder. */
export function listLibraryPage(filter: LibraryFilter, offset: number) {
  return apiRequest<FileList>("/api/workspace/files", {
    query: {
      limit: LIBRARY_PAGE_SIZE,
      offset,
      q: filter.q ?? undefined,
      origin: filter.origin ?? undefined,
      expert_id: filter.expertID ?? undefined,
      folder_id: filter.folderID ?? undefined,
      root_only: !filter.folderID && !filter.q ? true : undefined,
    },
  });
}

export async function listFolders() {
  const data = await apiRequest<{ folders: WorkspaceFolder[] }>("/api/workspace/folders");
  return data.folders;
}

export function getStorageUsage() {
  return apiRequest<StorageUsage>("/api/workspace/storage/usage");
}

const FILES_PAGE_SIZE = 200;
// A guard against a server that keeps saying there is more.
const MAX_FILE_PAGES = 25;

export function listSessionFiles(sessionID: string) {
  return listAllFiles({ session_id: sessionID });
}

export function listExpertFiles(expertID: string) {
  return listAllFiles({ expert_id: expertID, origin: "generated" });
}

/** Every page of a file listing; the route answers 200 at a time. */
async function listAllFiles(filter: Record<string, string>) {
  const files: WorkspaceFile[] = [];
  for (let page = 0; page < MAX_FILE_PAGES; page += 1) {
    const data = await apiRequest<FileList>("/api/workspace/files", {
      query: { ...filter, limit: FILES_PAGE_SIZE, offset: files.length },
    });
    files.push(...data.files);
    if (!data.has_more || data.files.length === 0) break;
  }
  return files;
}

export function fileDownloadURL(fileID: string) {
  return apiUrl(`/api/workspace/files/${encodeURIComponent(fileID)}/download`);
}

export interface FileText {
  text: string;
  isPreview: boolean;
}

/** A large file comes from the preview route, which sends only its first bytes. */
export async function downloadFileText(
  file: WorkspaceFile,
): Promise<FileText> {
  if (isPreviewed(file)) {
    const response = await fetchWithFreshToken(filePreviewURL(file.id));
    // The preview route is paywalled and the download route is not, so a
    // refused preview falls back to the whole file.
    if (response.status !== 402) {
      ensureOk(response);
      const bytes = new Uint8Array(await response.arrayBuffer());
      return { text: decodeUtf8(trimPartialUtf8(bytes)), isPreview: true };
    }
  }
  // The whole file is read into memory, so a large one is refused first.
  if (file.size_bytes > FULL_TEXT_LIMIT) {
    throw new ApiError(413, "Too large to show");
  }
  const response = await fetchWithFreshToken(fileDownloadURL(file.id));
  ensureOk(response);
  return { text: await response.text(), isPreview: false };
}

/** The preview route can't cut this file: not a text type (415) or too big (413). */
export function isPreviewUnavailable(error: unknown) {
  return (
    error instanceof ApiError && (error.status === 413 || error.status === 415)
  );
}

/** A refusal (402, 413, 415) won't change on a second try. */
export function shouldRetryFileText(failureCount: number, error: unknown) {
  if (error instanceof ApiError && [402, 413, 415].includes(error.status)) {
    return false;
  }
  return failureCount < 1;
}

function filePreviewURL(fileID: string) {
  return apiUrl(`/api/workspace/files/${encodeURIComponent(fileID)}/preview`, {
    bytes: PREVIEW_BYTES,
  });
}

function ensureOk(response: Response) {
  if (!response.ok) {
    throw new ApiError(response.status, `Download failed (${response.status})`);
  }
}

interface UploadedFile {
  file_id: string;
  name: string;
  path: string;
  mime_type: string;
  size_bytes: number;
}

interface LocalFile {
  uri: string;
  name: string;
  mimeType: string;
}

/** Uploads a local file into the chat's folder, or the workspace when there is no chat yet. Throws an ApiError holding the server's detail. */
export async function uploadSessionFile(sessionID: string | null, file: LocalFile) {
  const url = apiUrl("/api/workspace/files/upload", {
    session_id: sessionID ?? undefined,
  });
  // A fresh body per request: React Native reads the file as it sends.
  const response = await fetchWithFreshToken(url, () => {
    const body = new FormData();
    // React Native's FormData takes a file by its URI.
    body.append("file", {
      uri: file.uri,
      name: file.name,
      type: file.mimeType,
    } as unknown as Blob);
    return { method: "POST", body };
  });
  if (!response.ok) {
    throw new ApiError(response.status, await readDetail(response));
  }
  return (await response.json()) as UploadedFile;
}

async function readDetail(response: Response) {
  try {
    const data = (await response.json()) as { detail?: unknown };
    if (typeof data.detail === "string") return data.detail;
  } catch {
    // Not JSON: no detail to read.
  }
  return "";
}

async function fetchWithFreshToken(
  url: string,
  init: () => RequestInit = () => ({}),
) {
  const mark = sessionMark();
  let response = await fetch(url, { ...init(), headers: await authHeaders() });
  // Same as apiRequest: one retry with a fresh token before the session counts
  // as gone, and none for an account that signed in after this started.
  if (response.status === 401) {
    assertSameSession(mark);
    clearAccessToken();
    response = await fetch(url, { ...init(), headers: await authHeaders() });
    assertSameSession(mark);
  }
  return response;
}
