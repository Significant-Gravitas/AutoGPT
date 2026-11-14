import {
  decodeUtf8,
  isPreviewed,
  PREVIEW_BYTES,
  trimPartialUtf8,
} from "@/features/files/helpers";
import { clearAccessToken } from "@/lib/auth/token";

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
};

export async function listSessionFiles(sessionID: string) {
  const data = await apiRequest<FileList>("/api/workspace/files", {
    query: { session_id: sessionID },
  });
  return data.files;
}

export async function listExpertFiles(expertID: string) {
  const data = await apiRequest<FileList>("/api/workspace/files", {
    query: { expert_id: expertID, origin: "generated", limit: 200 },
  });
  return data.files;
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
  let response = await fetch(url, { ...init(), headers: await authHeaders() });
  // Same as apiRequest: one retry with a fresh token before the session counts as gone.
  if (response.status === 401) {
    clearAccessToken();
    response = await fetch(url, { ...init(), headers: await authHeaders() });
  }
  return response;
}
