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

export async function downloadFileText(fileID: string) {
  let response = await fetchFile(fileID);
  // Same as apiRequest: one retry with a fresh token before the session counts as gone.
  if (response.status === 401) {
    clearAccessToken();
    response = await fetchFile(fileID);
  }
  if (!response.ok) {
    throw new ApiError(response.status, `Download failed (${response.status})`);
  }
  return response.text();
}

async function fetchFile(fileID: string) {
  return fetch(fileDownloadURL(fileID), { headers: await authHeaders() });
}
