import { getWebSocketToken } from "@/lib/supabase/actions";
import { environment } from "@/services/environment";

interface WorkspaceUploadResponse {
  file_id: string;
  name: string;
  path: string;
  mime_type: string;
  size_bytes: number;
}

/**
 * Uploads here go straight to the Python backend, bypassing the Next.js
 * serverless proxy. The proxy has a ~4.5MB body size limit (Vercel) that
 * rejects larger files with HTTP 413 *before* the request reaches the backend,
 * where the 413 body carries no useful message. Going direct lets the real
 * backend size limits apply and surfaces the backend's own error messages.
 */

// Backend upload size limits (keep in sync with the backend):
// - store submission media (agent thumbnails, profile avatars): 50MB
//   (backend/api/features/store/media.py)
// - OAuth app logos: 3MB (backend/api/features/oauth.py)
export const SUBMISSION_MEDIA_MAX_SIZE_MB = 50;
export const OAUTH_LOGO_MAX_SIZE_MB = 3;

const BYTES_PER_MB = 1024 * 1024;
const UPLOAD_TIMEOUT_MS = 5 * 60 * 1000;

/**
 * Returns a user-facing message when a file exceeds `maxSizeMB`, or null when
 * it is within the limit. Lets callers reject oversized files before uploading.
 */
export function getFileSizeError(file: File, maxSizeMB: number): string | null {
  if (file.size <= maxSizeMB * BYTES_PER_MB) return null;
  const sizeMB = (file.size / BYTES_PER_MB).toFixed(1);
  return `File is too large (${sizeMB}MB). Maximum size is ${maxSizeMB}MB — please choose a smaller file.`;
}

interface FileSizeGuardArgs {
  file: File;
  maxSizeMB: number;
  toast: (options: {
    title: string;
    description?: string;
    variant?: "destructive";
  }) => void;
}

export function isFileTooLarge({
  file,
  maxSizeMB,
  toast,
}: FileSizeGuardArgs): boolean {
  const error = getFileSizeError(file, maxSizeMB);
  if (!error) return false;
  toast({
    title: "File too large",
    description: error,
    variant: "destructive",
  });
  return true;
}

interface DirectUploadArgs {
  path: string;
  file: File;
  searchParams?: Record<string, string>;
}

async function postFileToBackend({
  path,
  file,
  searchParams,
}: DirectUploadArgs): Promise<Response> {
  const { token, error: tokenError } = await getWebSocketToken();
  if (tokenError || !token) {
    throw new Error("Authentication error — please sign in again.");
  }

  const url = new URL(path, environment.getAGPTServerBaseUrl());
  for (const [key, value] of Object.entries(searchParams ?? {})) {
    url.searchParams.set(key, value);
  }

  const formData = new FormData();
  formData.append("file", file);

  return fetch(url.toString(), {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
    body: formData,
    // Guard against a stalled connection leaving the UI stuck "Uploading…".
    // Generous so large (up to 50MB) uploads on slow links aren't cut off.
    signal: AbortSignal.timeout(UPLOAD_TIMEOUT_MS),
  });
}

async function readUploadError(res: Response): Promise<string> {
  // A proxy/ingress in front of the backend may still reject an oversized body
  // with a 413 whose payload isn't backend JSON — surface a clear size message
  // instead of a bare "HTTP 413".
  if (res.status === 413) {
    return "File is too large. Please choose a smaller file.";
  }
  const text = await res.text();
  try {
    const body = JSON.parse(text) as {
      detail?: string | { message?: string };
      message?: string;
    };
    // Backend returns { detail: "..." } or { detail: { message: "..." } }.
    const detail = body?.detail;
    if (typeof detail === "string") return detail;
    if (typeof detail?.message === "string") return detail.message;
    if (typeof body?.message === "string") return body.message;
  } catch {
    // Response body wasn't JSON — surface the raw text so it reaches the toast.
    if (text) return text;
  }
  return res.statusText || `Upload failed (HTTP ${res.status})`;
}

export async function uploadFileDirect(
  file: File,
  sessionID?: string,
): Promise<WorkspaceUploadResponse> {
  const res = await postFileToBackend({
    path: "/api/workspace/files/upload",
    file,
    searchParams: {
      overwrite: "true",
      ...(sessionID ? { session_id: sessionID } : {}),
    },
  });
  if (!res.ok) throw new Error(await readUploadError(res));
  return res.json();
}

/**
 * Uploads store submission media (agent thumbnails, profile avatars) directly
 * to the backend. Returns the public URL of the stored media.
 */
export async function uploadSubmissionMediaDirect(file: File): Promise<string> {
  const res = await postFileToBackend({
    path: "/api/store/submissions/media",
    file,
  });
  if (!res.ok) throw new Error(await readUploadError(res));
  // The endpoint returns the URL as a JSON string.
  const url = (await res.json()) as unknown;
  return typeof url === "string" ? url : String(url);
}

/** Uploads an OAuth application logo directly to the backend. */
export async function uploadOAuthAppLogoDirect(
  appId: string,
  file: File,
): Promise<void> {
  const res = await postFileToBackend({
    path: `/api/oauth/apps/${encodeURIComponent(appId)}/logo/upload`,
    file,
  });
  if (!res.ok) throw new Error(await readUploadError(res));
}
