import { getWebSocketToken } from "@/lib/supabase/actions";
import { environment } from "@/services/environment";

interface UploadFileResponse {
  file_id: string;
  name: string;
  path: string;
  mime_type: string;
  size_bytes: number;
}

/**
 * Upload a file directly to the Python backend, bypassing the Next.js proxy.
 * The Next.js serverless proxy has a ~4.5MB body size limit (Vercel) which
 * rejects larger files with HTTP 413.
 */
export async function uploadFileDirect(
  file: File,
  sessionID?: string,
): Promise<UploadFileResponse> {
  const { token, error: tokenError } = await getWebSocketToken();
  if (tokenError || !token) {
    throw new Error("Authentication error — please sign in again.");
  }

  const backendBase = environment.getAGPTServerBaseUrl();
  const url = new URL("/api/workspace/files/upload", backendBase);
  if (sessionID) {
    url.searchParams.set("session_id", sessionID);
  }
  url.searchParams.set("overwrite", "true");

  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(url.toString(), {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
    body: formData,
  });

  if (!res.ok) {
    let message: string;
    try {
      const body = await res.json();
      // Backend returns { detail: "..." } or { detail: { message: "..." } }
      message =
        typeof body.detail === "string"
          ? body.detail
          : (body.detail?.message ?? res.statusText);
    } catch {
      message = res.statusText;
    }
    throw new Error(message);
  }

  return res.json();
}
