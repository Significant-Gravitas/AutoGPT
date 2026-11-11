import { authClient } from "@/lib/auth/client";
import { FRONTEND_URL } from "@/lib/env";

/** A refused transcription; `status` is null when no answer came back. */
export class TranscribeError extends Error {
  status: number | null;

  constructor(status: number | null) {
    super(`Transcription failed (${status ?? "no response"})`);
    this.name = "TranscribeError";
    this.status = status;
  }
}

/**
 * Sends a recording to the web app's transcription route, which signs in by
 * the session cookie. The cookie goes to FRONTEND_URL only, set by hand with
 * `credentials: "omit"` so the native cookie store adds nothing of its own.
 */
export async function transcribe(uri: string, context?: string) {
  const cookie = await authClient.getCookie();
  const body = new FormData();
  // The route names the upstream file by the subtype: `recording.m4a`.
  body.append("audio", {
    uri,
    name: "recording.m4a",
    type: "audio/m4a",
  } as unknown as Blob);
  const draft = context?.trim();
  if (draft) body.append("context", draft);

  let response: Response;
  try {
    response = await fetch(`${FRONTEND_URL}/api/transcribe`, {
      method: "POST",
      headers: cookie ? { Cookie: cookie } : {},
      body,
      credentials: "omit",
    });
  } catch {
    throw new TranscribeError(null);
  }
  if (!response.ok) throw new TranscribeError(response.status);
  const data = (await response.json().catch(() => ({}))) as { text?: unknown };
  return typeof data.text === "string" ? data.text : "";
}
