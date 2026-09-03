import { environment } from "@/services/environment";

import { getCopilotAuthHeaders } from "../helpers";

/**
 * Synthesised audio, keyed by the exact text. Only the acknowledgement bank
 * ever repeats, and re-synthesising those is real money on every turn.
 */
const cache = new Map<string, Blob>();

export async function synthesizeSpeech(
  text: string,
  sessionId: string | null,
): Promise<Blob> {
  const cached = cache.get(text);
  if (cached) return cached;

  const response = await fetch(
    `${environment.getAGPTServerBaseUrl()}/api/chat/speech`,
    {
      method: "POST",
      headers: {
        ...(await getCopilotAuthHeaders()),
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text, session_id: sessionId }),
    },
  );

  if (!response.ok) {
    throw new Error(`Speech synthesis failed (${response.status})`);
  }

  const blob = await response.blob();
  if (cache.size < CACHE_LIMIT) cache.set(text, blob);
  return blob;
}

/** Enough for the acknowledgement bank plus a little headroom. */
const CACHE_LIMIT = 40;

export async function transcribeUtterance(audio: Blob): Promise<string> {
  const body = new FormData();
  body.append("audio", audio);

  const response = await fetch("/api/transcribe", { method: "POST", body });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.error || "Transcription failed");
  }

  const data = await response.json();
  return typeof data.text === "string" ? data.text : "";
}
