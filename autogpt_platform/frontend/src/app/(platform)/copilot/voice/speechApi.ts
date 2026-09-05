import { environment } from "@/services/environment";

import { getCopilotAuthHeaders } from "../helpers";

export async function synthesizeSpeech(
  text: string,
  sessionId: string | null,
): Promise<Blob> {
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

  return response.blob();
}

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
