import { getServerAuthToken } from "@/lib/auth/server/getServerAuthToken";
import { NextRequest, NextResponse } from "next/server";

const DEFAULT_TRANSCRIPTION_API_BASE_URL = "https://api.openai.com/v1";
const DEFAULT_TRANSCRIPTION_MODEL = "whisper-1";
const MAX_FILE_SIZE = 25 * 1024 * 1024; // 25MB - Whisper's limit
// The composer sends its current draft as `context`; it rides upstream as
// the transcription `prompt` so names already on screen are spelled the
// same way in the transcript. Whisper only reads the last ~224 tokens of a
// prompt, so a long draft is trimmed from the front rather than sent whole.
const MAX_CONTEXT_CHARS = 800;

function getTranscriptionApiBaseUrl(): string {
  return (
    process.env.TRANSCRIPTION_API_BASE_URL ||
    process.env.OPENAI_API_BASE_URL ||
    DEFAULT_TRANSCRIPTION_API_BASE_URL
  ).replace(/\/+$/, "");
}

function getTranscriptionApiUrl(): string {
  return `${getTranscriptionApiBaseUrl()}/audio/transcriptions`;
}

function getTranscriptionModel(): string {
  return process.env.TRANSCRIPTION_MODEL || DEFAULT_TRANSCRIPTION_MODEL;
}

function getTranscriptionApiKey(): string | undefined {
  if (process.env.TRANSCRIPTION_API_KEY) {
    return process.env.TRANSCRIPTION_API_KEY;
  }

  return usesDefaultOpenAIEndpoint() ? process.env.OPENAI_API_KEY : undefined;
}

function usesDefaultOpenAIEndpoint(): boolean {
  return (
    getTranscriptionApiBaseUrl().toLowerCase() ===
    DEFAULT_TRANSCRIPTION_API_BASE_URL
  );
}

function getExtensionFromMimeType(mimeType: string): string {
  const subtype = mimeType.split("/")[1]?.split(";")[0];
  return subtype || "webm";
}

export async function POST(request: NextRequest) {
  const token = await getServerAuthToken();

  if (!token) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const apiKey = getTranscriptionApiKey();

  if (!apiKey && usesDefaultOpenAIEndpoint()) {
    return NextResponse.json(
      { error: "OpenAI API key not configured" },
      { status: 401 },
    );
  }

  try {
    const formData = await request.formData();
    const audioFile = formData.get("audio");

    if (!audioFile || !(audioFile instanceof Blob)) {
      return NextResponse.json(
        { error: "No audio file provided" },
        { status: 400 },
      );
    }

    if (audioFile.size > MAX_FILE_SIZE) {
      return NextResponse.json(
        { error: "File too large. Maximum size is 25MB." },
        { status: 413 },
      );
    }

    const ext = getExtensionFromMimeType(audioFile.type);
    const whisperFormData = new FormData();
    whisperFormData.append("file", audioFile, `recording.${ext}`);
    whisperFormData.append("model", getTranscriptionModel());

    const context = formData.get("context");
    const prompt =
      typeof context === "string"
        ? context.trim().slice(-MAX_CONTEXT_CHARS)
        : "";
    if (prompt) {
      whisperFormData.append("prompt", prompt);
    }

    const headers = new Headers();
    if (apiKey) {
      headers.set("Authorization", `Bearer ${apiKey}`);
    }

    const response = await fetch(getTranscriptionApiUrl(), {
      method: "POST",
      headers,
      body: whisperFormData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      console.error("Transcription API error:", errorData);
      return NextResponse.json(
        { error: errorData.error?.message || "Transcription failed" },
        { status: response.status },
      );
    }

    const result = await response.json();
    return NextResponse.json({ text: result.text });
  } catch (error) {
    console.error("Transcription error:", error);
    return NextResponse.json(
      { error: "Failed to process audio" },
      { status: 500 },
    );
  }
}
