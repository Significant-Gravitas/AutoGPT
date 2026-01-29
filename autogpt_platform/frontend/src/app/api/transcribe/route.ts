import { getServerAuthToken } from "@/lib/autogpt-server-api/helpers";
import { NextRequest, NextResponse } from "next/server";

const WHISPER_API_URL = "https://api.openai.com/v1/audio/transcriptions";
const MAX_FILE_SIZE = 25 * 1024 * 1024; // 25MB - Whisper's limit

function getExtensionFromMimeType(mimeType: string): string {
  const subtype = mimeType.split("/")[1]?.split(";")[0];
  return subtype || "webm";
}

export async function POST(request: NextRequest) {
  const token = await getServerAuthToken();

  if (!token || token === "no-token-found") {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const apiKey = process.env.OPENAI_API_KEY;

  if (!apiKey) {
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
    whisperFormData.append("model", "whisper-1");

    const response = await fetch(WHISPER_API_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
      },
      body: whisperFormData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      console.error("Whisper API error:", errorData);
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
