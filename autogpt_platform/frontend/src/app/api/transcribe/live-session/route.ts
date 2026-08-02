import { getServerAuthToken } from "@/lib/auth/server/getServerAuthToken";
import { NextRequest, NextResponse } from "next/server";

// Mints a short-lived token so the browser can stream mic audio DIRECTLY
// to the transcription provider for live captions — no proxy hop, and the
// real API keys never leave the server. Two providers, A/B-switched by
// NEXT_PUBLIC_LIVE_CAPTIONS_ENGINE on the client:
//  - elevenlabs: Scribe v2 Realtime, ~150ms partials, strongest accuracy
//  - deepgram:   nova-3, fast but looser on words

const DEEPGRAM_GRANT_URL = "https://api.deepgram.com/v1/auth/grant";
const ELEVENLABS_TOKEN_URL =
  "https://api.elevenlabs.io/v1/single-use-token/realtime_scribe";
// Deepgram tokens: long enough to cover one recording, short enough that
// a leaked token is worthless soon after. ElevenLabs single-use tokens
// are fixed at 15 minutes and consumed on connect.
const TOKEN_TTL_SECONDS = 600;

export async function POST(request: NextRequest) {
  const authToken = await getServerAuthToken();
  if (!authToken) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { provider } = (await request.json().catch(() => ({}))) as {
    provider?: string;
  };
  return provider === "deepgram" ? mintDeepgram() : mintElevenLabs();
}

async function mintElevenLabs() {
  const apiKey = process.env.ELEVENLABS_API_KEY;
  if (!apiKey) return notConfigured();

  const response = await fetch(ELEVENLABS_TOKEN_URL, {
    method: "POST",
    headers: { "xi-api-key": apiKey },
  });
  if (!response.ok) return mintFailed("elevenlabs", response);

  const data = (await response.json()) as { token?: string };
  if (!data.token) return mintFailed("elevenlabs");
  return NextResponse.json({ token: data.token });
}

async function mintDeepgram() {
  const apiKey = process.env.DEEPGRAM_API_KEY;
  if (!apiKey) return notConfigured();

  const response = await fetch(DEEPGRAM_GRANT_URL, {
    method: "POST",
    headers: {
      Authorization: `Token ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ ttl_seconds: TOKEN_TTL_SECONDS }),
  });
  if (!response.ok) return mintFailed("deepgram", response);

  const data = (await response.json()) as { access_token?: string };
  if (!data.access_token) return mintFailed("deepgram");
  return NextResponse.json({ token: data.access_token });
}

function notConfigured() {
  return NextResponse.json(
    { error: "Live transcription is not configured" },
    { status: 503 },
  );
}

async function mintFailed(provider: string, response?: Response) {
  console.error(
    `Live transcription token mint failed (${provider}):`,
    response?.status,
    await response?.text(),
  );
  return NextResponse.json(
    { error: "Could not start live transcription" },
    { status: 502 },
  );
}
