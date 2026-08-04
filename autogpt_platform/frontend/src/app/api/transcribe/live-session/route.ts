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
// Deepgram tokens are minted immediately before the socket opens and the
// socket authenticates once at handshake, so the TTL only has to cover
// that handshake — not the recording, which can outrun any grant. Keeping
// it near Deepgram's own default limits what a lifted token (it is visible
// to the browser) can spend. ElevenLabs single-use tokens are fixed at 15
// minutes and consumed on connect.
const TOKEN_TTL_SECONDS = 60;
// A stalled provider must not hold the route open until the platform's
// own limit fires: failing fast lets useLiveCaptions drop to the browser
// engine while the user is still talking.
const UPSTREAM_TIMEOUT_MS = 5000;

export async function POST(request: NextRequest) {
  const authToken = await getServerAuthToken();
  if (!authToken) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { provider } = (await request.json().catch(() => ({}))) as {
    provider?: string;
  };
  try {
    return provider === "deepgram"
      ? await mintDeepgram()
      : await mintElevenLabs();
  } catch (error) {
    // Includes the timeout aborting. Captions are cosmetic, so a dead
    // provider is a 502 and a browser-engine fallback, never a failed
    // onboarding step.
    console.error("Live transcription token mint errored:", error);
    return NextResponse.json(
      { error: "Could not start live transcription" },
      { status: 502 },
    );
  }
}

async function mintElevenLabs() {
  const apiKey = process.env.ELEVENLABS_API_KEY;
  if (!apiKey) return notConfigured();

  const response = await fetch(ELEVENLABS_TOKEN_URL, {
    method: "POST",
    headers: { "xi-api-key": apiKey },
    signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
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
    signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
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
