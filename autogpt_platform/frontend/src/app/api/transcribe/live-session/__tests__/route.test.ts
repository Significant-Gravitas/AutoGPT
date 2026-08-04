import type { NextRequest } from "next/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/auth/server/getServerAuthToken", () => ({
  getServerAuthToken: vi.fn(),
}));

import { getServerAuthToken } from "@/lib/auth/server/getServerAuthToken";
import { POST } from "../route";

const ELEVENLABS_KEY = "xi-secret-key-do-not-leak";
const DEEPGRAM_KEY = "dg-secret-key-do-not-leak";

function makeRequest(body?: unknown): NextRequest {
  return new Request("http://localhost/api/transcribe/live-session", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? "" : JSON.stringify(body),
  }) as NextRequest;
}

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

let fetchMock: ReturnType<typeof vi.fn>;

beforeEach(() => {
  vi.mocked(getServerAuthToken).mockResolvedValue("session-token");
  fetchMock = vi.fn();
  vi.stubGlobal("fetch", fetchMock);
  vi.spyOn(console, "error").mockImplementation(() => {});
  delete process.env.ELEVENLABS_API_KEY;
  delete process.env.DEEPGRAM_API_KEY;
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("live-session token route — auth", () => {
  it("rejects an unauthenticated caller without touching a provider", async () => {
    vi.mocked(getServerAuthToken).mockResolvedValue(null);
    process.env.ELEVENLABS_API_KEY = ELEVENLABS_KEY;

    const response = await POST(makeRequest({ provider: "elevenlabs" }));

    expect(response.status).toBe(401);
    await expect(response.json()).resolves.toEqual({ error: "Unauthorized" });
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

describe("live-session token route — elevenlabs (default provider)", () => {
  it("mints a single-use token and returns only the token", async () => {
    process.env.ELEVENLABS_API_KEY = ELEVENLABS_KEY;
    fetchMock.mockResolvedValue(jsonResponse({ token: "eleven-single-use" }));

    const response = await POST(makeRequest({}));

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({
      token: "eleven-single-use",
    });

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe(
      "https://api.elevenlabs.io/v1/single-use-token/realtime_scribe",
    );
    expect(init.method).toBe("POST");
    expect((init.headers as Record<string, string>)["xi-api-key"]).toBe(
      ELEVENLABS_KEY,
    );
    expect(init.signal).toBeInstanceOf(AbortSignal);
  });

  it("is the fallback for an unknown provider and for an unparseable body", async () => {
    process.env.ELEVENLABS_API_KEY = ELEVENLABS_KEY;
    fetchMock.mockResolvedValue(jsonResponse({ token: "eleven-single-use" }));

    await POST(makeRequest({ provider: "whisper" }));
    await POST(makeRequest());

    expect(fetchMock).toHaveBeenCalledTimes(2);
    for (const [url] of fetchMock.mock.calls) {
      expect(url).toBe(
        "https://api.elevenlabs.io/v1/single-use-token/realtime_scribe",
      );
    }
  });

  it("reports 'not configured' when no key is present, without calling out", async () => {
    const response = await POST(makeRequest({ provider: "elevenlabs" }));

    expect(response.status).toBe(503);
    await expect(response.json()).resolves.toEqual({
      error: "Live transcription is not configured",
    });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("returns 502 when the provider rejects the mint", async () => {
    process.env.ELEVENLABS_API_KEY = ELEVENLABS_KEY;
    fetchMock.mockResolvedValue(
      new Response("quota exceeded", { status: 429 }),
    );

    const response = await POST(makeRequest({}));

    expect(response.status).toBe(502);
    await expect(response.json()).resolves.toEqual({
      error: "Could not start live transcription",
    });
  });

  it("returns 502 when the provider answers 200 with no token", async () => {
    process.env.ELEVENLABS_API_KEY = ELEVENLABS_KEY;
    fetchMock.mockResolvedValue(jsonResponse({}));

    const response = await POST(makeRequest({}));

    expect(response.status).toBe(502);
    await expect(response.json()).resolves.toEqual({
      error: "Could not start live transcription",
    });
  });
});

describe("live-session token route — deepgram", () => {
  it("grants a short-lived token and returns it under `token`", async () => {
    process.env.DEEPGRAM_API_KEY = DEEPGRAM_KEY;
    fetchMock.mockResolvedValue(jsonResponse({ access_token: "dg-grant" }));

    const response = await POST(makeRequest({ provider: "deepgram" }));

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({ token: "dg-grant" });

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("https://api.deepgram.com/v1/auth/grant");
    expect((init.headers as Record<string, string>).Authorization).toBe(
      `Token ${DEEPGRAM_KEY}`,
    );
    // Only has to cover the handshake — see TOKEN_TTL_SECONDS.
    expect(JSON.parse(init.body as string)).toEqual({ ttl_seconds: 60 });
  });

  it("reports 'not configured' when no key is present, without calling out", async () => {
    process.env.ELEVENLABS_API_KEY = ELEVENLABS_KEY;

    const response = await POST(makeRequest({ provider: "deepgram" }));

    expect(response.status).toBe(503);
    await expect(response.json()).resolves.toEqual({
      error: "Live transcription is not configured",
    });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("returns 502 when the grant is refused", async () => {
    process.env.DEEPGRAM_API_KEY = DEEPGRAM_KEY;
    fetchMock.mockResolvedValue(new Response("forbidden", { status: 403 }));

    const response = await POST(makeRequest({ provider: "deepgram" }));

    expect(response.status).toBe(502);
    await expect(response.json()).resolves.toEqual({
      error: "Could not start live transcription",
    });
  });

  it("returns 502 when the grant succeeds without an access token", async () => {
    process.env.DEEPGRAM_API_KEY = DEEPGRAM_KEY;
    fetchMock.mockResolvedValue(jsonResponse({ access_token: "" }));

    const response = await POST(makeRequest({ provider: "deepgram" }));

    expect(response.status).toBe(502);
  });
});

describe("live-session token route — upstream failure", () => {
  it("turns a thrown request (timeout, DNS, abort) into a 502 the client can fall back from", async () => {
    process.env.ELEVENLABS_API_KEY = ELEVENLABS_KEY;
    fetchMock.mockRejectedValue(
      new DOMException("The operation timed out.", "TimeoutError"),
    );

    const response = await POST(makeRequest({}));

    expect(response.status).toBe(502);
    await expect(response.json()).resolves.toEqual({
      error: "Could not start live transcription",
    });
  });
});

describe("live-session token route — key containment", () => {
  const cases: {
    name: string;
    provider: string;
    env: () => void;
    upstream: () => Response | Promise<never>;
  }[] = [
    {
      name: "elevenlabs success",
      provider: "elevenlabs",
      env: () => {
        process.env.ELEVENLABS_API_KEY = ELEVENLABS_KEY;
      },
      upstream: () => jsonResponse({ token: "eleven-single-use" }),
    },
    {
      name: "elevenlabs upstream error",
      provider: "elevenlabs",
      env: () => {
        process.env.ELEVENLABS_API_KEY = ELEVENLABS_KEY;
      },
      upstream: () => new Response(ELEVENLABS_KEY, { status: 401 }),
    },
    {
      name: "deepgram success",
      provider: "deepgram",
      env: () => {
        process.env.DEEPGRAM_API_KEY = DEEPGRAM_KEY;
      },
      upstream: () => jsonResponse({ access_token: "dg-grant" }),
    },
    {
      name: "deepgram upstream error",
      provider: "deepgram",
      env: () => {
        process.env.DEEPGRAM_API_KEY = DEEPGRAM_KEY;
      },
      upstream: () => new Response(DEEPGRAM_KEY, { status: 401 }),
    },
  ];

  it.each(cases)(
    "never echoes the provider API key back to the browser ($name)",
    async ({ provider, env, upstream }) => {
      env();
      fetchMock.mockResolvedValue(upstream());

      const response = await POST(makeRequest({ provider }));
      const body = await response.text();

      expect(body).not.toContain(ELEVENLABS_KEY);
      expect(body).not.toContain(DEEPGRAM_KEY);
    },
  );
});
