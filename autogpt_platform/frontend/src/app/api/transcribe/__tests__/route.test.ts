import { beforeEach, describe, expect, it, vi } from "vitest";
import type { NextRequest } from "next/server";

vi.mock("@/lib/autogpt-server-api/helpers", () => ({
  getServerAuthToken: vi.fn(),
}));

import { getServerAuthToken } from "@/lib/autogpt-server-api/helpers";
import { POST } from "../route";

function makeAudioRequest(type = "audio/webm"): NextRequest {
  const formData = new FormData();
  formData.append("audio", new Blob(["audio-bytes"], { type }));
  return new Request("http://localhost/api/transcribe", {
    method: "POST",
    body: formData,
  }) as NextRequest;
}

describe("transcribe route", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.unstubAllGlobals();
    vi.mocked(getServerAuthToken).mockResolvedValue("session-token");
    delete process.env.OPENAI_API_KEY;
    delete process.env.OPENAI_API_BASE_URL;
    delete process.env.TRANSCRIPTION_API_KEY;
    delete process.env.TRANSCRIPTION_API_BASE_URL;
    delete process.env.TRANSCRIPTION_MODEL;
  });

  it("sends audio to a configured OpenAI-compatible transcription endpoint", async () => {
    process.env.TRANSCRIPTION_API_BASE_URL = "http://funasr.local:8000/v1/";
    process.env.TRANSCRIPTION_MODEL = "iic/SenseVoiceSmall";
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ text: "hello from funasr" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const response = await POST(makeAudioRequest());

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({
      text: "hello from funasr",
    });
    expect(fetchMock).toHaveBeenCalledWith(
      "http://funasr.local:8000/v1/audio/transcriptions",
      expect.objectContaining({
        method: "POST",
      }),
    );
    const fetchInit = fetchMock.mock.calls[0]?.[1] as RequestInit;
    expect(fetchInit.headers).toBeInstanceOf(Headers);
    expect((fetchInit.headers as Headers).has("Authorization")).toBe(false);
    const upstreamBody = fetchInit.body as FormData;
    expect(upstreamBody.get("model")).toBe("iic/SenseVoiceSmall");
    const file = upstreamBody.get("file") as File;
    expect(file.name).toBe("recording.webm");
  });

  it("uses a transcription API key when one is configured separately from the OpenAI key", async () => {
    process.env.TRANSCRIPTION_API_BASE_URL = "http://funasr.local:8000/v1";
    process.env.TRANSCRIPTION_API_KEY = "self-hosted-token";
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ text: "ok" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await POST(makeAudioRequest());

    const fetchInit = fetchMock.mock.calls[0]?.[1] as RequestInit;
    expect(fetchInit.headers).toBeInstanceOf(Headers);
    expect((fetchInit.headers as Headers).get("Authorization")).toBe(
      "Bearer self-hosted-token",
    );
  });

  it("uses the OpenAI API key for the default transcription endpoint", async () => {
    process.env.OPENAI_API_KEY = "openai-token";
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ text: "ok" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await POST(makeAudioRequest());

    expect(fetchMock).toHaveBeenCalledWith(
      "https://api.openai.com/v1/audio/transcriptions",
      expect.objectContaining({
        headers: expect.any(Headers),
      }),
    );
    const fetchInit = fetchMock.mock.calls[0]?.[1] as RequestInit;
    expect((fetchInit.headers as Headers).get("Authorization")).toBe(
      "Bearer openai-token",
    );
  });

  it("does not send the OpenAI API key to a custom transcription endpoint", async () => {
    process.env.TRANSCRIPTION_API_BASE_URL = "http://funasr.local:8000/v1";
    process.env.OPENAI_API_KEY = "openai-token-for-other-features";
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ text: "ok" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await POST(makeAudioRequest());

    const fetchInit = fetchMock.mock.calls[0]?.[1] as RequestInit;
    expect(fetchInit.headers).toBeInstanceOf(Headers);
    expect((fetchInit.headers as Headers).has("Authorization")).toBe(false);
  });

  it("returns upstream transcription API error messages", async () => {
    process.env.OPENAI_API_KEY = "openai-token";
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({ error: { message: "model is unavailable" } }),
        {
          status: 503,
          headers: { "content-type": "application/json" },
        },
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    const response = await POST(makeAudioRequest());

    expect(response.status).toBe(503);
    await expect(response.json()).resolves.toEqual({
      error: "model is unavailable",
    });
  });

  it("keeps requiring an API key for the default OpenAI transcription endpoint", async () => {
    const response = await POST(makeAudioRequest());

    expect(response.status).toBe(401);
    await expect(response.json()).resolves.toEqual({
      error: "OpenAI API key not configured",
    });
  });
});
