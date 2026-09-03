import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../../helpers", () => ({
  getCopilotAuthHeaders: async () => ({ Authorization: "Bearer test" }),
}));

vi.mock("@/services/environment", () => ({
  environment: { getAGPTServerBaseUrl: () => "http://backend" },
}));

import { synthesizeSpeech, transcribeUtterance } from "../speechApi";

function audioResponse() {
  return new Response(new Blob(["mp3"]), { status: 200 });
}

describe("synthesizeSpeech", () => {
  beforeEach(() => vi.stubGlobal("fetch", vi.fn(audioResponse)));
  afterEach(() => vi.unstubAllGlobals());

  it("posts the text and session to the backend, authenticated", async () => {
    await synthesizeSpeech("A first phrase.", "session-1");

    const [url, init] = vi.mocked(fetch).mock.calls[0];
    expect(url).toBe("http://backend/api/chat/speech");
    expect(JSON.parse(init!.body as string)).toEqual({
      text: "A first phrase.",
      session_id: "session-1",
    });
    expect((init!.headers as Record<string, string>).Authorization).toBe(
      "Bearer test",
    );
  });

  it("serves a repeated phrase from cache — the acknowledgement bank repeats", async () => {
    await synthesizeSpeech("One moment.", null);
    await synthesizeSpeech("One moment.", null);

    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it("surfaces the status code when synthesis is refused", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(new Response("", { status: 404 }));

    await expect(synthesizeSpeech("Flag is off.", null)).rejects.toThrow("404");
  });
});

describe("transcribeUtterance", () => {
  beforeEach(() => vi.stubGlobal("fetch", vi.fn()));
  afterEach(() => vi.unstubAllGlobals());

  it("uploads the clip to the existing transcribe route", async () => {
    vi.mocked(fetch).mockResolvedValue(Response.json({ text: "run it" }));

    await expect(transcribeUtterance(new Blob(["wav"]))).resolves.toBe(
      "run it",
    );
    expect(vi.mocked(fetch).mock.calls[0][0]).toBe("/api/transcribe");
  });

  it("returns an empty transcript when the route sends no text", async () => {
    vi.mocked(fetch).mockResolvedValue(Response.json({}));

    await expect(transcribeUtterance(new Blob(["wav"]))).resolves.toBe("");
  });

  it("raises the route's own error message", async () => {
    vi.mocked(fetch).mockResolvedValue(
      Response.json(
        { error: "OpenAI API key not configured" },
        { status: 401 },
      ),
    );

    await expect(transcribeUtterance(new Blob(["wav"]))).rejects.toThrow(
      "OpenAI API key not configured",
    );
  });
});
