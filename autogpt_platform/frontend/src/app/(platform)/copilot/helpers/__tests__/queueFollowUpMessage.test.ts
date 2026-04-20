import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { queueFollowUpMessage } from "../queueFollowUpMessage";

vi.mock("@/services/environment", () => ({
  environment: {
    getAGPTServerBaseUrl: () => "https://api.example.test",
  },
}));

vi.mock("../../helpers", () => ({
  getCopilotAuthHeaders: async () => ({ Authorization: "Bearer test-token" }),
}));

describe("queueFollowUpMessage", () => {
  const originalFetch = global.fetch;

  beforeEach(() => {
    global.fetch = vi.fn();
  });

  afterEach(() => {
    global.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it("POSTs to /stream and returns kind=queued on 202", async () => {
    const fetchMock = vi.mocked(global.fetch);
    fetchMock.mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          buffer_length: 2,
          max_buffer_length: 10,
          turn_in_flight: true,
        }),
        { status: 202, headers: { "Content-Type": "application/json" } },
      ),
    );

    const result = await queueFollowUpMessage("sess-1", "hello");

    expect(result).toEqual({
      kind: "queued",
      buffer_length: 2,
      max_buffer_length: 10,
      turn_in_flight: true,
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(
      "https://api.example.test/api/chat/sessions/sess-1/stream",
    );
    expect(init?.method).toBe("POST");
    const headers = init?.headers as Record<string, string>;
    expect(headers["Content-Type"]).toBe("application/json");
    expect(headers.Authorization).toBe("Bearer test-token");
    expect(JSON.parse(init?.body as string)).toEqual({
      message: "hello",
      is_user_message: true,
      context: null,
      file_ids: null,
    });
  });

  it("returns kind=raced_started_turn on 200 (race: server started a new turn)", async () => {
    const fetchMock = vi.mocked(global.fetch);
    // SSE-shaped 200 response — the body must be drainable
    const sseBody = new ReadableStream({
      start(controller) {
        controller.enqueue(new TextEncoder().encode("data: hello\n\n"));
        controller.close();
      },
    });
    fetchMock.mockResolvedValueOnce(
      new Response(sseBody, {
        status: 200,
        headers: { "Content-Type": "text/event-stream" },
      }),
    );

    const result = await queueFollowUpMessage("sess-1", "hi");

    expect(result).toEqual({ kind: "raced_started_turn", status: 200 });
  });

  it("throws when response is neither 200 nor 202", async () => {
    const fetchMock = vi.mocked(global.fetch);
    fetchMock.mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: "boom" }), { status: 500 }),
    );

    await expect(queueFollowUpMessage("sess-1", "hi")).rejects.toThrow(
      /Expected 202 \(queued\) or 200 \(raced new turn\) from \/stream; got 500/,
    );
  });

  it("throws on 429 rate-limit response", async () => {
    const fetchMock = vi.mocked(global.fetch);
    fetchMock.mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: "too many" }), { status: 429 }),
    );

    await expect(queueFollowUpMessage("sess-1", "hi")).rejects.toThrow(
      /got 429/,
    );
  });
});
