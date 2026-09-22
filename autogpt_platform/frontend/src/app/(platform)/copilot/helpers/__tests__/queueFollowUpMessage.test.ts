import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  QueueFollowUpNotActiveError,
  queueFollowUpMessage,
} from "../queueFollowUpMessage";

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

  it("POSTs to /messages/pending and returns queued state on 200", async () => {
    const fetchMock = vi.mocked(global.fetch);
    fetchMock.mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          buffer_length: 2,
          max_buffer_length: 10,
          turn_in_flight: true,
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    );

    const result = await queueFollowUpMessage("sess-1", "hello");

    expect(result).toEqual({
      buffer_length: 2,
      max_buffer_length: 10,
      turn_in_flight: true,
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(
      "https://api.example.test/api/chat/sessions/sess-1/messages/pending",
    );
    expect(init?.method).toBe("POST");
    const headers = init?.headers as Record<string, string>;
    expect(headers["Content-Type"]).toBe("application/json");
    expect(headers.Authorization).toBe("Bearer test-token");
    expect(JSON.parse(init?.body as string)).toEqual({
      message: "hello",
      context: null,
      file_ids: null,
    });
  });

  it("treats a 200 inactive response as not active", async () => {
    const fetchMock = vi.mocked(global.fetch);
    fetchMock.mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          buffer_length: 0,
          max_buffer_length: 10,
          turn_in_flight: false,
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    );

    await expect(queueFollowUpMessage("sess-1", "hi")).rejects.toBeInstanceOf(
      QueueFollowUpNotActiveError,
    );
  });

  it("throws a typed error when there is no active turn to queue against", async () => {
    const fetchMock = vi.mocked(global.fetch);
    fetchMock.mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: "no active turn" }), {
        status: 409,
      }),
    );

    await expect(queueFollowUpMessage("sess-1", "hi")).rejects.toBeInstanceOf(
      QueueFollowUpNotActiveError,
    );
  });

  it("throws when response is not 200", async () => {
    const fetchMock = vi.mocked(global.fetch);
    fetchMock.mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: "boom" }), { status: 500 }),
    );

    await expect(queueFollowUpMessage("sess-1", "hi")).rejects.toThrow(
      /Expected 200 from \/messages\/pending; got 500/,
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
