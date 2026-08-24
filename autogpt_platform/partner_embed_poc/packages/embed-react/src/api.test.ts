import { afterEach, describe, expect, it, vi } from "vitest";

import { createEmbedSession } from "./api";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("createEmbedSession", () => {
  it("uses a fresh host-provided token and the restricted embed endpoint", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          id: "session-1",
          created_at: "2026-08-24T12:00:00+00:00",
        }),
        { status: 201, headers: { "content-type": "application/json" } },
      ),
    );
    vi.stubGlobal("fetch", fetchMock);
    const getAccessToken = vi.fn().mockResolvedValue("embed-token");

    const session = await createEmbedSession(
      "http://localhost:8006/",
      getAccessToken,
    );

    expect(session).toEqual({
      id: "session-1",
      createdAt: "2026-08-24T12:00:00+00:00",
    });
    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:8006/api/embed/v1/sessions",
      {
        method: "POST",
        headers: { Authorization: "Bearer embed-token" },
      },
    );
  });

  it("does not hide a rejected embed token", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response("Unauthorized", { status: 401 })),
    );

    await expect(
      createEmbedSession("http://localhost:8006", async () => "expired"),
    ).rejects.toThrow("Unable to create embedded chat (401)");
  });
});
