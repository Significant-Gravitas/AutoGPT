import { afterEach, describe, expect, it, vi } from "vitest";

import { updateEmbedSessionTitle } from "./session-api";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("updateEmbedSessionTitle", () => {
  it("uses a fresh token and sends the first message to the restricted route", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        new Response(
          JSON.stringify({ title: "Compare the active shipment lanes and..." }),
          { status: 200, headers: { "content-type": "application/json" } },
        ),
      );
    vi.stubGlobal("fetch", fetchMock);
    const getAccessToken = vi.fn().mockResolvedValue("embed-token");

    const title = await updateEmbedSessionTitle(
      "",
      "session/1",
      "Compare the active shipment lanes and flag the highest risk",
      getAccessToken,
    );

    expect(title).toBe("Compare the active shipment lanes and...");
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/embed/v1/sessions/session%2F1/title",
      {
        method: "PATCH",
        headers: {
          Authorization: "Bearer embed-token",
          "content-type": "application/json",
        },
        body: JSON.stringify({
          message:
            "Compare the active shipment lanes and flag the highest risk",
        }),
      },
    );
  });
});
