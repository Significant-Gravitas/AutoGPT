import { NextRequest } from "next/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { POST } from "../route";

function post(path: string, headers: Record<string, string> = {}) {
  return POST(
    new NextRequest(`https://app.test/api/ld-events/${path}`, {
      method: "POST",
      headers: { "content-type": "application/json", ...headers },
      body: '[{"kind":"summary"}]',
    }),
    { params: Promise.resolve({ path: path.split("/") }) },
  );
}

const CLIENT_ID = "6735da35404f1f08a005f897";

describe("LaunchDarkly events proxy", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response(null, { status: 202 })),
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("forwards a bulk flush to LaunchDarkly and answers with its status", async () => {
    const response = await post(`events/bulk/${CLIENT_ID}`);

    expect(response.status).toBe(202);
    const [url, init] = vi.mocked(fetch).mock.calls[0];
    expect(url).toBe(
      `https://events.launchdarkly.com/events/bulk/${CLIENT_ID}`,
    );
    expect(init?.method).toBe("POST");
    expect(init?.body).toBe('[{"kind":"summary"}]');
  });

  it("forwards the event-schema and payload-id headers the event API reads", async () => {
    await post(`events/bulk/${CLIENT_ID}`, {
      "x-launchdarkly-event-schema": "4",
      "x-launchdarkly-payload-id": "payload-1",
    });

    expect(vi.mocked(fetch).mock.calls[0][1]?.headers).toEqual({
      "Content-Type": "application/json",
      "x-launchdarkly-event-schema": "4",
      "x-launchdarkly-payload-id": "payload-1",
    });
  });

  it("builds the upstream headers from the allowlist, so nothing the browser attached rides along", async () => {
    await post(`events/bulk/${CLIENT_ID}`, {
      authorization: "Bearer session-secret",
      "x-impersonate-user": "session-secret",
    });

    const sent = vi.mocked(fetch).mock.calls[0][1]?.headers as Record<
      string,
      string
    >;
    expect(Object.keys(sent)).toEqual(["Content-Type"]);
    expect(JSON.stringify(sent)).not.toContain("session-secret");
  });

  it("refuses a path the SDK never posts, rather than proxying it", async () => {
    const response = await post("api/v2/flags");

    expect(response.status).toBe(404);
    expect(fetch).not.toHaveBeenCalled();
  });

  it("swallows an upstream failure: a dropped analytics batch is not the user's problem", async () => {
    vi.mocked(fetch).mockRejectedValue(new Error("upstream down"));

    expect((await post(`events/bulk/${CLIENT_ID}`)).status).toBe(202);
  });
});
