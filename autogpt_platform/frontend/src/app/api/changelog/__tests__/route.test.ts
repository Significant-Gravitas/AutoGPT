import { afterEach, describe, expect, it, vi } from "vitest";

import { GET } from "../route";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("GET /api/changelog", () => {
  it("proxies the changelog index markdown server-side", async () => {
    const fetchSpy = vi
      .spyOn(global, "fetch")
      .mockResolvedValue(new Response("| index |", { status: 200 }));

    const res = await GET();

    expect(res.status).toBe(200);
    expect(await res.text()).toContain("| index |");
    expect(fetchSpy.mock.calls[0][0]).toBe(
      "https://agpt.co/docs/platform/changelog/changelog.md",
    );
  });

  it("returns 502 when the docs site is unavailable", async () => {
    vi.spyOn(global, "fetch").mockResolvedValue(
      new Response("nope", { status: 503 }),
    );

    const res = await GET();

    expect(res.status).toBe(502);
  });

  it("returns 502 when the fetch throws", async () => {
    vi.spyOn(global, "fetch").mockRejectedValue(new Error("network"));

    const res = await GET();

    expect(res.status).toBe(502);
  });
});
