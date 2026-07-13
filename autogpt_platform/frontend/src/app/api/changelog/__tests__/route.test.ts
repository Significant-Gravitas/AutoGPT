import { afterEach, describe, expect, it, vi } from "vitest";

import { GET } from "../route";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("GET /api/changelog", () => {
  it("proxies the changelog index README from the gitbook branch", async () => {
    const fetchSpy = vi
      .spyOn(global, "fetch")
      .mockResolvedValue(new Response("| index |", { status: 200 }));

    const res = await GET();

    expect(res.status).toBe(200);
    expect(await res.text()).toContain("| index |");
    expect(String(fetchSpy.mock.calls[0][0])).toContain(
      "/gitbook/docs/platform/changelog/README.md",
    );
  });

  it("returns 502 when the source is unavailable", async () => {
    vi.spyOn(global, "fetch").mockResolvedValue(
      new Response("nope", { status: 404 }),
    );

    expect((await GET()).status).toBe(502);
  });

  it("returns 502 when the fetch throws", async () => {
    vi.spyOn(global, "fetch").mockRejectedValue(new Error("network"));

    expect((await GET()).status).toBe(502);
  });
});
