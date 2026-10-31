import { afterEach, describe, expect, it, vi } from "vitest";

import { GET } from "../route";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("GET /api/changelog", () => {
  it("proxies the published site's changelog index", async () => {
    const fetchSpy = vi
      .spyOn(global, "fetch")
      .mockResolvedValue(new Response("| index |", { status: 200 }));

    const res = await GET();

    expect(res.status).toBe(200);
    expect(await res.text()).toContain("| index |");
    expect(fetchSpy).toHaveBeenCalledWith(
      "https://agpt.co/docs/platform/changelog/changelog.md",
      expect.objectContaining({ next: { revalidate: 3600 } }),
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
