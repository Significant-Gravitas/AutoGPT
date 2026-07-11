import { NextRequest } from "next/server";
import { afterEach, describe, expect, it, vi } from "vitest";

import { GET } from "../route";

function request(url: string) {
  return new NextRequest(url);
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("GET /api/changelog", () => {
  it("proxies the index README from the gitbook branch", async () => {
    const fetchSpy = vi
      .spyOn(global, "fetch")
      .mockResolvedValue(new Response("| index |", { status: 200 }));

    const res = await GET(request("http://localhost/api/changelog"));

    expect(res.status).toBe(200);
    expect(await res.text()).toContain("| index |");
    expect(String(fetchSpy.mock.calls[0][0])).toContain(
      "/gitbook/docs/platform/changelog/README.md",
    );
  });

  it("proxies a single release by slug", async () => {
    const fetchSpy = vi
      .spyOn(global, "fetch")
      .mockResolvedValue(new Response("# Release", { status: 200 }));

    const res = await GET(
      request("http://localhost/api/changelog?slug=may-7-june-10-2026"),
    );

    expect(res.status).toBe(200);
    expect(String(fetchSpy.mock.calls[0][0])).toContain(
      "/changelog/may-7-june-10-2026.md",
    );
  });

  it("rejects an invalid slug without hitting GitHub", async () => {
    const fetchSpy = vi.spyOn(global, "fetch");

    const res = await GET(
      request("http://localhost/api/changelog?slug=../../etc/passwd"),
    );

    expect(res.status).toBe(400);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("returns 502 when the source is unavailable", async () => {
    vi.spyOn(global, "fetch").mockResolvedValue(
      new Response("nope", { status: 404 }),
    );

    const res = await GET(request("http://localhost/api/changelog"));

    expect(res.status).toBe(502);
  });
});
