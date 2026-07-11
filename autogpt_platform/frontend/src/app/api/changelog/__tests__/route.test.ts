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
  it("proxies the changelog index markdown", async () => {
    const fetchSpy = vi
      .spyOn(global, "fetch")
      .mockResolvedValue(new Response("| index |", { status: 200 }));

    const res = await GET(request("http://localhost/api/changelog"));

    expect(res.status).toBe(200);
    expect(await res.text()).toContain("| index |");
    expect(fetchSpy.mock.calls[0][0]).toBe(
      "https://agpt.co/docs/platform/changelog/changelog.md",
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
    expect(fetchSpy.mock.calls[0][0]).toBe(
      "https://agpt.co/docs/platform/changelog/changelog/may-7-june-10-2026.md",
    );
  });

  it("rejects an invalid slug without hitting the docs site", async () => {
    const fetchSpy = vi.spyOn(global, "fetch");

    const res = await GET(
      request("http://localhost/api/changelog?slug=../../etc/passwd"),
    );

    expect(res.status).toBe(400);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("returns 502 when the docs site is unavailable", async () => {
    vi.spyOn(global, "fetch").mockResolvedValue(
      new Response("nope", { status: 503 }),
    );

    const res = await GET(request("http://localhost/api/changelog"));

    expect(res.status).toBe(502);
  });
});
