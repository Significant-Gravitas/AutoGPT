import { NextRequest } from "next/server";
import { afterEach, describe, expect, it, vi } from "vitest";

import { GET } from "../route";

function request(url: string) {
  return new NextRequest(url);
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("GET /api/changelog/image", () => {
  it("proxies a valid image from the gitbook assets, preserving content-type", async () => {
    const fetchSpy = vi.spyOn(global, "fetch").mockResolvedValue(
      new Response(new Uint8Array([1, 2, 3]), {
        status: 200,
        headers: { "content-type": "image/png" },
      }),
    );

    const res = await GET(
      request("http://localhost/api/changelog/image?file=hero.png"),
    );

    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toBe("image/png");
    expect(String(fetchSpy.mock.calls[0][0])).toContain(
      "/.gitbook/assets/hero.png",
    );
  });

  it("rejects a path-traversal / non-image filename without hitting GitHub", async () => {
    const fetchSpy = vi.spyOn(global, "fetch");

    const res = await GET(
      request("http://localhost/api/changelog/image?file=../secret.txt"),
    );

    expect(res.status).toBe(400);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("returns 502 when the image is missing upstream", async () => {
    vi.spyOn(global, "fetch").mockResolvedValue(
      new Response("x", { status: 404 }),
    );

    const res = await GET(
      request("http://localhost/api/changelog/image?file=missing.png"),
    );

    expect(res.status).toBe(502);
  });
});
