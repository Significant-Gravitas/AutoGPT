import { join } from "node:path";
import type { NextConfig } from "next";
import { NextRequest } from "next/server";
import {
  getRewrittenUrl,
  unstable_getResponseFromNextConfig,
} from "next/experimental/testing/server";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { isLocalStoreMediaUrl } from "@/lib/store-media";

vi.mock("@sentry/nextjs", () => ({
  withSentryConfig: (config: NextConfig) => config,
}));
vi.mock("@/lib/auth/server/getServerAuthToken", () => ({
  getServerAuthToken: vi.fn().mockResolvedValue(null),
}));
vi.mock("@/services/environment", () => ({
  environment: {
    getAGPTServerBaseUrl: () => "http://internal-backend:8006",
  },
}));

import { GET } from "@/app/api/proxy/[...path]/route";

beforeEach(() => {
  vi.stubEnv("NEXT_PUBLIC_AGPT_SERVER_URL", "http://localhost:8006/api");
  vi.stubGlobal("fetch", vi.fn());
});
afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

it.each([
  ["images/thumbnail.png", "image/png"],
  ["videos/preview.webm", "video/webm"],
])(
  "serves backend-relative %s through the anonymous frontend proxy",
  async (file, mime) => {
    const { default: nextConfig } = await vi.importActual<{
      default: NextConfig;
    }>(join(process.cwd(), "next.config.mjs"));
    const relativeURL = `/api/store/media/user-123/${file}`;
    const browserURL = `http://localhost:3000${relativeURL}?v=2`;
    const routing = await unstable_getResponseFromNextConfig({
      url: browserURL,
      nextConfig,
    });
    const proxyURL = getRewrittenUrl(routing);
    expect(proxyURL).toBe(`http://localhost:3000/api/proxy${relativeURL}?v=2`);

    const bytes = Uint8Array.from(
      { length: 256 * 1024 },
      (_, index) => index % 251,
    );
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        for (let index = 0; index < bytes.length; index += 8191) {
          controller.enqueue(bytes.slice(index, index + 8191));
        }
        controller.close();
      },
    });
    vi.mocked(fetch).mockResolvedValue(
      new Response(body, {
        headers: { "content-type": mime, "x-content-type-options": "nosniff" },
      }),
    );
    const response = await GET(new NextRequest(proxyURL!), {
      params: Promise.resolve({
        path: ["api", "store", "media", "user-123", ...file.split("/")],
      }),
    });
    expect(fetch).toHaveBeenCalledWith(
      `http://internal-backend:8006${relativeURL}?v=2`,
      expect.anything(),
    );
    expect(
      (vi.mocked(fetch).mock.calls[0][1]?.headers as Headers).has(
        "authorization",
      ),
    ).toBe(false);
    expect(response.headers.get("content-type")).toBe(mime);
    expect(response.headers.get("x-content-type-options")).toBe("nosniff");
    expect(new Uint8Array(await response.arrayBuffer())).toEqual(bytes);

    if (mime === "image/png") {
      const { getImageProps } =
        await vi.importActual<typeof import("next/image")>("next/image");
      expect(isLocalStoreMediaUrl(relativeURL)).toBe(true);
      const { props } = getImageProps({
        src: relativeURL,
        alt: "Thumbnail",
        width: 320,
        height: 180,
        unoptimized: isLocalStoreMediaUrl(relativeURL),
      });
      expect(props.src).toBe(relativeURL);
      expect(props.srcSet).toBeUndefined();
    }
  },
);

it("does not rewrite unrelated frontend API routes", async () => {
  const { default: nextConfig } = await vi.importActual<{
    default: NextConfig;
  }>(join(process.cwd(), "next.config.mjs"));
  const response = await unstable_getResponseFromNextConfig({
    url: "http://localhost:3000/api/store/submissions",
    nextConfig,
  });
  expect(getRewrittenUrl(response)).toBeNull();
});
