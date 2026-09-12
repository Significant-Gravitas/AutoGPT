import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { isLocalStoreMediaUrl } from "@/lib/store-media";
import { isRenderableImageUrl } from "@/lib/next-image";

const mediaPath = "/store/media/user-123/images/thumbnail.png";
const defaultImageURLs = [
  `http://localhost:8006/api${mediaPath}`,
  `http://localhost:8006/api${mediaPath}?version=2#preview`,
  "http://localhost:8006/api/store/media/user.name_1-2/images/preview.v2_1-thumb.webp",
  `/api/proxy/api${mediaPath}`,
  `http://localhost:3000/api/proxy/api${mediaPath}`,
];

beforeEach(() => {
  vi.stubEnv("NEXT_PUBLIC_AGPT_SERVER_URL", undefined);
  vi.stubEnv("BETTER_AUTH_URL", undefined);
  vi.stubEnv("NEXT_PUBLIC_FRONTEND_BASE_URL", undefined);
  vi.stubEnv("AGPT_SERVER_URL", "http://internal-backend:8006/api");
});

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
});

describe("configured local store media", () => {
  it.each(defaultImageURLs)(
    "recognizes the default backend or proxy: %s",
    (src) => {
      expect(isLocalStoreMediaUrl(src)).toBe(true);
      expect(isRenderableImageUrl(src)).toBe(true);
    },
  );

  it.each([
    `https://unknown.example/api${mediaPath}`,
    `http://localhost:8007/api${mediaPath}`,
    `http://localhost:8006/wrong/api${mediaPath}`,
    `http://internal-backend:8006/api${mediaPath}`,
    `/api${mediaPath}`,
    `https://unknown.example/api/proxy/api${mediaPath}`,
    `http://localhost:8006/api/proxy/api${mediaPath}`,
    `/unrelated/api/proxy/api${mediaPath}`,
    `//localhost:8006/api${mediaPath}`,
    `http://user@localhost:8006/api${mediaPath}`,
    `http://:test@localhost:8006/api${mediaPath}`,
  ])("rejects an unconfigured origin or prefix: %s", (src) => {
    expect(isLocalStoreMediaUrl(src)).toBe(false);
  });

  it.each(["browser", "SSR"])(
    "uses the public absolute API URL in %s",
    (mode) => {
      vi.stubEnv(
        "NEXT_PUBLIC_AGPT_SERVER_URL",
        "https://media.appliance.example/team/backend/api",
      );
      if (mode === "SSR") vi.stubGlobal("window", undefined);

      expect(
        isLocalStoreMediaUrl(
          `https://media.appliance.example/team/backend/api${mediaPath}`,
        ),
      ).toBe(true);
      expect(
        isLocalStoreMediaUrl(`https://media.appliance.example/api${mediaPath}`),
      ).toBe(false);
      expect(
        isLocalStoreMediaUrl(
          `https://other.example/team/backend/api${mediaPath}`,
        ),
      ).toBe(false);
      expect(isLocalStoreMediaUrl(`/team/backend/api${mediaPath}`)).toBe(false);
    },
  );

  it("resolves an appliance API prefix against the browser origin", () => {
    vi.stubEnv("NEXT_PUBLIC_AGPT_SERVER_URL", "/_agpt/api");
    vi.stubEnv("BETTER_AUTH_URL", "https://wrong.example");
    vi.stubGlobal("window", {
      location: { origin: "https://appliance.example" },
    });

    expect(isLocalStoreMediaUrl(`/_agpt/api${mediaPath}`)).toBe(true);
    expect(
      isLocalStoreMediaUrl(`https://appliance.example/_agpt/api${mediaPath}`),
    ).toBe(true);
    expect(
      isLocalStoreMediaUrl(`https://wrong.example/_agpt/api${mediaPath}`),
    ).toBe(false);
    expect(isLocalStoreMediaUrl(`/api${mediaPath}`)).toBe(false);
    expect(
      isLocalStoreMediaUrl(
        `https://appliance.example/api/proxy/api${mediaPath}`,
      ),
    ).toBe(true);
  });

  it.each(["BETTER_AUTH_URL", "NEXT_PUBLIC_FRONTEND_BASE_URL"])(
    "resolves the appliance API prefix using %s during SSR",
    (frontendEnv) => {
      vi.stubGlobal("window", undefined);
      vi.stubEnv("NEXT_PUBLIC_AGPT_SERVER_URL", "/_agpt/api");
      vi.stubEnv(frontendEnv, "https://appliance.example");
      if (frontendEnv === "BETTER_AUTH_URL") {
        vi.stubEnv("NEXT_PUBLIC_FRONTEND_BASE_URL", "https://wrong.example");
      }

      expect(isLocalStoreMediaUrl(`/_agpt/api${mediaPath}`)).toBe(true);
      expect(
        isLocalStoreMediaUrl(`https://appliance.example/_agpt/api${mediaPath}`),
      ).toBe(true);
      expect(
        isLocalStoreMediaUrl(`https://wrong.example/_agpt/api${mediaPath}`),
      ).toBe(false);
      expect(isLocalStoreMediaUrl(`/api${mediaPath}`)).toBe(false);
      expect(isLocalStoreMediaUrl(`/api/proxy/api${mediaPath}`)).toBe(true);
      expect(
        isLocalStoreMediaUrl(
          `https://appliance.example/api/proxy/api${mediaPath}`,
        ),
      ).toBe(true);
    },
  );

  it("uses the default frontend origin during SSR", () => {
    vi.stubGlobal("window", undefined);
    vi.stubEnv("NEXT_PUBLIC_AGPT_SERVER_URL", "/_agpt/api");

    expect(
      isLocalStoreMediaUrl(`http://localhost:3000/_agpt/api${mediaPath}`),
    ).toBe(true);
    expect(
      isLocalStoreMediaUrl(`http://localhost:3001/_agpt/api${mediaPath}`),
    ).toBe(false);
  });
});

describe("malformed and ordinary URLs", () => {
  beforeEach(() => vi.stubEnv("NEXT_PUBLIC_AGPT_SERVER_URL", "/api"));

  it.each([
    null,
    undefined,
    "",
    "not a URL",
    "/images/thumbnail.png",
    "https://cdn.example.com/images/thumbnail.png",
    "/api/store/media/user-123/videos/preview.mp4",
    "/api/store/media/user-123/images/thumbnail.png/extra",
    "/api/store/media/user-123/images/thumbnail.png/",
    "/api/store/media/user-123/images/",
    "/api/store/media//images/thumbnail.png",
    "/api/store/media/user 123/images/thumbnail.png",
    "/api/store/media/user%20123/images/thumbnail.png",
    "/api/store/media/user-123/images/thumbnail%20image.png",
    "/api/store/media/user-123/images/thumbnail@2x.png",
    "/api/store/media/./images/thumbnail.png",
    "/api/store/media/../images/thumbnail.png",
    "/api/store/media/user-123/images/.",
    "/api/store/media/user-123/images/..",
    "/api/store/media/user-123/images/%2e%2e",
    "/api/store/media/user-123/images/%2Fthumbnail.png",
    "/api/store/media/user-123/images/nested/../thumbnail.png",
    "http://localhost:3000/api/store/media/user-123/images/nested/../thumbnail.png",
    "http://localhost:3000/api/store/media/user-123/images/%2e%2e/images/thumbnail.png",
    "/api/store/media/user-123/images/thumbnail\\image.png",
    "api/store/media/user-123/images/thumbnail.png",
    "//localhost:3000/api/store/media/user-123/images/thumbnail.png",
    "ftp://localhost:3000/api/store/media/user-123/images/thumbnail.png",
    "file:///api/store/media/user-123/images/thumbnail.png",
    "javascript:/api/store/media/user-123/images/thumbnail.png",
    "data:image/png;base64,abc",
    "blob:http://localhost:3000/api/store/media/user-123/images/thumbnail.png",
    `https://cdn.example.com/image.png?src=/api${mediaPath}`,
  ])("rejects %s", (src) => {
    expect(isLocalStoreMediaUrl(src)).toBe(false);
  });
});

describe("local store images with Next.js", () => {
  it.each(defaultImageURLs)("loads %s directly without srcSet", async (src) => {
    const { getImageProps } =
      await vi.importActual<typeof import("next/image")>("next/image");
    const { props } = getImageProps({
      src,
      alt: "Agent thumbnail",
      width: 320,
      height: 180,
      unoptimized: isLocalStoreMediaUrl(src),
    });

    expect(props.src).toBe(src);
    expect(props.srcSet).toBeUndefined();
  });

  it.each([
    "https://storage.googleapis.com/agent-uploads/user-123/images/thumbnail.png",
    `https://storage.googleapis.com/api${mediaPath}`,
    "/images/thumbnail.png",
  ])("keeps ordinary and GCS images optimized: %s", async (src) => {
    const { getImageProps } =
      await vi.importActual<typeof import("next/image")>("next/image");
    const { props } = getImageProps({
      src,
      alt: "Agent thumbnail",
      width: 320,
      height: 180,
      unoptimized: isLocalStoreMediaUrl(src),
    });

    expect(props.src).toMatch(/^\/_next\/image\?/);
    expect(new URL(props.src, "http://localhost").searchParams.get("url")).toBe(
      src,
    );
    expect(props.srcSet).toContain("/_next/image?");
  });

  it("keeps unconfigured URLs outside the render gate", () => {
    expect(isRenderableImageUrl("https://unknown.example/thumbnail.png")).toBe(
      false,
    );
    expect(
      isRenderableImageUrl(`https://unknown.example/api${mediaPath}`),
    ).toBe(false);
  });
});
