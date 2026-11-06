// @vitest-environment node
import { GET as appleAssociation } from "@/app/.well-known/apple-app-site-association/route";
import { GET as androidAssetLinks } from "@/app/.well-known/assetlinks.json/route";
import { GET as callbackPage } from "@/app/(platform)/auth/mobile/callback/route";
import { afterEach, describe, expect, it, vi } from "vitest";

const RELEASE = Array.from({ length: 32 }, () => "AB").join(":");
const DEBUG = Array.from({ length: 32 }, () => "0f").join(":");

afterEach(() => {
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

describe("Apple app site association", () => {
  it("is a 404 until the team is configured, so sign-in fails closed", () => {
    vi.stubEnv("MOBILE_APPLE_TEAM_ID", "");
    expect(appleAssociation().status).toBe(404);
  });

  it("names only our app, for only the sign-in callback path", async () => {
    vi.stubEnv("MOBILE_APPLE_TEAM_ID", "TEAM123456");
    const response = appleAssociation();
    expect(response.headers.get("content-type")).toContain("application/json");
    expect(await response.json()).toEqual({
      applinks: {
        details: [
          {
            appIDs: ["TEAM123456.co.agpt.autogpt"],
            components: [{ "/": "/auth/mobile/callback" }],
          },
        ],
      },
      webcredentials: { apps: ["TEAM123456.co.agpt.autogpt"] },
    });
  });

  it.each(["*", "team123456", "TEAM123456.co.evil", "TEAM123456*"])(
    "refuses the team value %s",
    (value) => {
      vi.spyOn(console, "error").mockImplementation(() => {});
      vi.stubEnv("MOBILE_APPLE_TEAM_ID", value);
      expect(appleAssociation().status).toBe(404);
    },
  );
});

describe("Android asset links", () => {
  it("is a 404 until a fingerprint is configured", () => {
    vi.stubEnv("MOBILE_ANDROID_CERT_SHA256", "");
    vi.stubEnv("MOBILE_ANDROID_DEBUG_CERT_SHA256", "");
    expect(androidAssetLinks().status).toBe(404);
  });

  it("lists the release and debug packages with their own keys", async () => {
    vi.stubEnv("MOBILE_ANDROID_CERT_SHA256", RELEASE);
    vi.stubEnv("MOBILE_ANDROID_DEBUG_CERT_SHA256", DEBUG);
    const statements = await androidAssetLinks().json();
    expect(
      statements.map(
        (statement: {
          target: { package_name: string; sha256_cert_fingerprints: string[] };
        }) => [
          statement.target.package_name,
          statement.target.sha256_cert_fingerprints,
        ],
      ),
    ).toEqual([
      ["co.agpt.autogpt", [RELEASE]],
      ["co.agpt.autogpt.dev", [DEBUG.toUpperCase()]],
    ]);
  });

  it.each(["*", `${RELEASE},*`, RELEASE.slice(3), `${RELEASE}:AB`])(
    "refuses the fingerprint list %s",
    (value) => {
      vi.spyOn(console, "error").mockImplementation(() => {});
      vi.stubEnv("MOBILE_ANDROID_CERT_SHA256", value);
      vi.stubEnv("MOBILE_ANDROID_DEBUG_CERT_SHA256", "");
      expect(androidAssetLinks().status).toBe(404);
    },
  );
});

describe("sign-in callback page", () => {
  it("runs no script and is never cached or sent as a referrer", async () => {
    const response = callbackPage();
    const page = await response.text();
    expect(response.headers.get("cache-control")).toBe("no-store");
    expect(page).not.toContain("<script");
    expect(page).toContain('<meta name="referrer" content="no-referrer">');
    expect(page).toContain("default-src 'none'");
  });
});
