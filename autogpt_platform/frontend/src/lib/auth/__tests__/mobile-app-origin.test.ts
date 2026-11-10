// @vitest-environment node
import { describe, expect, it } from "vitest";
import {
  MOBILE_APP_ORIGIN_HEADER,
  getMobileAppConfig,
  mobileAppOrigin,
} from "../mobile-app-origin";

const BASE_URL = "https://platform.example.com/some/path";
const SITE_ORIGIN = "https://platform.example.com";
const SIGN_OUT_URL = "https://platform.example.com/api/auth/sign-out";

const PRODUCTION = { scheme: "autogpt-app", allowExpoGo: false };
const DEVELOPMENT = { scheme: "autogpt-app", allowExpoGo: true };

async function runPlugin(
  request: Request,
  config: { scheme: string; allowExpoGo: boolean } = PRODUCTION,
) {
  const result = await mobileAppOrigin(config, BASE_URL).onRequest?.(
    request,
    {} as never,
  );
  return result && "request" in result ? result.request : request;
}

async function originAfterPlugin(
  headers: Record<string, string>,
  config?: { scheme: string; allowExpoGo: boolean },
) {
  const request = new Request(SIGN_OUT_URL, { method: "POST", headers });
  return (await runPlugin(request, config)).headers.get("origin");
}

describe("getMobileAppConfig", () => {
  it("is off until a scheme is set", () => {
    expect(getMobileAppConfig({})).toBeNull();
    expect(getMobileAppConfig({ AUTH_MOBILE_APP_SCHEME: "  " })).toBeNull();
  });

  it("accepts only the app's own scheme outside development", () => {
    expect(
      getMobileAppConfig({
        AUTH_MOBILE_APP_SCHEME: "autogpt-app",
        NODE_ENV: "production",
      }),
    ).toEqual(PRODUCTION);
  });

  it("also accepts Expo Go in development", () => {
    expect(
      getMobileAppConfig({
        AUTH_MOBILE_APP_SCHEME: "autogpt-app",
        NODE_ENV: "development",
      }),
    ).toEqual(DEVELOPMENT);
  });

  it.each(["https", "http", "autogpt-app://", "*", "evil.example.com/x"])(
    "refuses %s as a scheme",
    (scheme) => {
      expect(() =>
        getMobileAppConfig({ AUTH_MOBILE_APP_SCHEME: scheme }),
      ).toThrow(/AUTH_MOBILE_APP_SCHEME/);
    },
  );
});

describe("mobileAppOrigin", () => {
  it("gives the app's request this site's own origin", async () => {
    expect(
      await originAfterPlugin({ [MOBILE_APP_ORIGIN_HEADER]: "autogpt-app://" }),
    ).toBe(SITE_ORIGIN);
  });

  it.each([
    "evil://",
    "autogpt-app://x",
    "autogpt-app:",
    "exp://10.0.0.2:8081",
  ])("leaves a request naming %s alone", async (appOrigin) => {
    expect(
      await originAfterPlugin({ [MOBILE_APP_ORIGIN_HEADER]: appOrigin }),
    ).toBeNull();
  });

  it("accepts Expo Go only in development", async () => {
    expect(
      await originAfterPlugin(
        { [MOBILE_APP_ORIGIN_HEADER]: "exp://10.0.0.2:8081" },
        DEVELOPMENT,
      ),
    ).toBe(SITE_ORIGIN);
  });

  it("never replaces an origin the client sent", async () => {
    expect(
      await originAfterPlugin({
        origin: "https://evil.example.com",
        [MOBILE_APP_ORIGIN_HEADER]: "autogpt-app://",
      }),
    ).toBe("https://evil.example.com");
  });

  it("leaves a request without either header alone", async () => {
    expect(await originAfterPlugin({})).toBeNull();
  });

  it("keeps the request body", async () => {
    const request = new Request(SIGN_OUT_URL, {
      method: "POST",
      headers: { [MOBILE_APP_ORIGIN_HEADER]: "autogpt-app://" },
      body: JSON.stringify({ email: "a@example.com" }),
    });
    const seen = await runPlugin(request);
    expect(seen.headers.get("origin")).toBe(SITE_ORIGIN);
    expect(await seen.json()).toEqual({ email: "a@example.com" });
  });
});
