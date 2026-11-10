// @vitest-environment node
import { describe, expect, it } from "vitest";
import {
  MOBILE_APP_ORIGIN_HEADER,
  getMobileAppOrigins,
  mobileAppOrigin,
} from "../mobile-app-origin";

const SIGN_OUT_URL = "http://localhost:3000/api/auth/sign-out";

async function originAfterPlugin(headers: Record<string, string>) {
  const request = new Request(SIGN_OUT_URL, { method: "POST", headers });
  const result = await mobileAppOrigin().onRequest?.(request, {} as never);
  const seen = result && "request" in result ? result.request : request;
  return seen.headers.get("origin");
}

describe("getMobileAppOrigins", () => {
  it("trusts no app origin until a scheme is set", () => {
    expect(getMobileAppOrigins({})).toEqual([]);
    expect(getMobileAppOrigins({ AUTH_MOBILE_APP_SCHEME: "  " })).toEqual([]);
  });

  it("trusts only the app's own scheme outside development", () => {
    expect(
      getMobileAppOrigins({
        AUTH_MOBILE_APP_SCHEME: "autogpt-app",
        NODE_ENV: "production",
      }),
    ).toEqual(["autogpt-app://"]);
  });

  it("also trusts Expo Go in development", () => {
    expect(
      getMobileAppOrigins({
        AUTH_MOBILE_APP_SCHEME: "autogpt-app",
        NODE_ENV: "development",
      }),
    ).toEqual(["autogpt-app://", "exp://"]);
  });

  it.each(["https", "http", "autogpt-app://", "*", "evil.example.com/x"])(
    "refuses %s as a scheme",
    (scheme) => {
      expect(() =>
        getMobileAppOrigins({ AUTH_MOBILE_APP_SCHEME: scheme }),
      ).toThrow(/AUTH_MOBILE_APP_SCHEME/);
    },
  );
});

describe("mobileAppOrigin", () => {
  it("reads the app's origin from its own header when none was sent", async () => {
    expect(
      await originAfterPlugin({ [MOBILE_APP_ORIGIN_HEADER]: "autogpt-app://" }),
    ).toBe("autogpt-app://");
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
    const result = await mobileAppOrigin().onRequest?.(request, {} as never);
    const seen = result && "request" in result ? result.request : request;
    expect(await seen.json()).toEqual({ email: "a@example.com" });
  });
});
