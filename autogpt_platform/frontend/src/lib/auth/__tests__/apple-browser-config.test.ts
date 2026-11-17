// @vitest-environment node
import { randomUUID } from "node:crypto";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { exportPKCS8, generateKeyPair } from "jose";
import {
  authRequest,
  bundleID,
  getAppleTestState,
  loadAppleAuth,
  origin,
  signInApple,
} from "./apple-auth-fixture";
import {
  appleTokenExchange,
  beginAppleBrowserLogin,
  finishAppleBrowserLogin,
  prepareAppleWebAuth,
} from "./apple-oauth-fixture";

beforeEach(prepareAppleWebAuth);
afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

describe("Apple browser configuration", () => {
  it.each([
    { bundle: bundleID, key: "malformed" },
    { bundle: "", key: "malformed" },
    { bundle: bundleID, key: "wrong algorithm" },
  ])(
    "isolates invalid browser signing credentials: %j",
    async ({ bundle, key }) => {
      const privateKey =
        key === "malformed"
          ? "malformed-apple-private-key"
          : await exportPKCS8(
              (await generateKeyPair("RS256", { extractable: true }))
                .privateKey,
            );
      vi.stubEnv("AUTH_APPLE_APP_BUNDLE_IDENTIFIER", bundle);
      vi.stubEnv("AUTH_APPLE_PRIVATE_KEY", privateKey);
      const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
      const handler = await loadAppleAuth();
      const credentials = {
        email: "email-user@example.test",
        password: randomUUID(),
      };
      const signedUp = await handler(
        authRequest("sign-up/email", { ...credentials, name: "Email User" }),
      );
      expect(signedUp.status).toBe(200);
      const emailLogin = await handler(
        authRequest("sign-in/email", credentials),
      );
      expect(emailLogin.status).toBe(200);
      expect(emailLogin.headers.get("set-auth-token")).toBeTruthy();
      if (bundle) {
        const native = await signInApple(handler);
        expect(native.status).toBe(200);
        expect(native.headers.get("set-auth-token")).toBeTruthy();
      }
      const sessionCount = getAppleTestState().database.UserAuthSession.length;
      const browser = await handler(
        authRequest(
          "sign-in/social",
          { provider: "apple" },
          { Origin: origin },
        ),
      );
      expect(browser.status).toBe(503);
      expect(await browser.json()).toMatchObject({
        code: "APPLE_SIGN_IN_UNAVAILABLE",
      });
      expect(getAppleTestState().database.UserAuthSession).toHaveLength(
        sessionCount,
      );
      expect(appleTokenExchange).not.toHaveBeenCalled();
      expect(warn).toHaveBeenCalled();
      expect(JSON.stringify(warn.mock.calls)).not.toContain(privateKey);
      expect(
        (await handler(authRequest("sign-in/email", credentials))).status,
      ).toBe(200);
    },
  );

  it.each([
    "AUTH_APPLE_CLIENT_ID",
    "AUTH_APPLE_TEAM_ID",
    "AUTH_APPLE_KEY_ID",
    "AUTH_APPLE_PRIVATE_KEY",
  ])("keeps native login available when %s is absent", async (key) => {
    vi.stubEnv(key, "");
    const handler = await loadAppleAuth();
    const browser = await handler(
      authRequest("sign-in/social", { provider: "apple" }, { Origin: origin }),
    );
    expect(browser.status).toBe(400);
    expect((await browser.json()).code).toBe("APPLE_WEB_LOGIN_NOT_CONFIGURED");
    expect((await signInApple(handler)).status).toBe(200);
  });

  it("accepts a private key configured with escaped newlines", async () => {
    vi.stubEnv(
      "AUTH_APPLE_PRIVATE_KEY",
      process.env.AUTH_APPLE_PRIVATE_KEY!.replaceAll("\n", "\\n"),
    );
    const handler = await loadAppleAuth();
    const response = await finishAppleBrowserLogin(
      handler,
      await beginAppleBrowserLogin(handler),
    );
    expect(response.headers.get("location")).toBe(`${origin}/auth/callback`);
  });
});
