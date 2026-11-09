// @vitest-environment node
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  authRequest,
  loadAppleAuth,
  origin,
  signInApple,
} from "./apple-auth-fixture";
import {
  beginAppleBrowserLogin,
  finishAppleBrowserLogin,
  prepareAppleWebAuth,
} from "./apple-oauth-fixture";

beforeEach(prepareAppleWebAuth);
afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

describe("Apple browser configuration", () => {
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
