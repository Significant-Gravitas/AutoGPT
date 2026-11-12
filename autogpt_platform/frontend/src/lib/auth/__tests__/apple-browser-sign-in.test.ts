// @vitest-environment node
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  authRequest,
  getAppleTestState,
  loadAppleAuth,
  nativeHeaders,
  origin,
  signInApple,
} from "./apple-auth-fixture";
import {
  appleOrigin,
  appleTokenExchange,
  answerAppleCode,
  beginAppleBrowserLogin,
  cookiesOf,
  finishAppleBrowserLogin,
  prepareAppleWebAuth,
} from "./apple-oauth-fixture";

beforeEach(prepareAppleWebAuth);
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

describe("browser Sign in with Apple", () => {
  it("completes Apple's form callback into a browser cookie session and revokes it on logout", async () => {
    const handler = await loadAppleAuth();
    const login = await beginAppleBrowserLogin(handler);
    const response = await finishAppleBrowserLogin(handler, login);
    expect(response.status).toBe(302);
    expect(response.headers.get("location")).toBe(`${origin}/auth/callback`);
    expect(response.headers.get("set-auth-token")).toBeNull();
    expect(
      response.headers
        .getSetCookie()
        .some((cookie) => cookie.includes("HttpOnly")),
    ).toBe(true);
    const headers = {
      Cookie: cookiesOf(response),
      ...nativeHeaders,
      Origin: origin,
    };
    const session = await handler(
      authRequest("get-session", undefined, headers),
    );
    expect((await session.json()).user.name).toBe("Apple Browser");
    expect(session.headers.get("set-auth-token")).toBeNull();
    expect(
      (await handler(authRequest("token", undefined, headers))).status,
    ).toBe(200);
    const logout = await handler(authRequest("sign-out", {}, headers));
    expect(logout.status).toBe(200);
    expect(getAppleTestState().database.UserAuthSession).toHaveLength(0);
    expect(
      (
        await handler(
          authRequest("token", undefined, {
            Origin: origin,
            Cookie: cookiesOf(logout),
          }),
        )
      ).status,
    ).toBe(401);
    expect(appleTokenExchange).toHaveBeenCalledOnce();
  });

  it("uses the same account for native and browser login with grouped Apple identifiers", async () => {
    const handler = await loadAppleAuth();
    const native = await signInApple(handler);
    expect(native.status).toBe(200);
    const { user } = await native.json();
    answerAppleCode({ email: undefined, email_verified: undefined });
    const browser = await finishAppleBrowserLogin(
      handler,
      await beginAppleBrowserLogin(handler),
    );
    expect(browser.headers.get("location")).toBe(`${origin}/auth/callback`);
    const session = await handler(
      authRequest("get-session", undefined, { Cookie: cookiesOf(browser) }),
    );
    expect((await session.json()).user.id).toBe(user.id);
    expect(getAppleTestState().database.UserAuthIdentity).toHaveLength(1);
  });

  it("supports browser-only configuration without a native bundle identifier", async () => {
    vi.stubEnv("AUTH_APPLE_APP_BUNDLE_IDENTIFIER", "");
    const handler = await loadAppleAuth();
    const response = await finishAppleBrowserLogin(
      handler,
      await beginAppleBrowserLogin(handler),
    );
    expect(response.headers.get("location")).toBe(`${origin}/auth/callback`);
    expect((await signInApple(handler)).status).toBe(401);
  });

  it("generates a fresh client secret after the user spends several minutes authorizing", async () => {
    vi.useFakeTimers({ toFake: ["Date"] });
    const handler = await loadAppleAuth();
    const login = await beginAppleBrowserLogin(handler);
    vi.setSystemTime(Date.now() + 310_000);
    const response = await finishAppleBrowserLogin(handler, login);
    expect(response.headers.get("location")).toBe(`${origin}/auth/callback`);
    expect(appleTokenExchange).toHaveBeenCalledWith(
      expect.objectContaining({ issuedAt: Math.floor(Date.now() / 1000) }),
    );
  });

  it("does not trust Apple's origin on other authentication endpoints", async () => {
    const handler = await loadAppleAuth();
    const signedIn = await signInApple(handler);
    expect(signedIn.status).toBe(200);
    const response = await handler(
      authRequest(
        "sign-out",
        {},
        { Origin: appleOrigin, Cookie: cookiesOf(signedIn) },
      ),
    );
    expect(response.status).toBe(403);
  });
});
