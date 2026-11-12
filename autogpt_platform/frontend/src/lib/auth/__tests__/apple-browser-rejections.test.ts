// @vitest-environment node
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  authRequest,
  bundleID,
  getAppleTestState,
  loadAppleAuth,
  origin,
} from "./apple-auth-fixture";
import {
  answerAppleCode,
  appleTokenExchange,
  beginAppleBrowserLogin,
  finishAppleBrowserLogin,
  prepareAppleWebAuth,
} from "./apple-oauth-fixture";

beforeEach(prepareAppleWebAuth);
afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

function expectNoSession(response: Response) {
  expect(response.headers.get("set-auth-token")).toBeNull();
  expect(getAppleTestState().database.UserAuthSession).toHaveLength(0);
  expect(getAppleTestState().database.UserAuthIdentity).toHaveLength(0);
}

function callbackError(response: Response) {
  expect(response.status).toBe(302);
  return new URL(response.headers.get("location")!).searchParams.get("error");
}

describe("Apple browser callback validation", () => {
  it.each(["missing cookie", "another browser", "unknown state"])(
    "rejects a callback with %s before exchanging the code",
    async (scenario) => {
      const handler = await loadAppleAuth();
      const login = await beginAppleBrowserLogin(handler);
      if (scenario === "missing cookie") login.cookie = "";
      if (scenario === "another browser") {
        login.cookie = (await beginAppleBrowserLogin(handler)).cookie;
      }
      if (scenario === "unknown state") login.state = "unrecognized-state";
      const response = await finishAppleBrowserLogin(handler, login);
      expect(callbackError(response)).toMatch(/state/);
      expectNoSession(response);
      expect(appleTokenExchange).not.toHaveBeenCalled();
    },
  );

  it("consumes state so a callback cannot create a second session", async () => {
    const handler = await loadAppleAuth();
    const login = await beginAppleBrowserLogin(handler);
    const first = await finishAppleBrowserLogin(handler, login);
    expect(first.headers.get("location")).toBe(`${origin}/auth/callback`);
    const replay = await finishAppleBrowserLogin(handler, login);
    expect(callbackError(replay)).toMatch(/state/);
    expect(getAppleTestState().database.UserAuthSession).toHaveLength(1);
    expect(appleTokenExchange).toHaveBeenCalledOnce();
  });

  it("returns a cancelled authorization without creating a session", async () => {
    const handler = await loadAppleAuth();
    const response = await finishAppleBrowserLogin(
      handler,
      await beginAppleBrowserLogin(handler),
      { error: "access_denied", code: "" },
    );
    expect(callbackError(response)).toBe("access_denied");
    expectNoSession(response);
    expect(appleTokenExchange).not.toHaveBeenCalled();
  });

  it.each([
    { aud: bundleID },
    { aud: "another-service" },
    { iss: "https://untrusted.example.test" },
    { exp: 1 },
    { sub: undefined },
  ])("rejects callback identity claims %j", async (claims) => {
    answerAppleCode(claims);
    const handler = await loadAppleAuth();
    const response = await finishAppleBrowserLogin(
      handler,
      await beginAppleBrowserLogin(handler),
    );
    expect(callbackError(response)).toBe("unable_to_get_user_info");
    expectNoSession(response);
    expect(appleTokenExchange).toHaveBeenCalledOnce();
  });

  it.each([{ omitIDToken: true }, { rejectCode: true }])(
    "refuses an incomplete or rejected code exchange: %j",
    async (options) => {
      answerAppleCode({}, options);
      const handler = await loadAppleAuth();
      const response = await finishAppleBrowserLogin(
        handler,
        await beginAppleBrowserLogin(handler),
      );
      expect(callbackError(response)).toMatch(
        /unable_to_get_user_info|invalid_code/,
      );
      expectNoSession(response);
      expect(appleTokenExchange).toHaveBeenCalledOnce();
    },
  );

  it("fails closed when the callback identity cannot retrieve Apple's keys", async () => {
    const handler = await loadAppleAuth();
    const login = await beginAppleBrowserLogin(handler);
    const appleFetch = globalThis.fetch;
    vi.stubGlobal("fetch", (input: RequestInfo | URL, init?: RequestInit) => {
      if (String(input).endsWith("/auth/keys"))
        throw new Error("Apple unavailable");
      return appleFetch(input, init);
    });
    const response = await finishAppleBrowserLogin(handler, login);
    expect(callbackError(response)).toBe("unable_to_get_user_info");
    expectNoSession(response);
    expect(appleTokenExchange).toHaveBeenCalledOnce();
  });

  it("retains the signup allowlist for browser Apple login", async () => {
    vi.stubEnv("AUTH_SIGNUP_ALLOWLIST", "invited@example.test");
    const handler = await loadAppleAuth();
    const response = await finishAppleBrowserLogin(
      handler,
      await beginAppleBrowserLogin(handler),
    );
    expect(response.headers.get("location")).not.toBe(
      `${origin}/auth/callback`,
    );
    expectNoSession(response);
    expect(appleTokenExchange).toHaveBeenCalledOnce();
  });

  it("refuses an untrusted destination before starting Apple login", async () => {
    const handler = await loadAppleAuth();
    const response = await handler(
      authRequest(
        "sign-in/social",
        {
          provider: "apple",
          callbackURL: "https://untrusted.example.test/callback",
        },
        { Origin: origin },
      ),
    );
    expect(response.status).toBe(403);
    expectNoSession(response);
    expect(appleTokenExchange).not.toHaveBeenCalled();
  });
});
