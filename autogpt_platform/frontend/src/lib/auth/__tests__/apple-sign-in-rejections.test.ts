// @vitest-environment node
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  appleIdentity,
  authRequest,
  getAppleTestState,
  loadAppleAuth,
  nonce,
  prepareAppleAuth,
  signInApple,
} from "./apple-auth-fixture";

const state = getAppleTestState();

beforeEach(prepareAppleAuth);
afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

describe("Apple identity validation", () => {
  it("stays disabled without an explicitly configured Apple App ID", async () => {
    vi.stubEnv("AUTH_APPLE_APP_BUNDLE_IDENTIFIER", "");
    const response = await signInApple(await loadAppleAuth());
    expect(response.status).toBe(404);
    expect(state.database.UserAuthSession).toHaveLength(0);
  });

  it.each([
    { aud: "com.someone-else.app" },
    { iss: "https://untrusted.example.test" },
    { exp: 1 },
    { iat: 1 },
    { iat: Math.floor(Date.now() / 1000) + 3600 },
    { sub: undefined },
    { exp: undefined },
    { nonce: "another-authorization" },
    { nonce: undefined },
  ])("rejects invalid signed claims: %j", async (claims) => {
    const response = await signInApple(await loadAppleAuth(), claims);
    expect(response.status).toBe(401);
    expect(response.headers.get("set-auth-token")).toBeNull();
    expect(state.database.UserAuthIdentity).toHaveLength(0);
    expect(state.database.UserAuthSession).toHaveLength(0);
  });

  it.each([null, "", "   "])(
    "requires the authorization nonce: %j",
    async (value) => {
      const response = await signInApple(
        await loadAppleAuth(),
        {},
        { nonce: value },
      );
      expect(response.status).toBe(400);
      expect((await response.json()).code).toBe("APPLE_NONCE_REQUIRED");
      expect(state.database.UserAuthSession).toHaveLength(0);
    },
  );

  it("refuses an unsigned token", async () => {
    const valid = await appleIdentity();
    const [, payload] = valid.split(".");
    const unsigned = `${Buffer.from(JSON.stringify({ alg: "none" })).toString("base64url")}.${payload}.`;
    const handler = await loadAppleAuth();
    const response = await handler(
      authRequest("sign-in/social", {
        provider: "apple",
        idToken: { token: unsigned, nonce },
      }),
    );
    expect(response.status).toBe(401);
    expect(state.database.UserAuthSession).toHaveLength(0);
  });

  it("refuses a token with a forged signature", async () => {
    const [header, payload, signature] = (await appleIdentity()).split(".");
    const forged = `${signature[0] === "A" ? "B" : "A"}${signature.slice(1)}`;
    const handler = await loadAppleAuth();
    const response = await handler(
      authRequest("sign-in/social", {
        provider: "apple",
        idToken: { token: `${header}.${payload}.${forged}`, nonce },
      }),
    );
    expect(response.status).toBe(401);
    expect(state.database.UserAuthSession).toHaveLength(0);
  });

  it("fails closed when Apple's signing keys cannot be retrieved", async () => {
    const handler = await loadAppleAuth();
    vi.stubGlobal(
      "fetch",
      vi.fn().mockRejectedValue(new Error("Apple unavailable")),
    );
    const response = await signInApple(handler);
    expect(response.status).toBe(401);
    expect(state.database.UserAuthSession).toHaveLength(0);
  });

  it("refuses the browser redirect flow instead of starting an unconfigured OAuth exchange", async () => {
    const handler = await loadAppleAuth();
    const response = await handler(
      authRequest("sign-in/social", { provider: "apple" }),
    );
    expect(response.status).toBe(400);
    expect((await response.json()).code).toBe("APPLE_ID_TOKEN_REQUIRED");
  });

  it("does not use a client-supplied email to sign an unknown Apple subject into an account", async () => {
    const handler = await loadAppleAuth();
    const first = await signInApple(handler);
    expect(first.status).toBe(200);
    const { user } = await first.json();
    const response = await handler(
      authRequest("sign-in/social", {
        provider: "apple",
        idToken: {
          token: await appleIdentity({
            sub: "unknown-apple-user",
            email: undefined,
          }),
          nonce,
          user: { email: user.email },
        },
      }),
    );
    expect(response.status).toBe(401);
    expect(state.database.UserAuthIdentity).toHaveLength(1);
    expect(state.database.UserAuthSession).toHaveLength(1);
  });

  it("does not restore an email from an orphaned Apple account", async () => {
    const handler = await loadAppleAuth();
    expect((await signInApple(handler)).status).toBe(200);
    state.database.UserAuthIdentity.length = 0;
    const response = await signInApple(handler, { email: undefined });
    expect(response.status).toBe(401);
  });

  it("cannot link an unverified Apple email to an existing verified account", async () => {
    const handler = await loadAppleAuth();
    expect((await signInApple(handler)).status).toBe(200);
    const response = await signInApple(handler, {
      sub: "unlinked-apple-user",
      email_verified: "false",
    });
    expect(response.status).toBe(401);
    expect(state.database.UserAuthIdentity).toHaveLength(1);
    expect(state.database.UserAuthAccount).toHaveLength(1);
    expect(state.database.UserAuthSession).toHaveLength(1);
  });

  it("honors account bans for returning Apple users", async () => {
    const handler = await loadAppleAuth();
    expect((await signInApple(handler)).status).toBe(200);
    Object.assign(state.database.UserAuthIdentity[0]!, { banned: true });
    const response = await signInApple(handler);
    expect(response.status).toBe(403);
    expect(response.headers.get("set-auth-token")).toBeNull();
    expect(state.database.UserAuthSession).toHaveLength(1);
  });

  it("enforces the signup allowlist on Apple's verified email", async () => {
    vi.stubEnv("AUTH_SIGNUP_ALLOWLIST", "invited@example.test");
    const handler = await loadAppleAuth();
    const response = await signInApple(handler);
    expect(response.status).toBeGreaterThanOrEqual(400);
    expect(state.database.UserAuthIdentity).toHaveLength(0);
    expect(state.database.UserAuthSession).toHaveLength(0);
  });
});
