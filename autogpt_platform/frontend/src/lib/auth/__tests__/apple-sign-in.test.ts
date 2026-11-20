// @vitest-environment node
import { createHash } from "node:crypto";
import { createLocalJWKSet, jwtVerify } from "jose";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  getAppleTestState,
  authRequest,
  loadAppleAuth,
  nativeHeaders,
  nonce,
  origin,
  prepareAppleAuth,
  signInApple,
} from "./apple-auth-fixture";

const appleTestState = getAppleTestState();

beforeEach(prepareAppleAuth);
afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

describe("native Sign in with Apple", () => {
  it("exchanges an Apple identity for a renewable signed session and revokes it on logout", async () => {
    const handler = await loadAppleAuth();
    const response = await signInApple(handler);
    expect(response.status).toBe(200);
    const signedSession = response.headers.get("set-auth-token");
    expect(signedSession).toContain(".");
    const { user, token: unsigned } = await response.json();
    expect(user).toMatchObject({
      email: "apple-user@privaterelay.appleid.com",
      emailVerified: true,
      name: "Apple User",
    });
    expect(appleTestState.provisioned).toHaveBeenCalledWith(
      expect.anything(),
      expect.objectContaining({ id: user.id, email: user.email }),
    );
    const headers = {
      ...nativeHeaders,
      Authorization: `Bearer ${signedSession}`,
    };
    const jwks = await (await handler(authRequest("jwks"))).json();
    for (let attempt = 0; attempt < 2; attempt++) {
      const tokenResponse = await handler(
        authRequest("token", undefined, headers),
      );
      expect(tokenResponse.status).toBe(200);
      const { token } = await tokenResponse.json();
      const { payload } = await jwtVerify(token, createLocalJWKSet(jwks), {
        issuer: origin,
        audience: "authenticated",
      });
      expect(payload.sub).toBe(user.id);
      expect(payload.sid).toEqual(expect.any(String));
    }
    const refused = await handler(
      authRequest("get-session", undefined, {
        ...nativeHeaders,
        Authorization: `Bearer ${unsigned}`,
      }),
    );
    expect(await refused.json()).toBeNull();
    expect((await handler(authRequest("sign-out", {}, headers))).status).toBe(
      200,
    );
    const signedOut = await handler(
      authRequest("get-session", undefined, headers),
    );
    expect(await signedOut.json()).toBeNull();
    expect(
      (await handler(authRequest("token", undefined, headers))).status,
    ).toBe(401);
  });

  it("accepts the SHA-256 nonce used by native Apple authorization", async () => {
    const handler = await loadAppleAuth();
    const response = await signInApple(handler, {
      nonce: createHash("sha256").update(nonce).digest("hex"),
    });
    expect(response.status).toBe(200);
  });

  it("signs the same Apple subject into its existing account without repeating profile consent", async () => {
    const handler = await loadAppleAuth();
    const first = await signInApple(handler);
    expect(first.status).toBe(200);
    const firstUser = (await first.json()).user;
    const repeated = await signInApple(handler, {
      email: undefined,
      email_verified: undefined,
    });
    expect(repeated.status).toBe(200);
    expect((await repeated.json()).user.id).toBe(firstUser.id);
    expect(appleTestState.database.UserAuthIdentity).toHaveLength(1);
  });

  it("keeps browser sessions in cookies without exporting a native bearer", async () => {
    const handler = await loadAppleAuth();
    const response = await signInApple(
      handler,
      {},
      {
        headers: { ...nativeHeaders, Origin: origin, "Sec-Fetch-Mode": "cors" },
      },
    );
    expect(response.status).toBe(200);
    expect(response.headers.get("set-auth-token")).toBeNull();
    expect(
      response.headers
        .getSetCookie()
        .some((cookie) => cookie.includes("HttpOnly")),
    ).toBe(true);
  });
});
