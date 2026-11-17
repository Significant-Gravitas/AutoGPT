// @vitest-environment node
import { betterAuth } from "better-auth";
import { memoryAdapter } from "better-auth/adapters/memory";
import { describe, expect, it } from "vitest";
import { nativeSessionBearer } from "../native-session-bearer";

const origin = "http://localhost:3000";
const nativeHeaders = { "X-AutoGPT-Native": "1" };

function createAuth(
  database: Record<string, unknown[]> = { user: [], session: [], account: [] },
) {
  return betterAuth({
    baseURL: origin,
    secret: "test-only-native-session-secret-32-characters", // pragma: allowlist secret
    database: memoryAdapter(database),
    emailAndPassword: { enabled: true },
    plugins: [nativeSessionBearer()],
  });
}

function signup(headers: Record<string, string> = {}) {
  return new Request(`${origin}/api/auth/sign-up/email`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...headers },
    body: JSON.stringify({
      email: "native@example.test",
      password: "only-a-local-test-password", // pragma: allowlist secret
      name: "Native",
    }),
  });
}

describe("native session credentials", () => {
  it.each([
    {},
    { ...nativeHeaders, Origin: origin },
    { ...nativeHeaders, "Sec-Fetch-Mode": "cors" },
    { ...nativeHeaders, "Sec-Fetch-Site": "same-origin" },
    { ...nativeHeaders, "Sec-Fetch-Dest": "empty" },
    { ...nativeHeaders, "Sec-Fetch-User": "?1" },
    { ...nativeHeaders, Cookie: "some-browser-cookie=present" },
  ])(
    "keeps browser session credentials in HttpOnly cookies: %j",
    async (headers) => {
      const auth = createAuth();
      const response = await auth.handler(signup(headers));
      expect(response.status).toBe(200);
      expect(
        response.headers
          .getSetCookie()
          .some((cookie) => cookie.includes("HttpOnly")),
      ).toBe(true);
      expect(response.headers.has("set-auth-token")).toBe(false);
      expect(
        response.headers.get("access-control-expose-headers") ?? "",
      ).not.toContain("set-auth-token");
    },
  );

  it("does not export a browser session during its daily refresh, even with the native marker", async () => {
    const database: Record<string, unknown[]> = {
      user: [],
      session: [],
      account: [],
    };
    const auth = createAuth(database);
    const created = await auth.handler(signup(nativeHeaders));
    const cookie = created.headers.getSetCookie()[0].split(";")[0];
    const stored = database.session[0] as { expiresAt: Date };
    stored.expiresAt = new Date(
      stored.expiresAt.getTime() - 2 * 24 * 60 * 60 * 1000,
    );
    const refreshed = await auth.handler(
      new Request(`${origin}/api/auth/get-session`, {
        headers: { ...nativeHeaders, Cookie: cookie },
      }),
    );
    expect(refreshed.status).toBe(200);
    expect((await refreshed.json()).user.email).toBe("native@example.test");
    expect(refreshed.headers.getSetCookie().length).toBeGreaterThan(0);
    expect(refreshed.headers.has("set-auth-token")).toBe(false);
  });

  it("issues a signed native credential that authenticates and can be revoked", async () => {
    const auth = createAuth();
    const response = await auth.handler(signup(nativeHeaders));
    expect(response.status).toBe(200);
    const signed = response.headers.get("set-auth-token");
    expect(signed).toContain(".");
    const { token: unsigned } = await response.json();
    const session = (token: string) =>
      auth.handler(
        new Request(`${origin}/api/auth/get-session`, {
          headers: { ...nativeHeaders, Authorization: `Bearer ${token}` },
        }),
      );
    expect(await (await session(unsigned)).json()).toBeNull();
    const authenticated = await (await session(signed!)).json();
    expect(authenticated.user.email).toBe("native@example.test");
    const logout = await auth.handler(
      new Request(`${origin}/api/auth/sign-out`, {
        method: "POST",
        headers: {
          ...nativeHeaders,
          Authorization: `Bearer ${signed}`,
          "Content-Type": "application/json",
        },
        body: "{}",
      }),
    );
    expect(logout.status).toBe(200);
    expect(await (await session(signed!)).json()).toBeNull();
  });
});
