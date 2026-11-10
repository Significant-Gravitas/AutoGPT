// @vitest-environment node
import { randomBytes, createHash } from "node:crypto";
import { betterAuth } from "better-auth";
import { memoryAdapter } from "better-auth/adapters/memory";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { admin } from "better-auth/plugins";
import { MOBILE_AUTH_CALLBACK_PATH } from "../mobile-app-links";
import { mobileAuth } from "../mobile-auth";
import { mobileConsentPage } from "../mobile-consent-page";

const baseURL = "https://platform.example.com";
const verifier = randomBytes(32).toString("base64url");
const challenge = createHash("sha256").update(verifier).digest("base64url");
const state = randomBytes(32).toString("base64url");

function createAuth() {
  return betterAuth({
    baseURL,
    secret: randomBytes(32).toString("hex"),
    database: memoryAdapter({
      user: [],
      session: [],
      account: [],
      verification: [],
    }),
    emailAndPassword: { enabled: true },
    plugins: [admin(), mobileAuth()],
    rateLimit: { enabled: false },
  });
}

describe("mobile sign-in", () => {
  let auth = createAuth();

  beforeEach(() => {
    auth = createAuth();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  function post(
    path: string,
    body: Record<string, string>,
    cookie = "",
    origin = baseURL,
  ) {
    return auth.handler(
      new Request(`${baseURL}/api/auth/mobile/${path}`, {
        method: "POST",
        headers: {
          cookie,
          origin,
          "content-type": "application/x-www-form-urlencoded",
        },
        body: new URLSearchParams(body),
      }),
    );
  }

  async function grant() {
    const signup = await auth.api.signUpEmail({
      body: {
        email: "mobile@example.com",
        name: "Mobile",
        password: randomBytes(24).toString("base64url"),
      },
      returnHeaders: true,
    });
    const cookie = signup.headers.get("set-cookie")!.split(";")[0];
    const response = await post("approve", { challenge, state }, cookie);
    expect(response.status).toBe(302);
    const code = new URL(response.headers.get("location")!).searchParams.get(
      "code",
    )!;
    return { cookie, code, userID: signup.response.user.id };
  }

  it("sends a signed-out user through the existing login flow", async () => {
    const response = await auth.handler(
      new Request(
        `${baseURL}/api/auth/mobile/start?${new URLSearchParams({ challenge, state })}`,
      ),
    );
    expect(response.status).toBe(302);
    const destination = new URL(response.headers.get("location")!);
    expect(destination.origin).toBe(baseURL);
    expect(destination.pathname).toBe("/login");
    expect(destination.searchParams.get("next")).toBe(
      `/api/auth/mobile/start?${new URLSearchParams({ challenge, state })}`,
    );
  });

  it("requires consent, then gives the phone its own session without exposing a token", async () => {
    const signup = await auth.api.signUpEmail({
      body: {
        email: "mobile@example.com",
        name: "Mobile",
        password: randomBytes(24).toString("base64url"),
      },
      returnHeaders: true,
    });
    expect(signup.response.token).toBeTruthy();
    const cookie = signup.headers.get("set-cookie")?.split(";")[0] ?? "";
    expect(cookie).not.toBe("");
    expect(
      await auth.api.getSession({ headers: new Headers({ cookie }) }),
    ).not.toBeNull();
    const start = await auth.handler(
      new Request(
        `${baseURL}/api/auth/mobile/start?${new URLSearchParams({ challenge, state })}`,
        { headers: { cookie } },
      ),
    );
    expect(await start.text()).toContain("Continue to AutoGPT");
    const approved = await auth.handler(
      new Request(`${baseURL}/api/auth/mobile/approve`, {
        method: "POST",
        headers: {
          cookie,
          origin: baseURL,
          "content-type": "application/x-www-form-urlencoded",
        },
        body: new URLSearchParams({ challenge, state }),
      }),
    );
    expect(approved.status).toBe(302);
    const callback = new URL(approved.headers.get("location")!);
    expect(callback.origin).toBe(baseURL);
    expect(callback.pathname).toBe(MOBILE_AUTH_CALLBACK_PATH);
    expect(callback.searchParams.get("state")).toBe(state);
    expect(callback.searchParams.has("token")).toBe(false);
    const code = callback.searchParams.get("code")!;
    const exchange = await auth.handler(
      new Request(`${baseURL}/api/auth/mobile/exchange`, {
        method: "POST",
        headers: {
          origin: baseURL,
          "content-type": "application/x-www-form-urlencoded",
        },
        body: new URLSearchParams({ code, verifier }),
      }),
    );
    expect(exchange.status).toBe(302);
    expect(exchange.headers.get("location")).toBe(`${baseURL}/home`);
    const mobileCookie =
      exchange.headers.get("set-cookie")?.split(";")[0] ?? "";
    expect(mobileCookie).not.toBe(cookie);
    const session = await auth.api.getSession({
      headers: new Headers({ cookie: mobileCookie }),
    });
    expect(session?.user.email).toBe("mobile@example.com");
    await auth.api.signOut({ headers: new Headers({ cookie }) });
    expect(
      (
        await auth.api.getSession({
          headers: new Headers({ cookie: mobileCookie }),
        })
      )?.user.email,
    ).toBe("mobile@example.com");
  });

  it("rejects another device's verifier without consuming the real device's code", async () => {
    const { code } = await grant();
    expect(
      (
        await post("exchange", {
          code,
          verifier: randomBytes(32).toString("base64url"),
        })
      ).status,
    ).toBe(401);
    expect((await post("exchange", { code, verifier })).status).toBe(302);
  });

  it.each(["https://evil.example.com", "", "null"])(
    "rejects a cookieless exchange from origin %s without consuming the code",
    async (origin) => {
      const { code } = await grant();
      const rejected = await post("exchange", { code, verifier }, "", origin);
      expect(rejected.status).toBe(403);
      expect(rejected.headers.has("set-cookie")).toBe(false);
      expect((await post("exchange", { code, verifier })).status).toBe(302);
    },
  );

  it("rejects an exchange without an Origin header", async () => {
    const { code } = await grant();
    const response = await auth.handler(
      new Request(`${baseURL}/api/auth/mobile/exchange`, {
        method: "POST",
        headers: { "content-type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams({ code, verifier }),
      }),
    );
    expect(response.status).toBe(403);
    expect(response.headers.has("set-cookie")).toBe(false);
    expect((await post("exchange", { code, verifier })).status).toBe(302);
  });

  it("only redeems a code once, including concurrent requests", async () => {
    const { code } = await grant();
    const responses = await Promise.all([
      post("exchange", { code, verifier }),
      post("exchange", { code, verifier }),
    ]);
    expect(responses.map((response) => response.status).sort()).toEqual([
      302, 401,
    ]);
    expect((await post("exchange", { code, verifier })).status).toBe(401);
  });

  it("rejects codes after one minute", async () => {
    const { code } = await grant();
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(Date.now() + 61_000);
    expect((await post("exchange", { code, verifier })).status).toBe(401);
  });

  it("rejects a code if the source session was revoked", async () => {
    const { code, cookie } = await grant();
    await auth.api.signOut({ headers: new Headers({ cookie }) });
    expect((await post("exchange", { code, verifier })).status).toBe(401);
  });

  it("rejects a user banned after approving sign-in", async () => {
    const { code, userID } = await grant();
    const context = await auth.$context;
    await context.adapter.update({
      model: "user",
      where: [{ field: "id", value: userID }],
      update: { banned: true },
    });
    expect((await post("exchange", { code, verifier })).status).toBe(403);
  });

  it("allows sign-in after a temporary ban expires", async () => {
    const { code, userID } = await grant();
    const context = await auth.$context;
    await context.adapter.update({
      model: "user",
      where: [{ field: "id", value: userID }],
      update: { banned: true, banExpires: new Date(Date.now() - 1_000) },
    });
    expect((await post("exchange", { code, verifier })).status).toBe(302);
  });

  it("escapes consent fields and lets the form post only to its own origin", async () => {
    const page = await mobileConsentPage('<img src=x onerror="alert(1)">', {
      challenge: '"><script>alert(1)</script>',
      state: '"><img src=x onerror="alert(1)">',
    }).text();
    expect(page).not.toContain("<script>");
    expect(page).not.toContain("<img");
    expect(page).toContain('http-equiv="Content-Security-Policy"');
    expect(page).toContain("form-action 'self';");
  });

  it("requires a session and the exact origin to approve sign-in", async () => {
    expect((await post("approve", { challenge, state })).status).toBe(401);
    const { cookie } = await grant();
    expect(
      (
        await post(
          "approve",
          { challenge, state },
          cookie,
          "https://evil.example.com",
        )
      ).status,
    ).toBe(403);
    expect(
      (await post("approve", { challenge, state }, cookie, "")).status,
    ).toBe(403);
  });

  it("rejects malformed challenges before sending the user to login", async () => {
    const response = await auth.handler(
      new Request(
        `${baseURL}/api/auth/mobile/start?challenge=bad&state=${state}`,
      ),
    );
    expect(response.status).toBe(400);
  });
});
