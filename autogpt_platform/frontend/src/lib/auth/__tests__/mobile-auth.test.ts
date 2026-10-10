// @vitest-environment node
import { randomBytes } from "node:crypto";
import { beforeEach, describe, expect, it } from "vitest";
import {
  CHALLENGE,
  ORIGIN,
  STATE,
  VERIFIER,
  cookieHeader,
  createTestAuth,
} from "./mobile-auth-fixtures";

describe("mobile browser authentication handoff", () => {
  let auth: ReturnType<typeof createTestAuth>;
  let cookies: string;
  let userID: string;

  beforeEach(async () => {
    auth = createTestAuth();
    const response = await auth.api.signUpEmail({
      body: {
        name: "Mobile Tester",
        email: "mobile-tester@agpt.co",
        password: "a-long-test-password", // pragma: allowlist secret
      },
      asResponse: true,
    });
    cookies = cookieHeader(response);
    const body = await response.json();
    userID = body.user.id;
  });

  function request(
    path: string,
    body?: Record<string, string>,
    headers: Record<string, string> = {},
  ) {
    return auth.handler(
      new Request(`${ORIGIN}/api/auth/mobile/${path}`, {
        method: body ? "POST" : "GET",
        headers: {
          ...(body
            ? { "Content-Type": "application/json", Origin: ORIGIN }
            : {}),
          ...headers,
        },
        ...(body
          ? {
              body: JSON.stringify(
                path === "authorize"
                  ? { expected_user_id: userID, ...body }
                  : body,
              ),
            }
          : {}),
      }),
    );
  }

  async function authorize(headers = { Cookie: cookies, Origin: ORIGIN }) {
    return request(
      "authorize",
      { code_challenge: CHALLENGE, state: STATE },
      headers,
    );
  }

  async function issueCode() {
    const response = await authorize();
    expect(response.status).toBe(200);
    const body = await response.json();
    const callback = new URL(body.url);
    expect(callback.origin).toBe("null");
    expect(callback.protocol).toBe("autogpt:");
    expect(callback.host).toBe("auth");
    expect(callback.pathname).toBe("/callback");
    expect(callback.searchParams.get("state")).toBe(STATE);
    return callback.searchParams.get("code")!;
  }

  it("preserves a validated handoff through the existing login flow", async () => {
    const response = await request(
      `start?code_challenge=${CHALLENGE}&state=${STATE}`,
    );
    expect(response.status).toBe(302);
    const login = new URL(response.headers.get("location")!, ORIGIN);
    expect(login.pathname).toBe("/login");
    const next = new URL(login.searchParams.get("next")!, ORIGIN);
    expect(next.pathname).toBe("/auth/mobile");
    expect(next.searchParams.get("code_challenge")).toBe(CHALLENGE);
    expect(next.searchParams.get("state")).toBe(STATE);
  });

  it("requires explicit authenticated consent before issuing a ticket", async () => {
    const response = await request(
      `start?code_challenge=${CHALLENGE}&state=${STATE}`,
      undefined,
      { Cookie: cookies },
    );
    expect(response.headers.get("location")).toBe(
      `${ORIGIN}/auth/mobile?code_challenge=${CHALLENGE}&state=${STATE}`,
    );
    expect(await response.text()).not.toContain("autogpt://");
    const unauthorized = await authorize({ Cookie: "", Origin: ORIGIN });
    expect(unauthorized.status).toBe(401);
  });

  it("rejects cross-origin consent and missing origin with browser cookies", async () => {
    expect(
      (await authorize({ Cookie: cookies, Origin: "https://attacker.example" }))
        .status,
    ).toBe(403);
    expect((await authorize({ Cookie: cookies, Origin: "" })).status).toBe(403);
  });

  it.each([
    ["short", STATE],
    ["A".repeat(44), STATE],
    ["!".repeat(43), STATE],
    [CHALLENGE, "short"],
    [CHALLENGE, "A".repeat(129)],
    [CHALLENGE, "https://attacker.example"],
  ])("rejects malformed challenge or state", async (challenge, state) => {
    const response = await request(
      `start?code_challenge=${encodeURIComponent(challenge)}&state=${encodeURIComponent(state)}`,
    );
    expect(response.status).toBe(400);
  });

  it("never accepts an arbitrary callback destination", async () => {
    const response = await request(
      "authorize",
      {
        code_challenge: CHALLENGE,
        state: STATE,
        redirect_uri: "https://attacker.example",
      },
      { Cookie: cookies, Origin: ORIGIN },
    );
    expect(response.status).toBe(400);
  });

  it("sets a fresh secure HttpOnly session without disclosing it in JSON", async () => {
    const code = await issueCode();
    const response = await request("exchange", {
      code,
      code_verifier: VERIFIER,
    });
    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({ success: true });
    expect(response.headers.get("cache-control")).toContain("no-store");
    const setCookies = response.headers.getSetCookie();
    const sessionCookie = setCookies.find((value) =>
      value.startsWith("__Secure-better-auth.session_token="),
    );
    expect(sessionCookie).toContain("HttpOnly");
    expect(sessionCookie).toContain("Secure");
    expect(sessionCookie).toContain("SameSite=Lax");
    expect(cookieHeader(response)).not.toBe(cookies);
    const session = await auth.api.getSession({
      headers: new Headers({ Cookie: cookieHeader(response) }),
    });
    expect(session?.user.id).toBe(userID);
  });

  it("rejects the wrong verifier without consuming the legitimate ticket", async () => {
    const code = await issueCode();
    const wrong = await request("exchange", {
      code,
      code_verifier: randomBytes(32).toString("base64url"),
    });
    expect(wrong.status).toBe(400);
    expect(wrong.headers.has("set-cookie")).toBe(false);
    const valid = await request("exchange", { code, code_verifier: VERIFIER });
    expect(valid.status).toBe(200);
  });

  it("allows exactly one concurrent exchange and rejects replay", async () => {
    const code = await issueCode();
    const results = await Promise.all([
      request("exchange", { code, code_verifier: VERIFIER }),
      request("exchange", { code, code_verifier: VERIFIER }),
    ]);
    expect(results.map((response) => response.status).sort()).toEqual([
      200, 400,
    ]);
    expect(
      (await request("exchange", { code, code_verifier: VERIFIER })).status,
    ).toBe(400);
  });

  it("rejects an expired handoff", async () => {
    const code = await issueCode();
    const context = await auth.$context;
    const records = await context.adapter.findMany<{
      id: string;
      identifier: string;
    }>({ model: "verification" });
    const handoff = records.find((row) =>
      String(row.identifier).startsWith("mobile-auth:"),
    );
    expect(handoff).toBeDefined();
    await context.adapter.update({
      model: "verification",
      where: [{ field: "id", value: handoff!.id }],
      update: { expiresAt: new Date(Date.now() - 1000) },
    });
    expect(
      (await request("exchange", { code, code_verifier: VERIFIER })).status,
    ).toBe(400);
  });

  it("rechecks browser session revocation before creating an app session", async () => {
    const code = await issueCode();
    await auth.api.signOut({ headers: new Headers({ Cookie: cookies }) });
    const response = await request("exchange", {
      code,
      code_verifier: VERIFIER,
    });
    expect(response.status).toBe(401);
    expect(response.headers.has("set-cookie")).toBe(false);
  });

  it("rejects banned users even when their session cookie cache remains valid", async () => {
    const code = await issueCode();
    const context = await auth.$context;
    await context.internalAdapter.updateUser(userID, {
      banned: true,
      banExpires: new Date(Date.now() + 60000),
    });
    expect((await authorize()).status).toBe(403);
    const response = await request("exchange", {
      code,
      code_verifier: VERIFIER,
    });
    expect(response.status).toBe(403);
    expect(response.headers.has("set-cookie")).toBe(false);
  });
});
