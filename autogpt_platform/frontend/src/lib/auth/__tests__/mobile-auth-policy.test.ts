// @vitest-environment node
import { beforeEach, describe, expect, it } from "vitest";
import {
  CHALLENGE,
  ORIGIN,
  STATE,
  VERIFIER,
  cookieHeader,
  createTestAuth,
} from "./mobile-auth-fixtures";

describe("mobile handoff session policy", () => {
  let auth: ReturnType<typeof createTestAuth>;
  let cookies: string;
  let userID: string;
  let allowNewSession = true;
  let beforeNewSession: (() => Promise<void>) | null = null;

  beforeEach(async () => {
    allowNewSession = true;
    beforeNewSession = null;
    auth = createTestAuth(async () => {
      await beforeNewSession?.();
      return allowNewSession;
    });
    const response = await auth.api.signUpEmail({
      body: {
        name: "Mobile Tester",
        email: "mobile-tester@agpt.co",
        password: "a-long-test-password", // pragma: allowlist secret
      },
      asResponse: true,
    });
    cookies = cookieHeader(response);
    userID = (await response.json()).user.id;
  });

  function post(path: string, body: Record<string, string>, cookie = "") {
    return auth.handler(
      new Request(`${ORIGIN}/api/auth/mobile/${path}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Origin: ORIGIN,
          ...(cookie ? { Cookie: cookie } : {}),
        },
        body: JSON.stringify(
          path === "authorize" ? { expected_user_id: userID, ...body } : body,
        ),
      }),
    );
  }

  async function issueCode() {
    const response = await post(
      "authorize",
      { code_challenge: CHALLENGE, state: STATE },
      cookies,
    );
    expect(response.status).toBe(200);
    return new URL((await response.json()).url).searchParams.get("code")!;
  }

  it("does not turn admin impersonation into a lasting user session", async () => {
    const context = await auth.$context;
    const source = await auth.api.getSession({
      headers: new Headers({ Cookie: cookies }),
    });
    await context.adapter.update({
      model: "session",
      where: [{ field: "id", value: source!.session.id }],
      update: { impersonatedBy: "admin-user-id" },
    });
    const response = await post(
      "authorize",
      { code_challenge: CHALLENGE, state: STATE },
      cookies,
    );
    expect(response.status).toBe(403);
  });

  it("honors database session creation policy during exchange", async () => {
    const code = await issueCode();
    allowNewSession = false;
    const response = await post("exchange", { code, code_verifier: VERIFIER });
    expect(response.status).toBe(401);
    expect(response.headers.has("set-cookie")).toBe(false);
  });

  it("requires the source browser session to remain unexpired", async () => {
    const code = await issueCode();
    const context = await auth.$context;
    const source = await auth.api.getSession({
      headers: new Headers({ Cookie: cookies }),
    });
    await context.adapter.update({
      model: "session",
      where: [{ field: "id", value: source!.session.id }],
      update: { expiresAt: new Date(Date.now() - 1000) },
    });
    const response = await post("exchange", { code, code_verifier: VERIFIER });
    expect(response.status).toBe(401);
    expect(response.headers.has("set-cookie")).toBe(false);
  });

  it("permits an expired ban through the normal session policy", async () => {
    const code = await issueCode();
    const context = await auth.$context;
    await context.internalAdapter.updateUser(userID, {
      banned: true,
      banExpires: new Date(Date.now() - 1000),
    });
    const response = await post("exchange", { code, code_verifier: VERIFIER });
    expect(response.status).toBe(200);
  });

  it("rejects cookie-free exchange with a missing or foreign origin", async () => {
    const code = await issueCode();
    for (const origin of ["", "https://attacker.example"]) {
      const response = await auth.handler(
        new Request(`${ORIGIN}/api/auth/mobile/exchange`, {
          method: "POST",
          headers: { "Content-Type": "application/json", Origin: origin },
          body: JSON.stringify({ code, code_verifier: VERIFIER }),
        }),
      );
      expect(response.status).toBe(403);
      expect(response.headers.has("set-cookie")).toBe(false);
    }
    expect(
      (await post("exchange", { code, code_verifier: VERIFIER })).status,
    ).toBe(200);
  });

  it.each([true, false])(
    "applies current email verification requirement %s to existing unverified sessions",
    async (required) => {
      const code = await issueCode();
      const context = await auth.$context;
      context.options.emailAndPassword = {
        ...context.options.emailAndPassword,
        enabled: true,
        requireEmailVerification: required,
      };
      const consent = await post(
        "authorize",
        { code_challenge: CHALLENGE, state: STATE },
        cookies,
      );
      expect(consent.status).toBe(required ? 403 : 200);
      const exchange = await post("exchange", {
        code,
        code_verifier: VERIFIER,
      });
      expect(exchange.status).toBe(required ? 403 : 200);
      if (required) expect(exchange.headers.has("set-cookie")).toBe(false);
    },
  );

  it("revokes the new app session if its browser session disappears during creation", async () => {
    const code = await issueCode();
    const context = await auth.$context;
    const source = await auth.api.getSession({
      headers: new Headers({ Cookie: cookies }),
    });
    beforeNewSession = async () => {
      await context.internalAdapter.deleteSession(source!.session.token);
    };
    const exchange = await post("exchange", { code, code_verifier: VERIFIER });
    expect(exchange.status).toBe(401);
    expect(exchange.headers.has("set-cookie")).toBe(false);
    expect(await context.adapter.findMany({ model: "session" })).toHaveLength(
      0,
    );
  });

  it("does not authorize a different account than the consent page displayed", async () => {
    const other = await auth.api.signUpEmail({
      body: {
        name: "Other",
        email: "other@agpt.co",
        password: "a-long-test-password", // pragma: allowlist secret
      },
      asResponse: true,
    });
    const response = await post(
      "authorize",
      { code_challenge: CHALLENGE, state: STATE },
      cookieHeader(other),
    );
    expect(response.status).toBe(403);
    const context = await auth.$context;
    expect(
      await context.adapter.findMany({ model: "verification" }),
    ).toHaveLength(0);
  });
});
