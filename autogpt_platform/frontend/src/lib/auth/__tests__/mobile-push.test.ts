// @vitest-environment node
import { betterAuth } from "better-auth";
import { memoryAdapter } from "better-auth/adapters/memory";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { mobilePush } from "../mobile-push";
import { cookieHeader, ORIGIN } from "./mobile-auth-fixtures";

describe("session-bound native push registration", () => {
  const store = { save: vi.fn(), remove: vi.fn() };
  let auth: ReturnType<typeof makeAuth>;
  let cookies: string;
  let userID: string;
  const body = {
    provider: "apns",
    token: "ab".repeat(32),
    environment: "sandbox",
    binding_id: "1ca1b47e-4cb0-4f0f-9693-d44b40b6d05f",
  };

  function makeAuth() {
    return betterAuth({
      baseURL: ORIGIN,
      secret: "mobile-push-test-secret-at-least-32-characters", // pragma: allowlist secret
      database: memoryAdapter({
        user: [],
        session: [],
        account: [],
        verification: [],
      }),
      emailAndPassword: { enabled: true },
      rateLimit: { enabled: false },
      plugins: [mobilePush(store)],
    });
  }

  beforeEach(async () => {
    vi.clearAllMocks();
    auth = makeAuth();
    const signup = await auth.api.signUpEmail({
      body: {
        name: "Tester",
        email: "push@example.com",
        password: "test-only-long-password", // pragma: allowlist secret
      },
      asResponse: true,
    });
    cookies = cookieHeader(signup);
    userID = (await signup.json()).user.id;
  });

  function request(
    data: unknown = body,
    origin = ORIGIN,
    cookie = cookies,
    path = "push",
  ) {
    return auth.handler(
      new Request(`${ORIGIN}/api/auth/mobile/${path}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Origin: origin,
          Cookie: cookie,
        },
        body: JSON.stringify(
          path === "push" && typeof data === "object"
            ? { expected_user_id: userID, ...data }
            : data,
        ),
      }),
    );
  }

  it("requires a live session and the server's own origin", async () => {
    expect((await request(body, ORIGIN, "")).status).toBe(401);
    expect((await request(body, "https://other.example")).status).toBe(403);
    expect(store.save).not.toHaveBeenCalled();
  });

  it("binds to the authoritative session, not a caller-supplied user", async () => {
    expect((await request()).status).toBe(200);
    const session = await auth.api.getSession({
      headers: new Headers({ Cookie: cookies }),
    });
    expect(store.save).toHaveBeenCalledWith(
      expect.objectContaining({
        sessionID: session!.session.id,
        origin: ORIGIN,
        binding_id: body.binding_id,
      }),
    );
    expect((await request({ ...body, user_id: "another-user" })).status).toBe(
      400,
    );
    expect(
      (await request({ ...body, expected_user_id: "another-user" })).status,
    ).toBe(403);
  });

  it("does not accept arbitrary endpoints or malformed device tokens", async () => {
    for (const token of [
      "",
      "https://localhost/",
      "../another/path",
      "a".repeat(5000),
    ]) {
      expect((await request({ ...body, token })).status).toBe(400);
    }
    expect(store.save).not.toHaveBeenCalled();
  });

  it("removes only the current session's registration and rejects signed-out cookies", async () => {
    expect((await request({}, ORIGIN, cookies, "push/remove")).status).toBe(
      200,
    );
    expect(store.remove).toHaveBeenCalledOnce();
    await auth.api.signOut({ headers: new Headers({ Cookie: cookies }) });
    expect((await request()).status).toBe(401);
    expect(store.save).not.toHaveBeenCalled();
  });
});
