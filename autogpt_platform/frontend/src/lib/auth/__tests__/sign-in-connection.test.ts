// @vitest-environment node
import { tryGetCurrentAuthEndpointContext } from "@better-auth/core/context";
import type { OAuth2Tokens } from "@better-auth/core/oauth2";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { server } from "@/mocks/mock-server";
import { GOOGLE_ACCESS_SCOPES } from "@/lib/google-access";
import { askedToConnect } from "../connect-at-sign-in";
import { connectSignIn, keepingSignInTokens } from "../sign-in-connection";

vi.mock("@better-auth/core/context", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@better-auth/core/context")>()),
  tryGetCurrentAuthEndpointContext: vi.fn(),
}));
// What the sign-in's OAuth state says (sign-in-connection-flow.test.ts runs
// the real one).
vi.mock("../connect-at-sign-in", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../connect-at-sign-in")>()),
  askedToConnect: vi.fn(async () => true),
}));
vi.mock("../service-token", () => ({
  mintServiceToken: vi.fn(async () => "service-token"),
}));

const TOKENS: OAuth2Tokens = {
  accessToken: "access",
  refreshToken: "refresh",
  scopes: ["openid", ...GOOGLE_ACCESS_SCOPES],
  idToken: "id",
};

let handoffs: unknown[] = [];

beforeEach(() => {
  // Sign-in uses the client the backend's Google connections use.
  vi.stubEnv("AUTH_GOOGLE_CLIENT_ID", "integration-client");
  handoffs = [];
  server.use(
    http.get("*/api/auth/sign-in-connection", () =>
      HttpResponse.json({ google: "integration-client", microsoft: null }),
    ),
    http.post("*/api/auth/sign-in-connection", async ({ request }) => {
      handoffs.push(await request.json());
      return HttpResponse.json({ result: "created" });
    }),
  );
});

afterEach(() => {
  vi.unstubAllEnvs();
});

async function signInAt(path: string, tokens = TOKENS) {
  const ctx = { path };
  vi.mocked(tryGetCurrentAuthEndpointContext).mockReturnValue(ctx as never);
  const getUserInfo = vi.fn(async () => ({ user: { id: "google-id" } }));
  await keepingSignInTokens("google", getUserInfo)(tokens);
  await connectSignIn(ctx, "user-1");
  return getUserInfo;
}

describe("keepingSignInTokens", () => {
  it("connects the tokens of the provider's redirect", async () => {
    const getUserInfo = await signInAt("/callback/:id");

    expect(getUserInfo).toHaveBeenCalledWith(TOKENS);
    expect(handoffs).toEqual([
      expect.objectContaining({ provider: "google", access_token: "access" }),
    ]);
  });

  it("never connects tokens the browser sent with an ID token", async () => {
    // /sign-in/social takes an ID token plus any access token the caller
    // chooses; none of it came from the provider's redirect.
    await signInAt("/sign-in/social");

    expect(handoffs).toEqual([]);
  });

  it("connects nothing for a sign-in that didn't ask to connect", async () => {
    // Its token still carries the data scopes an earlier sign-in granted.
    vi.mocked(askedToConnect).mockResolvedValueOnce(false);

    await signInAt("/callback/:id");

    expect(handoffs).toEqual([]);
  });

  it("connects nothing when every data permission was declined", async () => {
    await signInAt("/callback/:id", { ...TOKENS, scopes: ["openid"] });

    expect(handoffs).toEqual([]);
  });

  it("connects nothing when sign-in has a client of its own", async () => {
    // Google may still return data scopes an earlier sign-in granted, but
    // the backend couldn't renew this client's tokens.
    vi.stubEnv("AUTH_GOOGLE_CLIENT_ID", "login-client");

    await signInAt("/callback/:id");

    expect(handoffs).toEqual([]);
  });

  it("hands each sign-in's tokens over once", async () => {
    const ctx = { path: "/callback/:id" };
    vi.mocked(tryGetCurrentAuthEndpointContext).mockReturnValue(ctx as never);
    await keepingSignInTokens("google", async () => null)(TOKENS);

    await connectSignIn(ctx, "user-1");
    await connectSignIn(ctx, "user-1");

    expect(handoffs).toHaveLength(1);
  });
});
