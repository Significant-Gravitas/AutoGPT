// @vitest-environment node
import { delay, http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { server } from "@/mocks/mock-server";
import { GOOGLE_ACCESS_SCOPES } from "@/lib/google-access";
import { withoutProviderTokens } from "../connect-at-sign-in";

// The sign-in's OAuth state exists only inside Better Auth's request
// (sign-in-connection-flow.test.ts runs the real one).
const oauthState = vi.hoisted(() => ({ addOAuthServerContext: vi.fn() }));
vi.mock("better-auth/api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("better-auth/api")>()),
  addOAuthServerContext: oauthState.addOAuthServerContext,
}));

// The module remembers the backend's answer, so each test loads it afresh.
async function load() {
  vi.resetModules();
  return import("../connect-at-sign-in");
}

let lookups = 0;

// What the backend says its Google and Microsoft connections use.
function backendClients(answer: Record<string, string | null> | "down") {
  server.use(
    http.get("*/api/auth/sign-in-connection", () => {
      lookups += 1;
      return answer === "down"
        ? HttpResponse.json({ detail: "unavailable" }, { status: 503 })
        : HttpResponse.json(answer);
    }),
  );
}

beforeEach(() => {
  lookups = 0;
  oauthState.addOAuthServerContext.mockClear();
  vi.stubEnv("AUTH_GOOGLE_CLIENT_ID", "google-client");
  vi.stubEnv("AUTH_MICROSOFT_CLIENT_ID", "microsoft-client");
  backendClients({ google: "google-client", microsoft: "microsoft-client" });
});

afterEach(() => {
  vi.unstubAllEnvs();
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe("connectAtSignInScopes", () => {
  it("asks for the data scopes when sign-in uses the connections' client", async () => {
    const { connectAtSignInScopes } = await load();

    expect(await connectAtSignInScopes("google")).toEqual(GOOGLE_ACCESS_SCOPES);
    expect(await connectAtSignInScopes("microsoft")).toEqual(
      expect.arrayContaining(["offline_access", "Mail.ReadWrite"]),
    );
  });

  it("asks only who the person is when sign-in has a client of its own", async () => {
    backendClients({ google: "connections-client", microsoft: null });
    const { connectAtSignInScopes } = await load();

    expect(await connectAtSignInScopes("google")).toBeNull();
    expect(await connectAtSignInScopes("microsoft")).toBeNull();
  });

  it("asks only who the person is without a sign-in client to compare", async () => {
    vi.stubEnv("AUTH_GOOGLE_CLIENT_ID", "");
    const { connectAtSignInScopes } = await load();

    expect(await connectAtSignInScopes("google")).toBeNull();
    expect(lookups).toBe(0);
  });

  it("only knows Google and Microsoft", async () => {
    const { connectAtSignInScopes } = await load();

    expect(await connectAtSignInScopes("github")).toBeNull();
    expect(await connectAtSignInScopes("__proto__")).toBeNull();
    expect(lookups).toBe(0);
  });

  it("asks the backend once for many sign-ins", async () => {
    const { connectAtSignInScopes } = await load();

    await Promise.all([
      connectAtSignInScopes("google"),
      connectAtSignInScopes("microsoft"),
    ]);
    await connectAtSignInScopes("google");

    expect(lookups).toBe(1);
  });

  it("asks the backend again once its answer is a few minutes old", async () => {
    vi.useFakeTimers({ toFake: ["Date"] });
    const { connectAtSignInScopes } = await load();
    await connectAtSignInScopes("google");
    // A redeploy gave Google's connections another client.
    backendClients({ google: "connections-client", microsoft: null });

    vi.setSystemTime(Date.now() + 4 * 60_000);
    expect(await connectAtSignInScopes("google")).toEqual(GOOGLE_ACCESS_SCOPES);
    vi.setSystemTime(Date.now() + 2 * 60_000);
    expect(await connectAtSignInScopes("google")).toBeNull();
    expect(lookups).toBe(2);
  });

  it("asks only who the person is while the backend can't say, and asks again soon", async () => {
    vi.useFakeTimers({ toFake: ["Date"] });
    const warnings = vi.spyOn(console, "warn").mockImplementation(() => {});
    backendClients("down");
    const { connectAtSignInScopes } = await load();

    expect(await connectAtSignInScopes("google")).toBeNull();
    expect(warnings).toHaveBeenCalledWith(
      "Couldn't ask the backend which clients its connections use",
      { error: "status 503" },
    );

    backendClients({ google: "google-client", microsoft: "microsoft-client" });
    expect(await connectAtSignInScopes("google")).toBeNull();
    vi.setSystemTime(Date.now() + 31_000);
    expect(await connectAtSignInScopes("google")).toEqual(GOOGLE_ACCESS_SCOPES);
    expect(lookups).toBe(2);
  });

  it("gives up on a stalled backend after two seconds", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    // The two seconds have already passed.
    const timeouts = vi
      .spyOn(AbortSignal, "timeout")
      .mockImplementation(() =>
        AbortSignal.abort(new DOMException("timed out", "TimeoutError")),
      );
    server.use(
      http.get("*/api/auth/sign-in-connection", async () => {
        await delay("infinite");
        return HttpResponse.json({});
      }),
    );
    const { connectAtSignInScopes } = await load();

    // A backend that never answers: the lookup ends at its timeout instead.
    expect(await connectAtSignInScopes("google")).toBeNull();
    expect(timeouts).toHaveBeenCalledWith(2_000);
  });
});

describe("assertSignInScopes", () => {
  async function signInCheck() {
    const { assertSignInScopes } = await load();
    return (provider: string, scopes?: unknown) =>
      assertSignInScopes("/sign-in/social", {
        provider,
        callbackURL: "/auth/callback",
        ...(scopes === undefined ? {} : { scopes }),
      });
  }

  it("lets a sign-in that can connect ask for the data scopes", async () => {
    const signIn = await signInCheck();

    await expect(
      signIn("google", [...GOOGLE_ACCESS_SCOPES]),
    ).resolves.toBeUndefined();
    await expect(signIn("google", [])).resolves.toBeUndefined();
    await expect(signIn("google")).resolves.toBeUndefined();
  });

  it("notes in the OAuth state only a sign-in that asks for data scopes", async () => {
    const signIn = await signInCheck();

    await signIn("google");
    await signIn("google", []);
    await signIn("microsoft", ["offline_access", "User.Read"]);
    expect(oauthState.addOAuthServerContext).not.toHaveBeenCalled();

    await signIn("google", [...GOOGLE_ACCESS_SCOPES]);
    expect(oauthState.addOAuthServerContext).toHaveBeenCalledWith({
      connectAtSignIn: true,
    });
  });

  it("refuses any other scope for Google or Microsoft", async () => {
    const signIn = await signInCheck();

    await expect(
      signIn("google", ["https://mail.google.com/"]),
    ).rejects.toThrow("can't ask for those permissions");
    await expect(
      signIn("microsoft", ["Mail.ReadWrite", "Sites.Read.All"]),
    ).rejects.toThrow();
    await expect(signIn("google", "gmail.readonly")).rejects.toThrow();
  });

  it("refuses the data scopes when sign-in has a client of its own", async () => {
    backendClients({ google: "connections-client", microsoft: null });
    const signIn = await signInCheck();

    await expect(signIn("google", [...GOOGLE_ACCESS_SCOPES])).rejects.toThrow();
  });

  it("leaves other providers and endpoints alone", async () => {
    const { assertSignInScopes } = await load();

    await expect(
      assertSignInScopes("/sign-in/social", {
        provider: "github",
        scopes: ["repo"],
      }),
    ).resolves.toBeUndefined();
    await expect(
      assertSignInScopes("/sign-in/email", {
        provider: "google",
        scopes: ["https://mail.google.com/"],
      }),
    ).resolves.toBeUndefined();
    expect(lookups).toBe(0);
  });

  it("checks linking a provider the same way", async () => {
    const { assertSignInScopes } = await load();

    await expect(
      assertSignInScopes("/link-social", {
        provider: "google",
        scopes: ["https://mail.google.com/"],
      }),
    ).rejects.toThrow();
  });
});

describe("withoutProviderTokens", () => {
  it("keeps Google's and Microsoft's tokens out of the auth table", () => {
    for (const providerId of ["google", "microsoft"]) {
      expect(
        withoutProviderTokens({
          providerId,
          accountId: "account",
          accessToken: "access",
          refreshToken: "refresh",
          idToken: "id",
        }),
      ).toEqual({
        providerId,
        accountId: "account",
        accessToken: null,
        refreshToken: null,
        idToken: null,
      });
    }
  });

  it("leaves other providers' rows as they are", () => {
    const github = { providerId: "github", accessToken: "access" };
    expect(withoutProviderTokens(github)).toBe(github);
    const unnamed: { providerId?: string; accessToken: string } = {
      accessToken: "access",
    };
    expect(withoutProviderTokens(unnamed)).toBe(unnamed);
  });
});
