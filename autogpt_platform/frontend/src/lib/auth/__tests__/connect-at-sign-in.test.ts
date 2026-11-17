import { afterEach, describe, expect, it, vi } from "vitest";
import { GOOGLE_ACCESS_SCOPES } from "@/lib/google-access";
import {
  assertSignInScopes,
  connectAtSignInScopes,
  withoutProviderTokens,
} from "../connect-at-sign-in";

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("connectAtSignInScopes", () => {
  it("asks for nothing more unless the provider is turned on", () => {
    vi.stubEnv("AUTH_CONNECT_AT_SIGN_IN", "");
    expect(connectAtSignInScopes("google")).toBeNull();

    vi.stubEnv("AUTH_CONNECT_AT_SIGN_IN", " microsoft ");
    expect(connectAtSignInScopes("google")).toBeNull();
    expect(connectAtSignInScopes("microsoft")).toEqual(
      expect.arrayContaining(["offline_access", "Mail.ReadWrite"]),
    );
  });

  it("asks Google for exactly what the Google step offers", () => {
    vi.stubEnv("AUTH_CONNECT_AT_SIGN_IN", "google");
    expect(connectAtSignInScopes("google")).toEqual(GOOGLE_ACCESS_SCOPES);
  });

  it("only knows Google and Microsoft", () => {
    vi.stubEnv("AUTH_CONNECT_AT_SIGN_IN", "github,google");
    expect(connectAtSignInScopes("github")).toBeNull();
    expect(connectAtSignInScopes("__proto__")).toBeNull();
  });
});

describe("assertSignInScopes", () => {
  function signIn(provider: string, scopes?: unknown) {
    return () =>
      assertSignInScopes("/sign-in/social", {
        provider,
        callbackURL: "/auth/callback",
        ...(scopes === undefined ? {} : { scopes }),
      });
  }

  it("lets a sign-in ask for the data scopes while connecting is on", () => {
    vi.stubEnv("AUTH_CONNECT_AT_SIGN_IN", "google");
    expect(signIn("google", [...GOOGLE_ACCESS_SCOPES])).not.toThrow();
    expect(signIn("google", [])).not.toThrow();
    expect(signIn("google")).not.toThrow();
  });

  it("refuses any other scope for Google or Microsoft", () => {
    vi.stubEnv("AUTH_CONNECT_AT_SIGN_IN", "google,microsoft");
    expect(signIn("google", ["https://mail.google.com/"])).toThrow(
      "can't ask for those permissions",
    );
    expect(signIn("microsoft", ["Mail.ReadWrite", "Sites.Read.All"])).toThrow();
    expect(signIn("google", "gmail.readonly")).toThrow();
  });

  it("refuses the data scopes while connecting is off", () => {
    vi.stubEnv("AUTH_CONNECT_AT_SIGN_IN", "");
    expect(signIn("google", [...GOOGLE_ACCESS_SCOPES])).toThrow();
  });

  it("leaves other providers and endpoints alone", () => {
    vi.stubEnv("AUTH_CONNECT_AT_SIGN_IN", "");
    expect(signIn("github", ["repo"])).not.toThrow();
    expect(() =>
      assertSignInScopes("/sign-in/email", {
        provider: "google",
        scopes: ["https://mail.google.com/"],
      }),
    ).not.toThrow();
  });

  it("checks linking a provider the same way", () => {
    vi.stubEnv("AUTH_CONNECT_AT_SIGN_IN", "google");
    expect(() =>
      assertSignInScopes("/link-social", {
        provider: "google",
        scopes: ["https://mail.google.com/"],
      }),
    ).toThrow();
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
