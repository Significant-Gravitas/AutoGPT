import { describe, expect, test } from "vitest";

import {
  buildSetupWizardRedirect,
  isHttpRedirectUrl,
  isRegisteredRedirectUri,
  registeredRedirectUrisOf,
} from "../redirect-safety";

describe("isHttpRedirectUrl", () => {
  test("accepts http and https URLs", () => {
    expect(isHttpRedirectUrl("https://client.example/callback")).toBe(true);
    expect(isHttpRedirectUrl("http://localhost:3000/callback")).toBe(true);
  });

  // `window.location.href = "javascript:..."` runs in the platform origin with
  // the signed-in user's session, which is the worse half of this bug.
  test("rejects schemes the browser would act on", () => {
    expect(isHttpRedirectUrl("javascript:alert(1)//")).toBe(false);
    expect(isHttpRedirectUrl("JavaScript:alert(1)//")).toBe(false);
    expect(isHttpRedirectUrl("data:text/html,<script>alert(1)</script>")).toBe(
      false,
    );
    expect(isHttpRedirectUrl("blob:https://client.example/uuid")).toBe(false);
    expect(isHttpRedirectUrl("myapp://callback")).toBe(false);
    expect(isHttpRedirectUrl("mailto:victim@example.com")).toBe(false);
  });

  test("rejects values that are not URLs", () => {
    expect(isHttpRedirectUrl(null)).toBe(false);
    expect(isHttpRedirectUrl(undefined)).toBe(false);
    expect(isHttpRedirectUrl("")).toBe(false);
    expect(isHttpRedirectUrl("not a url")).toBe(false);
    expect(isHttpRedirectUrl("//evil.example")).toBe(false);
  });
});

describe("isRegisteredRedirectUri", () => {
  const registered = [
    "https://client.example/callback",
    "http://localhost:3000/oauth/callback",
  ];

  test("accepts a registered redirect_uri", () => {
    expect(
      isRegisteredRedirectUri("https://client.example/callback", registered),
    ).toBe(true);
    expect(
      isRegisteredRedirectUri(
        "http://localhost:3000/oauth/callback",
        registered,
      ),
    ).toBe(true);
  });

  test("rejects a redirect_uri the app did not register", () => {
    expect(
      isRegisteredRedirectUri("https://evil.example/phish", registered),
    ).toBe(false);
    // Same origin, different path: a registered callback is an exact match.
    expect(
      isRegisteredRedirectUri("https://client.example/other", registered),
    ).toBe(false);
    // A registered URI is not a prefix of another host.
    expect(
      isRegisteredRedirectUri(
        "https://client.example.evil.example/callback",
        registered,
      ),
    ).toBe(false);
  });

  // Without a client_id the app is unknown, so only the scheme rule applies.
  // Callers should send client_id: an unidentified caller can still be sent to
  // any https host, which is the phishing case.
  test("falls back to allowing anything when no app is known", () => {
    expect(isRegisteredRedirectUri("https://any.example/cb", [])).toBe(true);
    expect(isRegisteredRedirectUri("https://any.example/cb", null)).toBe(true);
  });
});

describe("the wizard's two rules together", () => {
  const registered = ["https://client.example/callback"];

  const mayNavigateTo = (
    redirectUri: string | null | undefined,
    uris: readonly string[],
  ) =>
    isHttpRedirectUrl(redirectUri) &&
    isRegisteredRedirectUri(redirectUri, uris);

  test("accepts a registered https callback", () => {
    expect(mayNavigateTo("https://client.example/callback", registered)).toBe(
      true,
    );
  });

  test("rejects a dangerous scheme even when it is registered", () => {
    expect(
      mayNavigateTo("javascript:alert(1)//", ["javascript:alert(1)//"]),
    ).toBe(false);
    expect(mayNavigateTo("data:text/html,x", ["data:text/html,x"])).toBe(false);
  });

  test("rejects an unregistered https callback", () => {
    expect(mayNavigateTo("https://evil.example/phish", registered)).toBe(false);
  });
});

describe("buildSetupWizardRedirect", () => {
  test("appends the result parameters", () => {
    expect(
      buildSetupWizardRedirect("https://client.example/callback", {
        success: "true",
        state: "abc",
      }),
    ).toBe("https://client.example/callback?success=true&state=abc");
  });

  test("keeps a query string and fragment the target already had", () => {
    expect(
      buildSetupWizardRedirect(
        "https://client.example/callback?tenant=1#frag",
        { error: "user_cancelled" },
      ),
    ).toBe(
      "https://client.example/callback?tenant=1&error=user_cancelled#frag",
    );
  });

  test("encodes parameter values instead of pasting them into the URL", () => {
    const built = buildSetupWizardRedirect("https://client.example/cb", {
      error_description: "User cancelled & left? #1",
    });
    expect(built).toContain(
      "error_description=User+cancelled+%26+left%3F+%231",
    );
    expect(built).not.toContain("&left");
  });
});

describe("registeredRedirectUrisOf", () => {
  // GET /oauth/app/{client_id} does not return redirect_uris yet (#15057 adds
  // it), so the list is read defensively and the scheme rule applies alone
  // until the backend exposes it.
  test("returns an empty list when the field is absent", () => {
    expect(registeredRedirectUrisOf({ name: "App", scopes: [] })).toEqual([]);
    expect(registeredRedirectUrisOf(undefined)).toEqual([]);
    expect(registeredRedirectUrisOf(null)).toEqual([]);
    expect(registeredRedirectUrisOf({ redirect_uris: "nope" })).toEqual([]);
  });

  test("keeps only the string entries of a present list", () => {
    expect(
      registeredRedirectUrisOf({
        redirect_uris: [
          "https://a.example/cb",
          7,
          null,
          "https://b.example/cb",
        ],
      }),
    ).toEqual(["https://a.example/cb", "https://b.example/cb"]);
  });
});
