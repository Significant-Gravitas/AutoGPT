import { describe, expect, it } from "vitest";
import { GET } from "../route";

describe("Microsoft callback recovery", () => {
  it("returns a safe, state-bound setup error instead of losing Microsoft's reason", async () => {
    const response = await GET(
      new Request(
        "https://app.example/auth/integrations/oauth_callback?state=flow-state&error=invalid_resource&error_description=AADSTS650052%3A%20private-upstream-text",
      ),
    );
    const html = await response.text();
    expect(html).toContain("flow-state");
    expect(html).toContain("AADSTS650052");
    expect(html).toContain("enable Work IQ");
    expect(html).not.toContain("private-upstream-text");
    expect(response.headers.get("cache-control")).toBe("no-store");
  });

  it("confirms an organization-wide approval, which has no code to exchange", async () => {
    const response = await GET(
      new Request(
        "https://app.example/auth/integrations/oauth_callback?admin_consent=True&tenant=company",
      ),
    );
    const html = await response.text();
    expect(html).toContain("Approved for your organization");
    expect(html).not.toContain("oauth_popup_result");
  });

  it("says why an organization-wide approval didn't go through", async () => {
    // Microsoft marks a refused approval admin_consent=True as well.
    const response = await GET(
      new Request(
        "https://app.example/auth/integrations/oauth_callback?admin_consent=True&tenant=company&error=access_denied&error_description=AADSTS65004%3A%20private-upstream-text",
      ),
    );
    const html = await response.text();
    expect(html).not.toContain("Approved for your organization");
    expect(html).toContain("AADSTS65004");
    expect(html).toContain("Microsoft access was declined");
    expect(html).not.toContain("private-upstream-text");
    expect(html).not.toContain("oauth_popup_result");
  });

  it("never puts the callback's own text on the approval page", async () => {
    const hostile = encodeURIComponent('<img src=x onerror="alert(1)">');
    const response = await GET(
      new Request(
        `https://app.example/auth/integrations/oauth_callback?admin_consent=True&error=${hostile}&error_description=AADSTS65004${hostile}`,
      ),
    );
    const html = await response.text();
    expect(html).toContain("AADSTS65004");
    expect(html).not.toContain("<img");
    expect(html).not.toContain("onerror");
  });
});

describe("Sign-in started by the mobile app", () => {
  function callback(query: string) {
    return GET(
      new Request(
        `https://app.example/auth/integrations/oauth_callback?${query}`,
      ),
    );
  }

  it("sends the code and state back to the app", async () => {
    const response = await callback("code=abc&state=native.xyz");

    expect(response.status).toBe(302);
    expect(response.headers.get("location")).toBe(
      "autogpt-app://integrations/oauth?code=abc&state=native.xyz",
    );
    expect(response.headers.get("cache-control")).toBe("no-store");
    expect(response.headers.get("referrer-policy")).toBe("no-referrer");
  });

  it("sends the provider's error back to the app", async () => {
    const response = await callback(
      "state=native.xyz&error=access_denied&error_description=User%20said%20no",
    );

    expect(response.headers.get("location")).toBe(
      "autogpt-app://integrations/oauth?state=native.xyz&error=access_denied&error_description=User+said+no",
    );
  });

  it("says the code is missing when the provider sends neither", async () => {
    const response = await callback("state=native.xyz");

    expect(response.headers.get("location")).toBe(
      "autogpt-app://integrations/oauth?state=native.xyz&error=missing_code",
    );
  });

  it("keeps the popup page for a web sign-in", async () => {
    const response = await callback("code=abc&state=xyz");

    expect(response.status).toBe(200);
    expect(await response.text()).toContain("oauth_popup_result");
  });
});
