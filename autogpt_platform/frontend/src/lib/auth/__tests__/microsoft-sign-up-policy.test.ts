import { APIError } from "better-auth/api";
import { describe, expect, it } from "vitest";
import { assertMicrosoftSignUpAllowed } from "../microsoft-sign-up-policy";

const microsoftCallback = {
  path: "/callback/:id",
  params: { id: "microsoft" },
};

// The code the OAuth callback redirects with, or null when nothing is refused.
function refusalCode(run: () => void) {
  try {
    run();
  } catch (error) {
    return error instanceof APIError ? error.body?.code : error;
  }
  return null;
}

describe("assertMicrosoftSignUpAllowed", () => {
  it("refuses a Microsoft sign-up whose address Microsoft didn't confirm", () => {
    expect(
      refusalCode(() =>
        assertMicrosoftSignUpAllowed(
          { email: "pat@example.com", emailVerified: false },
          microsoftCallback,
        ),
      ),
    ).toBe("unconfirmed_email");
    expect(
      refusalCode(() =>
        assertMicrosoftSignUpAllowed(
          { email: "pat@example.com", emailVerified: true },
          microsoftCallback,
        ),
      ),
    ).toBeNull();
  });

  it("refuses one through an ID token too", () => {
    expect(
      refusalCode(() =>
        assertMicrosoftSignUpAllowed(
          { email: "pat@example.com", emailVerified: false },
          { path: "/sign-in/social", body: { provider: "microsoft" } },
        ),
      ),
    ).toBe("unconfirmed_email");
  });

  it("keeps an AutoGPT team address to Google, even one Microsoft confirmed", () => {
    expect(
      refusalCode(() =>
        assertMicrosoftSignUpAllowed(
          { email: "Pat@AGPT.co", emailVerified: true },
          microsoftCallback,
        ),
      ),
    ).toBe("team_email_requires_google");
    // Only the exact domain is the team's: QA accounts keep Microsoft.
    expect(
      refusalCode(() =>
        assertMicrosoftSignUpAllowed(
          { email: "qa@previews.agpt.co", emailVerified: true },
          microsoftCallback,
        ),
      ),
    ).toBeNull();
  });

  it("leaves other sign-up methods alone", () => {
    for (const ctx of [
      { path: "/callback/:id", params: { id: "google" } },
      { path: "/sign-up/email" },
      null,
      { path: "/sign-in/social", body: { provider: "google" } },
    ]) {
      expect(
        refusalCode(() =>
          assertMicrosoftSignUpAllowed(
            { email: "pat@agpt.co", emailVerified: false },
            ctx,
          ),
        ),
      ).toBeNull();
    }
  });
});
