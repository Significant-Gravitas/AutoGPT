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
  it("keeps an AutoGPT team address to Google", () => {
    expect(
      refusalCode(() =>
        assertMicrosoftSignUpAllowed(
          { email: "Pat@AGPT.co" },
          microsoftCallback,
        ),
      ),
    ).toBe("team_email_requires_google");
    // Only the exact domain is the team's: QA accounts keep Microsoft.
    expect(
      refusalCode(() =>
        assertMicrosoftSignUpAllowed(
          { email: "qa@previews.agpt.co" },
          microsoftCallback,
        ),
      ),
    ).toBeNull();
  });

  it("refuses one through an ID token too", () => {
    expect(
      refusalCode(() =>
        assertMicrosoftSignUpAllowed(
          { email: "pat@agpt.co" },
          { path: "/sign-in/social", body: { provider: "microsoft" } },
        ),
      ),
    ).toBe("team_email_requires_google");
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
          assertMicrosoftSignUpAllowed({ email: "pat@agpt.co" }, ctx),
        ),
      ).toBeNull();
    }
  });
});
