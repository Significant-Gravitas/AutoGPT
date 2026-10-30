import { describe, expect, it } from "vitest";
import { assertMicrosoftSignUpAllowed } from "../microsoft-sign-up-policy";

const microsoftCallback = {
  path: "/callback/:id",
  params: { id: "microsoft" },
};

describe("assertMicrosoftSignUpAllowed", () => {
  it("refuses a Microsoft sign-up whose address Microsoft didn't confirm", () => {
    expect(() =>
      assertMicrosoftSignUpAllowed(
        { email: "pat@example.com", emailVerified: false },
        microsoftCallback,
      ),
    ).toThrow("unconfirmed_email");
    expect(() =>
      assertMicrosoftSignUpAllowed(
        { email: "pat@example.com", emailVerified: true },
        microsoftCallback,
      ),
    ).not.toThrow();
  });

  it("refuses one through an ID token too", () => {
    expect(() =>
      assertMicrosoftSignUpAllowed(
        { email: "pat@example.com", emailVerified: false },
        { path: "/sign-in/social", body: { provider: "microsoft" } },
      ),
    ).toThrow("unconfirmed_email");
  });

  it("keeps an AutoGPT team address to Google, even one Microsoft confirmed", () => {
    expect(() =>
      assertMicrosoftSignUpAllowed(
        { email: "Pat@AGPT.co", emailVerified: true },
        microsoftCallback,
      ),
    ).toThrow("team_email_requires_google");
    // Only the exact domain is the team's: QA accounts keep Microsoft.
    expect(() =>
      assertMicrosoftSignUpAllowed(
        { email: "qa@previews.agpt.co", emailVerified: true },
        microsoftCallback,
      ),
    ).not.toThrow();
  });

  it("leaves other sign-up methods alone", () => {
    for (const ctx of [
      { path: "/callback/:id", params: { id: "google" } },
      { path: "/sign-up/email" },
      null,
      { path: "/sign-in/social", body: { provider: "google" } },
    ]) {
      expect(() =>
        assertMicrosoftSignUpAllowed(
          { email: "pat@agpt.co", emailVerified: false },
          ctx,
        ),
      ).not.toThrow();
    }
  });
});
