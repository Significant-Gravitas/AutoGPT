import { describe, expect, it } from "vitest";
import { assertMicrosoftSignUpConfirmed } from "../microsoft-sign-up-policy";

const microsoftCallback = {
  path: "/callback/:id",
  params: { id: "microsoft" },
};

describe("assertMicrosoftSignUpConfirmed", () => {
  it("refuses a Microsoft sign-up whose address Microsoft didn't confirm", () => {
    expect(() =>
      assertMicrosoftSignUpConfirmed(
        { emailVerified: false },
        microsoftCallback,
      ),
    ).toThrow("unconfirmed_email");
    expect(() =>
      assertMicrosoftSignUpConfirmed(
        { emailVerified: true },
        microsoftCallback,
      ),
    ).not.toThrow();
  });

  it("refuses one through an ID token too", () => {
    expect(() =>
      assertMicrosoftSignUpConfirmed(
        { emailVerified: false },
        { path: "/sign-in/social", body: { provider: "microsoft" } },
      ),
    ).toThrow("unconfirmed_email");
  });

  it("leaves other sign-up methods alone", () => {
    expect(() =>
      assertMicrosoftSignUpConfirmed(
        { emailVerified: false },
        { path: "/callback/:id", params: { id: "google" } },
      ),
    ).not.toThrow();
    expect(() =>
      assertMicrosoftSignUpConfirmed(
        { emailVerified: false },
        { path: "/sign-up/email" },
      ),
    ).not.toThrow();
    expect(() =>
      assertMicrosoftSignUpConfirmed({ emailVerified: false }, null),
    ).not.toThrow();
    expect(() =>
      assertMicrosoftSignUpConfirmed(
        { emailVerified: false },
        { path: "/sign-in/social", body: { provider: "google" } },
      ),
    ).not.toThrow();
  });
});
