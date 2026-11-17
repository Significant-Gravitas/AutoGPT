import { describe, expect, it } from "vitest";
import { providerRunsMailbox } from "../mailbox-provider";

const PERSONAL_TENANT = "9188040d-6c67-4c5b-b112-36a304b66dad";

describe("providerRunsMailbox", () => {
  it.each([
    ["pat@gmail.com", true, undefined, true],
    ["Pat@GoogleMail.com", true, undefined, true],
    // Google says it checked these, but doesn't run their mail.
    ["pat@company.example", true, { hd: "company.example" }, false],
    ["pat@yahoo.example", true, undefined, false],
    ["pat@gmail.com", false, undefined, false],
    ["not-an-address", true, undefined, false],
  ])(
    "Google: %s (verified %s, %o) is %s",
    (email, emailVerified, profile, expected) => {
      expect(
        providerRunsMailbox("google", { email, emailVerified, profile }),
      ).toBe(expected);
    },
  );

  it.each([
    [{ tid: PERSONAL_TENANT, email: "pat@outlook.com" }, true],
    [{ tid: PERSONAL_TENANT, email: "pat@hotmail.co.uk" }, true],
    // A personal account can carry any address; a work tenant vouches for a
    // domain it verified, but someone else runs its mail.
    [{ tid: PERSONAL_TENANT, email: "pat@gmail.com" }, false],
    [
      { tid: "company-tenant", email: "pat@company.example", xms_edov: true },
      false,
    ],
    [
      { tid: "company-tenant", email: "pat@outlook.com", xms_edov: true },
      false,
    ],
  ])("Microsoft: %o is %s", (profile, expected) => {
    expect(
      providerRunsMailbox("microsoft", {
        email: String(profile.email),
        emailVerified: true,
        profile,
      }),
    ).toBe(expected);
  });

  it.each(["github", "discord", "apple", "lenovo"])(
    "%s runs no mailbox this rule knows",
    (provider) => {
      expect(
        providerRunsMailbox(provider, {
          email: "pat@gmail.com",
          emailVerified: true,
          profile: {},
        }),
      ).toBe(false);
    },
  );
});
