import { describe, expect, it } from "vitest";
import { SIGN_IN_BLOCKED_CODE, SIGN_UP_BLOCKED_CODE } from "../access-blocks";
import { DISPOSABLE_EMAIL_CODE } from "../disposable-email";
import {
  getSocialSignInError,
  isWaitlistSignInError,
  microsoftProfileToUser,
} from "../microsoft-login";
import { SIGN_UP_NOT_ALLOWED_CODE } from "../signup-gate";
import { UNDELIVERABLE_EMAIL_CODE } from "../undeliverable-email";

const PERSONAL_TENANT = "9188040d-6c67-4c5b-b112-36a304b66dad";

describe("Continue with Microsoft", () => {
  it("trusts a personal account's own Microsoft mailbox and organization-verified work addresses", () => {
    expect(
      microsoftProfileToUser({
        sub: "s",
        tid: PERSONAL_TENANT,
        email: "me@outlook.com",
      }),
    ).toMatchObject({ emailVerified: true, email: "me@outlook.com" });
    expect(
      microsoftProfileToUser({
        sub: "s",
        tid: PERSONAL_TENANT,
        email: "Me@Hotmail.co.uk",
      }).emailVerified,
    ).toBe(true);
    expect(
      microsoftProfileToUser({
        sub: "s",
        tid: "company-tenant",
        email: "me@company.example",
        xms_edov: "1",
      }),
    ).toMatchObject({ emailVerified: true, email: "me@company.example" });
  });

  it.each([
    "https://sts.windows.net/company-tenant/",
    "https://login.microsoftonline.com/company-tenant/v2.0",
  ])("trusts a member whose idp is its own organization (%s)", (idp) => {
    expect(
      microsoftProfileToUser({
        sub: "s",
        tid: "company-tenant",
        idp,
        email: "me@company.example",
        xms_edov: true,
      }).emailVerified,
    ).toBe(true);
  });

  it.each([
    [
      "a personal account's other address",
      PERSONAL_TENANT,
      "me@company.example",
      {},
    ],
    [
      "a lookalike of a Microsoft domain",
      PERSONAL_TENANT,
      "me@outlook.com.example",
      {},
    ],
    [
      "a work address Microsoft did not verify",
      "attacker-tenant",
      "victim@company.example",
      {},
    ],
    [
      "a personal account invited into an organization as a guest",
      "attacker-tenant",
      "victim@company.example",
      { idp: "live.com", xms_edov: true },
    ],
    [
      "another organization's member invited as a guest",
      "attacker-tenant",
      "victim@company.example",
      { idp: "https://sts.windows.net/victim-tenant/", xms_edov: true },
    ],
    [
      "a guest from an identity provider whose address names the tenant",
      "attacker-tenant",
      "victim@company.example",
      { idp: "https://idp.example/attacker-tenant/", xms_edov: true },
    ],
  ])("doesn't vouch for %s", (_case, tid, email, claims) => {
    const user = microsoftProfileToUser({ sub: "s", tid, email, ...claims });

    // Kept, so an account already linked to this Microsoft login still signs
    // in (without an address, Better Auth stops at `email_not_found`).
    expect(user).toMatchObject({ emailVerified: false, email });
  });

  it.each([
    "pat@hotmail.se",
    "pat@live.se",
    "pat@outlook.com.br",
    "pat@msn.nl",
  ])("trusts %s, an Outlook.com mailbox on a country domain", (email) => {
    expect(
      microsoftProfileToUser({ sub: "s", tid: PERSONAL_TENANT, email })
        .emailVerified,
    ).toBe(true);
  });

  it.each(["pat@outlook.co.uk", "pat@hotmail.nl"])(
    "doesn't trust %s, a Microsoft-sounding domain whose mail goes elsewhere",
    (email) => {
      expect(
        microsoftProfileToUser({ sub: "s", tid: PERSONAL_TENANT, email })
          .emailVerified,
      ).toBe(false);
    },
  );

  it("explains refused sign-ins in plain language", () => {
    expect(getSocialSignInError(null)).toBeNull();
    // One answer whether or not an account uses the address, so it doesn't
    // tell which.
    const notLinked = getSocialSignInError("account_not_linked");
    expect(notLinked).toMatch(/If you already have an AutoGPT account, log in/);
    expect(notLinked).toMatch(/If not, sign up with your email address/);
    expect(notLinked).not.toMatch(/already uses/);
    expect(getSocialSignInError("email_not_found")).toMatch(
      /didn't share an email address/,
    );
    expect(getSocialSignInError("something_new")).toMatch(/didn't finish/);
    // Codes come from the URL, so an object's own names are just unknown codes.
    for (const code of ["__proto__", "constructor", "toString"]) {
      expect(getSocialSignInError(code)).toMatch(/didn't finish/);
    }
    // A refusal from one of our sign-up checks arrives as its code.
    for (const code of [
      SIGN_UP_BLOCKED_CODE,
      DISPOSABLE_EMAIL_CODE,
      UNDELIVERABLE_EMAIL_CODE,
    ]) {
      expect(getSocialSignInError(code)).toMatch(/sign-up isn't allowed/);
    }
    expect(getSocialSignInError(SIGN_IN_BLOCKED_CODE)).toMatch(
      /sign-in isn't allowed/,
    );
  });

  it("recognizes a sign-up the waitlist refused", () => {
    expect(isWaitlistSignInError(SIGN_UP_NOT_ALLOWED_CODE)).toBe(true);
    expect(isWaitlistSignInError("FORBIDDEN")).toBe(true);
    expect(isWaitlistSignInError("account_not_linked")).toBe(false);
    expect(isWaitlistSignInError(null)).toBe(false);
  });
});
