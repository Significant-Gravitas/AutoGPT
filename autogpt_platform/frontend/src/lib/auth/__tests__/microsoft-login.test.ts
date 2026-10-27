import { describe, expect, it } from "vitest";
import {
  getSocialSignInError,
  isWaitlistSignInError,
  microsoftProfileToUser,
} from "../microsoft-login";

const PERSONAL_TENANT = "9188040d-6c67-4c5b-b112-36a304b66dad";

describe("Continue with Microsoft", () => {
  it("keys the account on tenant and account id, not the address", () => {
    const personal = microsoftProfileToUser({
      sub: "pairwise-1",
      oid: "personal-id",
      tid: PERSONAL_TENANT,
      email: "same@company.example",
    });
    const work = microsoftProfileToUser({
      sub: "pairwise-2",
      oid: "work-id",
      tid: "company-tenant",
      email: "same@company.example",
      xms_edov: true,
    });

    expect(personal.id).toBe(`${PERSONAL_TENANT}:personal-id`);
    expect(work.id).toBe("company-tenant:work-id");
    expect(personal.id).not.toBe(work.id);
  });

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

  it("trusts a member whose idp names its own organization", () => {
    expect(
      microsoftProfileToUser({
        sub: "s",
        tid: "company-tenant",
        idp: "https://sts.windows.net/company-tenant/",
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
  ])("doesn't vouch for %s", (_case, tid, email, claims) => {
    const user = microsoftProfileToUser({ sub: "s", tid, email, ...claims });

    // Kept, so an account that already uses it can be named in the refusal.
    expect(user).toMatchObject({ emailVerified: false, email });
  });

  it("explains refused sign-ins in plain language", () => {
    expect(getSocialSignInError(null)).toBeNull();
    expect(getSocialSignInError("unconfirmed_email")).toMatch(
      /Sign up with your email address instead/,
    );
    expect(getSocialSignInError("email_not_found")).toMatch(
      /didn't share an email address/,
    );
    expect(getSocialSignInError("something_new")).toMatch(/didn't finish/);
  });

  it("recognizes a sign-up the waitlist refused", () => {
    expect(
      isWaitlistSignInError(
        "This_email_address_is_not_allowed_to_register._Please_contact_support_for_assistance.",
      ),
    ).toBe(true);
    expect(isWaitlistSignInError("FORBIDDEN")).toBe(true);
    expect(isWaitlistSignInError("account_not_linked")).toBe(false);
    expect(isWaitlistSignInError("unconfirmed_email")).toBe(false);
    expect(isWaitlistSignInError(null)).toBe(false);
  });
});
