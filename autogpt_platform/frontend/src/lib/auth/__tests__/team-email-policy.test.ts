import { describe, expect, it } from "vitest";
import { assertTeamEmailUsesGoogle, isTeamEmail } from "../team-email-policy";

describe("isTeamEmail", () => {
  it.each([
    ["someone@agpt.co", true],
    ["Someone@AGPT.CO", true],
    ["someone@previews.agpt.co", false],
    ["someone@agpt.com", false],
    ["agpt.co@evil.com", false],
    ["someone@agpt.co.evil.com", false],
    ["no-at-sign", false],
  ])("%s → %s", (email, expected) => {
    expect(isTeamEmail(email)).toBe(expected);
  });
});

describe("assertTeamEmailUsesGoogle", () => {
  it("throws only for a team address on the password sign-up path", () => {
    expect(() =>
      assertTeamEmailUsesGoogle("x@agpt.co", { path: "/sign-up/email" }),
    ).toThrow("Please use Google sign-in");
    expect(() =>
      assertTeamEmailUsesGoogle("x@agpt.co", { path: "/callback/google" }),
    ).not.toThrow();
    expect(() => assertTeamEmailUsesGoogle("x@agpt.co", null)).not.toThrow();
    expect(() =>
      assertTeamEmailUsesGoogle("x@example.com", { path: "/sign-up/email" }),
    ).not.toThrow();
  });
});
