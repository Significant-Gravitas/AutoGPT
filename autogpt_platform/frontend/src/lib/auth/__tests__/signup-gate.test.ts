import { describe, expect, it } from "vitest";
import {
  isSignupAllowed,
  parseAllowlist,
  readSignupGateConfig,
  type SignupGateConfig,
} from "../signup-gate";

describe("parseAllowlist", () => {
  it("returns an empty list for unset / empty input", () => {
    expect(parseAllowlist(undefined)).toEqual([]);
    expect(parseAllowlist("")).toEqual([]);
    expect(parseAllowlist("  , ,")).toEqual([]);
  });

  it("trims, lowercases, and drops blanks", () => {
    expect(parseAllowlist(" Me@Example.com , @AGPT.co ,, x@y.z")).toEqual([
      "me@example.com",
      "@agpt.co",
      "x@y.z",
    ]);
  });
});

describe("readSignupGateConfig", () => {
  it("is open by default (prod posture) when env is unset", () => {
    const cfg = readSignupGateConfig({});
    expect(cfg).toEqual({ allowNewAccounts: true, allowlist: [] });
  });

  it("only an explicit 'false' disables new accounts", () => {
    expect(
      readSignupGateConfig({
        AUTH_ALLOW_NEW_ACCOUNTS: "false",
      }).allowNewAccounts,
    ).toBe(false);
    expect(
      readSignupGateConfig({
        AUTH_ALLOW_NEW_ACCOUNTS: "true",
      }).allowNewAccounts,
    ).toBe(true);
    expect(
      readSignupGateConfig({
        AUTH_ALLOW_NEW_ACCOUNTS: "0",
      }).allowNewAccounts,
    ).toBe(true);
  });

  it("reads the allowlist from AUTH_SIGNUP_ALLOWLIST", () => {
    expect(
      readSignupGateConfig({
        AUTH_SIGNUP_ALLOWLIST: "@agpt.co, dev@example.com",
      }).allowlist,
    ).toEqual(["@agpt.co", "dev@example.com"]);
  });
});

const open: SignupGateConfig = { allowNewAccounts: true, allowlist: [] };

describe("isSignupAllowed", () => {
  it("allows anyone when open (no allowlist, accounts enabled)", () => {
    expect(isSignupAllowed("anyone@example.com", open).allowed).toBe(true);
  });

  it("blocks everyone when new accounts are disabled, even if allowlisted", () => {
    const cfg: SignupGateConfig = {
      allowNewAccounts: false,
      allowlist: ["me@example.com"],
    };
    const decision = isSignupAllowed("me@example.com", cfg);
    expect(decision.allowed).toBe(false);
    // Phrased so the frontend isWaitlistError() catches it.
    expect(decision.reason?.toLowerCase()).toContain("not allowed");
  });

  it("allows an exact email match on the allowlist", () => {
    const cfg: SignupGateConfig = {
      allowNewAccounts: true,
      allowlist: ["dev@example.com"],
    };
    expect(isSignupAllowed("dev@example.com", cfg).allowed).toBe(true);
    expect(isSignupAllowed("DEV@Example.com", cfg).allowed).toBe(true); // case-insensitive
  });

  it("allows a domain match via @domain entries", () => {
    const cfg: SignupGateConfig = {
      allowNewAccounts: true,
      allowlist: ["@agpt.co"],
    };
    expect(isSignupAllowed("someone@agpt.co", cfg).allowed).toBe(true);
    expect(isSignupAllowed("someone@AGPT.co", cfg).allowed).toBe(true);
  });

  it("rejects an email that matches no allowlist entry", () => {
    const cfg: SignupGateConfig = {
      allowNewAccounts: true,
      allowlist: ["@agpt.co", "vip@example.com"],
    };
    const decision = isSignupAllowed("random@gmail.com", cfg);
    expect(decision.allowed).toBe(false);
    expect(decision.reason?.toLowerCase()).toContain("not allowed");
  });

  it("does not treat a domain entry as an email match or vice versa", () => {
    expect(
      isSignupAllowed("agpt.co@evil.com", {
        allowNewAccounts: true,
        allowlist: ["@agpt.co"],
      }).allowed,
    ).toBe(false);
  });
});
