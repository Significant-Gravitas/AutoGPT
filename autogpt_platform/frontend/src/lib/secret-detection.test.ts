import { describe, expect, it } from "vitest";
import { containsLikelySecret } from "./secret-detection";

// Every sample is assembled at run time, so no secret scanner mistakes this
// file for a leaked key. None of them is a real credential.
function sample(prefix: string, body: string) {
  return prefix + body;
}

const ALNUM_40 = "aB3dE5gH7jK9mN1pQ3sT5vW7yZ9bC1eF3hJ5kL7n";

describe("containsLikelySecret", () => {
  it.each([
    ["OpenAI legacy key", sample("sk-", ALNUM_40 + "Xy12Ab34")],
    ["OpenAI project key", sample("sk-proj-", "Ab_" + ALNUM_40 + "-Cd")],
    [
      "OpenAI project key with frequent separators",
      sample("sk-proj-", "aB3d_E5gH-7jK9_mN1p-Q3sT_5vW7-yZ9b"),
    ],
    [
      "Anthropic key with frequent separators",
      sample("sk-ant-api03-", "aB3d_E5gH-7jK9_mN1p-Q3sT_5vW7-yZ9b"),
    ],
    ["Anthropic key", sample("sk-ant-api03-", ALNUM_40 + "_x-" + ALNUM_40)],
    ["GitHub classic token", sample("ghp_", ALNUM_40.slice(0, 36))],
    ["GitHub OAuth token", sample("gho_", ALNUM_40.slice(0, 36))],
    ["GitHub user-to-server token", sample("ghu_", ALNUM_40.slice(0, 36))],
    ["GitHub server-to-server token", sample("ghs_", ALNUM_40.slice(0, 36))],
    ["GitHub refresh token", sample("ghr_", ALNUM_40.slice(0, 36))],
    [
      "GitHub stateless installation token",
      sample("ghs_", "1234567_eyJhbGciOi." + ALNUM_40 + "." + ALNUM_40),
    ],
    [
      "GitHub fine-grained token",
      sample("github_", "pat_11ABCDEFG0" + ALNUM_40 + "_" + ALNUM_40),
    ],
    ["AWS access key", sample("AK", "IAZ2Q4W6E8R0T2Y4U6")],
    ["AWS temporary key", sample("AS", "IAZ2Q4W6E8R0T2Y4U6")],
    ["Slack bot token", sample("xo", "xb-123456789012-1234567890123-AbCdEfGh")],
    ["Slack user token", sample("xo", "xp-123456789012-98765-AbCdEfGhIjKl")],
    ["Slack refresh token", sample("xo", "xe-1-My0xNi0yNDY4MDEz-AbCdEfGh")],
    [
      "Slack app-level token",
      sample("xa", "pp-1-A0123456789-1234567-abcdef12"),
    ],
    [
      "Slack rotating bot token",
      sample("xoxe.", "xoxb-1-My0xNi0yNDY4MDEz-AbCdEfGh"),
    ],
    [
      "Slack rotating app token",
      sample("xoxe.", "xapp-1-A0123456789-1234567-abcdef12"),
    ],
    ["Stripe live key", sample("sk_", "live_" + ALNUM_40.slice(0, 24))],
    ["Stripe restricted key", sample("rk_", "live_" + ALNUM_40.slice(0, 24))],
    ["Google API key", sample("AI", "za" + ALNUM_40.slice(0, 35))],
    ["RSA private key", sample("-----BEGIN RSA ", "PRIVATE KEY-----\nMIIE")],
    ["PKCS#8 private key", sample("-----BEGIN ", "PRIVATE KEY-----")],
    ["OpenSSH private key", sample("-----BEGIN OPENSSH ", "PRIVATE KEY-----")],
    ["PGP private key", sample("-----BEGIN PGP ", "PRIVATE KEY BLOCK-----")],
  ])("flags a %s", (_name, secret) => {
    expect(containsLikelySecret(secret)).toBe(true);
    expect(containsLikelySecret(`here it is: ${secret} thanks`)).toBe(true);
  });

  it.each([
    ["plain text", "Summarise my inbox and draft replies"],
    ["a word starting with sk", "skeleton, skill, sky"],
    ["a short sk- word", "use sk-learn for the model"],
    ["a hyphenated sk- phrase", "sk-learn-and-pandas-for-quarterly-reports"],
    ["a word ending in sk", "risk-assessment-for-the-quarterly-review-2026"],
    ["a git SHA", "commit 3f786850e387550fdab836ed7e6dc881de23001b"],
    ["a UUID", "run 123e4567-e89b-12d3-a456-426614174000"],
    ["a short GitHub-like prefix", "ghp_short"],
    ["a Stripe test key", sample("sk_", "test_" + ALNUM_40.slice(0, 24))],
    ["a lowercase AKIA word", "akiaiosfodnn7example"],
    ["a public key block", "-----BEGIN PUBLIC KEY-----"],
    ["a certificate block", "-----BEGIN CERTIFICATE-----"],
    [
      "an sk- placeholder in a .env example",
      sample("OPENAI_API_KEY=sk-", "x".repeat(24)),
    ],
    ["an OpenAI project key placeholder", sample("sk-proj-", "X".repeat(24))],
    ["a GitHub token placeholder", sample("ghp_", "x".repeat(36))],
    ["AWS's documentation key", sample("AK", "IAIOSFODNN7EXAMPLE")],
  ])("leaves %s alone", (_name, text) => {
    expect(containsLikelySecret(text)).toBe(false);
  });

  it("stays fast on a long run of sk- prefixes", () => {
    const started = performance.now();
    expect(containsLikelySecret("sk-".repeat(50_000))).toBe(false);
    expect(performance.now() - started).toBeLessThan(1_000);
  });
});
