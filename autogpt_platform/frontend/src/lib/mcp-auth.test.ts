import { describe, expect, it } from "vitest";

import {
  detectMCPAuthScheme,
  prepareMCPAuthCredential,
  validateMCPAuthCredential,
} from "./mcp-auth";

describe("MCP manual authentication helpers", () => {
  it("does not guess a scheme for a bare credential", () => {
    expect(detectMCPAuthScheme("cGstbGYtYWJjZA==")).toBeNull();
  });

  it("detects explicit schemes, including a complete Authorization header", () => {
    expect(detectMCPAuthScheme("Basic cGstbGYtYWJjZA==")).toBe("basic");
    expect(detectMCPAuthScheme("Authorization: Bearer secret-token")).toBe(
      "bearer",
    );
  });

  it("prefixes a bare Bearer credential", () => {
    expect(prepareMCPAuthCredential(" secret-token ", "bearer")).toBe(
      "Bearer secret-token",
    );
  });

  it("prefixes a bare Bearer credential containing spaces", () => {
    expect(prepareMCPAuthCredential(" orgid api-key ", "bearer")).toBe(
      "Bearer orgid api-key",
    );
  });

  it("prefixes a bare Basic credential", () => {
    expect(prepareMCPAuthCredential(" cGstbGYtYWJjZA== ", "basic")).toBe(
      "Basic cGstbGYtYWJjZA==",
    );
  });

  it("canonicalizes provider-supplied prefixes and complete headers", () => {
    expect(
      prepareMCPAuthCredential(
        "Authorization: Basic cGstbGYtYWJjZA==",
        "basic",
      ),
    ).toBe("Authorization: Basic cGstbGYtYWJjZA==");
    expect(prepareMCPAuthCredential("Bearer secret-token", "bearer")).toBe(
      "Bearer secret-token",
    );
  });

  it("makes the selected scheme authoritative over a pasted prefix", () => {
    expect(prepareMCPAuthCredential("Basic abc", "bearer")).toBe("Bearer abc");
    expect(prepareMCPAuthCredential("Authorization: Bearer abc", "basic")).toBe(
      "Authorization: Basic abc",
    );
  });

  it("preserves unsupported complete Authorization headers for validation", () => {
    expect(
      prepareMCPAuthCredential("Authorization: Digest abc", "bearer"),
    ).toBe("Authorization: Digest abc");
  });

  // A scheme word followed by more than one word is a bare multi-word
  // credential, not a prefix — the backend reads it the same way, and the two
  // ends disagreeing is what silently rewrites a stored secret.
  it("does not read a scheme out of a bare multi-word credential", () => {
    expect(detectMCPAuthScheme("basic auth key")).toBeNull();
    expect(prepareMCPAuthCredential("basic auth key", "bearer")).toBe(
      "Bearer basic auth key",
    );
  });

  it("rejects an unencoded user:password under the Basic scheme", () => {
    // The Base64 alphabet has no ":", so this is the raw pair a provider's
    // docs show. Sent verbatim it 401s with nothing naming the missing step.
    expect(validateMCPAuthCredential("pk-lf-abc:sk-lf-xyz", "basic")).toMatch(
      /unencoded user:password/,
    );
    expect(
      validateMCPAuthCredential("Basic pk-lf-abc:sk-lf-xyz", "basic"),
    ).toMatch(/unencoded user:password/);
    // Bearer tokens legitimately contain colons.
    expect(validateMCPAuthCredential("user:pass", "bearer")).toBeNull();
  });

  it("rejects a Basic credential containing whitespace", () => {
    expect(validateMCPAuthCredential("dXNlcjpwYXNz", "basic")).toBeNull();
    expect(validateMCPAuthCredential("Basic dXNlcjpwYXNz", "basic")).toBeNull();
    expect(validateMCPAuthCredential("user pass", "basic")).toMatch(
      /cannot contain spaces/,
    );
  });

  it("allows whitespace in a Bearer credential", () => {
    expect(validateMCPAuthCredential("orgid api-key", "bearer")).toBeNull();
  });
});
