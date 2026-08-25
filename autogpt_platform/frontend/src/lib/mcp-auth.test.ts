import { describe, expect, it } from "vitest";

import { detectMCPAuthScheme, prepareMCPAuthCredential } from "./mcp-auth";

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
});
