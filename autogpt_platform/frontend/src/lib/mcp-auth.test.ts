import { describe, expect, it } from "vitest";

import {
  detectMCPAuthScheme,
  prepareMCPAuthCredential,
} from "./mcp-auth";

describe("MCP manual authentication helpers", () => {
  it("does not guess a scheme for a bare credential", () => {
    expect(detectMCPAuthScheme("cGstbGYtYWJjZA==")).toBeNull();
  });

  it("detects explicit schemes, including a complete Authorization header", () => {
    expect(detectMCPAuthScheme("Basic cGstbGYtYWJjZA==")).toBe("basic");
    expect(
      detectMCPAuthScheme("Authorization: Bearer secret-token"),
    ).toBe("bearer");
  });

  it("keeps bare Bearer tokens backward-compatible", () => {
    expect(prepareMCPAuthCredential(" secret-token ", "bearer")).toBe(
      "secret-token",
    );
  });

  it("prefixes a bare Basic credential", () => {
    expect(prepareMCPAuthCredential(" cGstbGYtYWJjZA== ", "basic")).toBe(
      "Basic cGstbGYtYWJjZA==",
    );
  });

  it("preserves provider-supplied prefixes and complete headers", () => {
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
});
