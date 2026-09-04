import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import {
  detectMCPAuthScheme,
  prepareMCPAuthCredential,
  validateMCPAuthCredential,
  type MCPAuthScheme,
} from "./mcp-auth";

interface AuthCase {
  input: string;
  scheme: MCPAuthScheme | null;
  canonical: string;
  headerForm?: boolean;
}

// Deliberately the backend's copy rather than a duplicate under src/. The two
// implementations of this grammar disagreed while each suite asserted its own
// table, so `Bearer orgid api-key` was scheme-prefixed to the backend and bare
// here — and this side rewrote it to `Bearer Bearer orgid api-key`. Reading the
// same file is what makes a future divergence fail a test.
const authCases: { accepted: AuthCase[] } = JSON.parse(
  readFileSync(
    resolve(
      dirname(fileURLToPath(import.meta.url)),
      "../../../backend/backend/blocks/mcp/mcp_auth_cases.json",
    ),
    "utf8",
  ),
);

describe("the shared backend/frontend credential grammar", () => {
  it.each(authCases.accepted)(
    "reads $input the way the backend does",
    (authCase) => {
      expect(detectMCPAuthScheme(authCase.input)).toBe(authCase.scheme);

      // A complete Authorization header is preserved verbatim on this side so
      // the backend can reject an unsupported scheme explicitly instead of
      // disguising it as a Bearer token; only the scheme reading is shared.
      if (authCase.headerForm) return;
      expect(
        prepareMCPAuthCredential(authCase.input, authCase.scheme ?? "bearer"),
      ).toBe(authCase.canonical);
    },
  );
});

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

  // A scheme word takes everything after it as the credential, whitespace and
  // all — `value.split(None, 1)` on the backend. This suite used to assert the
  // opposite (a single token68 run), which is what made `Bearer orgid api-key`
  // come out as `Bearer Bearer orgid api-key`. A multi-word Basic credential is
  // still wrong, but it is `validateMCPAuthCredential` that says so, with a
  // message naming the Base64 step.
  it("takes the whole remainder after a scheme word, as the backend does", () => {
    expect(detectMCPAuthScheme("basic auth key")).toBe("basic");
    expect(prepareMCPAuthCredential("basic auth key", "basic")).toBe(
      "Basic auth key",
    );
    expect(validateMCPAuthCredential("basic auth key", "basic")).toMatch(
      /cannot contain spaces/,
    );
    expect(prepareMCPAuthCredential("Bearer orgid api-key", "bearer")).toBe(
      "Bearer orgid api-key",
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
