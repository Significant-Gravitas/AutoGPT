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

// The backend's copy on purpose: one table keeps the two grammars in step.
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
      if (authCase.headerForm) return;
      expect(
        prepareMCPAuthCredential(authCase.input, authCase.scheme ?? "bearer"),
      ).toBe(authCase.canonical);
    },
  );
});

describe("MCP manual authentication helpers", () => {
  it("prefixes a bare Basic credential", () => {
    expect(prepareMCPAuthCredential(" cGstbGYtYWJjZA== ", "basic")).toBe(
      "Basic cGstbGYtYWJjZA==",
    );
  });

  it("keeps a complete Authorization header in header form", () => {
    expect(
      prepareMCPAuthCredential(
        "Authorization: Basic cGstbGYtYWJjZA==",
        "basic",
      ),
    ).toBe("Authorization: Basic cGstbGYtYWJjZA==");
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

  it("rejects an unencoded user:password under the Basic scheme", () => {
    expect(validateMCPAuthCredential("pk-lf-abc:sk-lf-xyz", "basic")).toMatch(
      /unencoded user:password/,
    );
    expect(
      validateMCPAuthCredential("Basic pk-lf-abc:sk-lf-xyz", "basic"),
    ).toMatch(/unencoded user:password/);
    expect(validateMCPAuthCredential("user:pass", "bearer")).toBeNull();
  });

  it("rejects a Basic credential containing whitespace", () => {
    expect(validateMCPAuthCredential("dXNlcjpwYXNz", "basic")).toBeNull();
    expect(validateMCPAuthCredential("Basic dXNlcjpwYXNz", "basic")).toBeNull();
    expect(validateMCPAuthCredential("Basic auth key", "basic")).toMatch(
      /cannot contain spaces/,
    );
  });

  it("allows whitespace in a Bearer credential", () => {
    expect(validateMCPAuthCredential("orgid api-key", "bearer")).toBeNull();
  });
});
