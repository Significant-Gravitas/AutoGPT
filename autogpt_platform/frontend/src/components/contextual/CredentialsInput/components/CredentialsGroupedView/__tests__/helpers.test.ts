import { describe, expect, it } from "vitest";
import { findSavedUserCredentialByProviderAndType } from "../helpers";

function makeProviders(creds: unknown[]) {
  return { mcp: { savedCredentials: creds } } as never;
}

function apiKeyCred(id: string, host: string) {
  return { id, provider: "mcp", type: "api_key", title: `MCP: ${host}`, host };
}

describe("findSavedUserCredentialByProviderAndType — MCP api_key host matching", () => {
  it("does not auto-assign an api_key MCP credential belonging to a different server", () => {
    const providers = makeProviders([
      apiKeyCred("b", "https://mcp.serverB.com/mcp"),
    ]);

    const result = findSavedUserCredentialByProviderAndType(
      ["mcp"],
      ["oauth2", "api_key"],
      undefined,
      providers,
      ["https://mcp.serverA.com/mcp"],
    );

    // Without host filtering an api_key cred for server B would be wrongly
    // auto-assigned to server A, causing a runtime 401.
    expect(result).toBeUndefined();
  });

  it("auto-assigns the api_key MCP credential whose host matches the server", () => {
    const providers = makeProviders([
      apiKeyCred("a", "https://mcp.serverA.com/mcp"),
      apiKeyCred("b", "https://mcp.serverB.com/mcp"),
    ]);

    const result = findSavedUserCredentialByProviderAndType(
      ["mcp"],
      ["oauth2", "api_key"],
      undefined,
      providers,
      ["https://mcp.serverA.com/mcp"],
    );

    expect(result?.id).toBe("a");
  });
});
