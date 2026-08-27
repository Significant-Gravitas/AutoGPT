import { describe, expect, it } from "vitest";
import type {
  CredentialsMetaResponse,
  CredentialsProviderName,
} from "@/lib/autogpt-server-api";
import {
  replaceMCPServerCredentials,
  upsertProviderCredentials,
  type CredentialsProvidersContextType,
} from "../credentials-provider";

function makeCred(
  partial: Partial<CredentialsMetaResponse>,
): CredentialsMetaResponse {
  return {
    id: "cred-id",
    provider: "google" as CredentialsProviderName,
    type: "oauth2",
    title: "Test Credential",
    scopes: [],
    ...partial,
  } as CredentialsMetaResponse;
}

function makeProviderMap(
  initial: Partial<
    Record<CredentialsProviderName, CredentialsMetaResponse[]>
  > = {},
): CredentialsProvidersContextType {
  const out: CredentialsProvidersContextType = {};
  for (const [provider, saved] of Object.entries(initial)) {
    out[provider as CredentialsProviderName] = {
      provider: provider as CredentialsProviderName,
      providerName: provider,
      savedCredentials: saved ?? [],
      isSystemProvider: false,
      oAuthCallback: async () => makeCred({}),
      mcpOAuthCallback: async () => makeCred({}),
      mcpStoreToken: async () => makeCred({}),
      createAPIKeyCredentials: async () => makeCred({}),
      createUserPasswordCredentials: async () => makeCred({}),
      createHostScopedCredentials: async () => makeCred({}),
      deleteCredentials: async () => ({ deleted: true, revoked: true }),
    };
  }
  return out;
}

describe("upsertProviderCredentials", () => {
  it("returns prev as-is when the provider isn't in the map", () => {
    const prev = makeProviderMap({ google: [] });
    const result = upsertProviderCredentials(
      prev,
      "github" as CredentialsProviderName,
      makeCred({ id: "new-gh" }),
    );
    expect(result).toBe(prev);
  });

  it("returns prev as-is when prev is null", () => {
    const result = upsertProviderCredentials(
      null,
      "google" as CredentialsProviderName,
      makeCred({ id: "anything" }),
    );
    expect(result).toBeNull();
  });

  it("appends a credential that isn't already in the list", () => {
    const prev = makeProviderMap({ google: [makeCred({ id: "existing" })] });
    const result = upsertProviderCredentials(
      prev,
      "google" as CredentialsProviderName,
      makeCred({ id: "new" }),
    );
    expect(result?.google?.savedCredentials.map((c) => c.id).sort()).toEqual([
      "existing",
      "new",
    ]);
  });

  it("replaces an existing credential with the same id (no duplication)", () => {
    // Regression coverage for the scope-upgrade path: after the callback
    // returns the upgraded credential, we must REPLACE the existing entry
    // in the sidebar — not append a second row with the same id.
    const prev = makeProviderMap({
      google: [
        makeCred({
          id: "cred-1",
          title: "Old",
          scopes: ["drive.file"],
        }),
      ],
    });
    const upgraded = makeCred({
      id: "cred-1",
      title: "Upgraded",
      scopes: ["drive.file", "drive.metadata"],
    });

    const result = upsertProviderCredentials(
      prev,
      "google" as CredentialsProviderName,
      upgraded,
    );
    const saved = result?.google?.savedCredentials;
    expect(saved?.length).toBe(1);
    expect(saved?.[0].title).toBe("Upgraded");
    expect(saved?.[0].scopes).toEqual(["drive.file", "drive.metadata"]);
  });

  it("returns a new top-level object (doesn't mutate prev)", () => {
    const prev = makeProviderMap({ google: [] });
    const snapshot = prev.google?.savedCredentials;
    const result = upsertProviderCredentials(
      prev,
      "google" as CredentialsProviderName,
      makeCred({ id: "x" }),
    );
    expect(result).not.toBe(prev);
    expect(result?.google?.savedCredentials).not.toBe(snapshot);
    // snapshot of the old list must still be empty
    expect(prev.google?.savedCredentials).toEqual([]);
  });
});

describe("replaceMCPServerCredentials", () => {
  function mcpMap(saved: CredentialsMetaResponse[]) {
    return {
      mcp: {
        provider: "mcp" as CredentialsProviderName,
        providerName: "mcp",
        savedCredentials: saved,
        isSystemProvider: false,
      },
    } as unknown as CredentialsProvidersContextType;
  }

  const fresh = makeCred({
    id: "new",
    provider: "mcp" as CredentialsProviderName,
    type: "api_key",
    host: "https://mcp.example.com/mcp",
  });

  it("drops the server's previous credential and appends the new one", () => {
    // The backend deletes every prior credential for the server before
    // returning a new ID, so a plain upsert would leave a deleted row in the
    // list and the picker would keep re-selecting it.
    const prev = mcpMap([
      makeCred({
        id: "old",
        provider: "mcp" as CredentialsProviderName,
        host: "https://mcp.example.com/mcp",
      }),
    ]);

    const result = replaceMCPServerCredentials(
      prev,
      "https://mcp.example.com/mcp",
      fresh,
    );

    expect(result?.mcp?.savedCredentials.map((c) => c.id)).toEqual(["new"]);
  });

  it("keeps credentials belonging to other servers", () => {
    const other = makeCred({
      id: "other",
      provider: "mcp" as CredentialsProviderName,
      host: "https://mcp.other.com/mcp",
    });
    const prev = mcpMap([other]);

    const result = replaceMCPServerCredentials(
      prev,
      "https://mcp.example.com/mcp",
      fresh,
    );

    expect(result?.mcp?.savedCredentials.map((c) => c.id)).toEqual([
      "other",
      "new",
    ]);
  });

  it("normalizes trailing slashes on both sides when matching", () => {
    const prev = mcpMap([
      makeCred({
        id: "old",
        provider: "mcp" as CredentialsProviderName,
        host: "https://mcp.example.com/mcp/",
      }),
    ]);

    const result = replaceMCPServerCredentials(
      prev,
      "https://mcp.example.com/mcp",
      fresh,
    );

    expect(result?.mcp?.savedCredentials.map((c) => c.id)).toEqual(["new"]);
  });

  it("never drops the incoming credential when it is already listed", () => {
    const prev = mcpMap([fresh]);

    const result = replaceMCPServerCredentials(
      prev,
      "https://mcp.example.com/mcp",
      fresh,
    );

    expect(result?.mcp?.savedCredentials.map((c) => c.id)).toEqual(["new"]);
  });

  it("leaves a map without an mcp provider untouched", () => {
    const prev = makeProviderMap({ google: [] });
    expect(
      replaceMCPServerCredentials(prev, "https://mcp.example.com/mcp", fresh),
    ).toBe(prev);
    expect(
      replaceMCPServerCredentials(null, "https://x/mcp", fresh),
    ).toBeNull();
  });

  it("does not mutate the previous map", () => {
    const prev = mcpMap([
      makeCred({
        id: "old",
        provider: "mcp" as CredentialsProviderName,
        host: "https://mcp.example.com/mcp",
      }),
    ]);

    replaceMCPServerCredentials(prev, "https://mcp.example.com/mcp", fresh);

    expect(prev.mcp?.savedCredentials.map((c) => c.id)).toEqual(["old"]);
  });
});
