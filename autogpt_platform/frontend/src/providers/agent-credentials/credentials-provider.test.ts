import { describe, expect, it } from "vitest";
import type {
  CredentialsMetaResponse,
  CredentialsProviderName,
} from "@/lib/autogpt-server-api";
import {
  upsertProviderCredentials,
  type CredentialsProvidersContextType,
} from "./credentials-provider";

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
