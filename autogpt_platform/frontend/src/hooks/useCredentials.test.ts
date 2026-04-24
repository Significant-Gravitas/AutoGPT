import { describe, expect, it } from "vitest";
import { renderHook } from "@testing-library/react";
import React from "react";
import { classifyCredentials } from "./useCredentials";
import useCredentials from "./useCredentials";
import type {
  BlockIOCredentialsSubSchema,
  CredentialsMetaResponse,
} from "@/lib/autogpt-server-api";
import {
  CredentialsProvidersContext,
  type CredentialsProvidersContextType,
} from "@/providers/agent-credentials/credentials-provider";

function makeSchema(
  partial: Partial<BlockIOCredentialsSubSchema> = {},
): BlockIOCredentialsSubSchema {
  return {
    credentials_provider: ["google"],
    credentials_types: ["oauth2"],
    credentials_scopes: [],
    ...partial,
  } as BlockIOCredentialsSubSchema;
}

function makeCred(
  partial: Partial<CredentialsMetaResponse>,
): CredentialsMetaResponse {
  return {
    id: "cred-id",
    provider: "google",
    type: "oauth2",
    title: "Test Credential",
    scopes: [],
    ...partial,
  } as CredentialsMetaResponse;
}

describe("classifyCredentials", () => {
  it("drops credentials of unsupported types", () => {
    const schema = makeSchema({ credentials_types: ["oauth2"] });
    const { savedCredentials, upgradeableCredentials } = classifyCredentials(
      [
        makeCred({ id: "a", type: "api_key" }),
        makeCred({ id: "b", type: "oauth2" }),
      ],
      schema,
      undefined,
    );

    expect(savedCredentials.map((c) => c.id)).toEqual(["b"]);
    expect(upgradeableCredentials).toEqual([]);
  });

  it("classifies OAuth2 creds with all required scopes as saved", () => {
    const schema = makeSchema({
      credentials_scopes: ["drive.file", "drive.metadata"],
    });
    const { savedCredentials, upgradeableCredentials } = classifyCredentials(
      [
        makeCred({
          id: "full",
          scopes: ["drive.file", "drive.metadata", "drive.readonly"],
        }),
      ],
      schema,
      undefined,
    );

    expect(savedCredentials.map((c) => c.id)).toEqual(["full"]);
    expect(upgradeableCredentials).toEqual([]);
  });

  it("classifies OAuth2 creds missing a scope as upgradeable (not discarded)", () => {
    // Regression coverage for the incremental-OAuth flow: a credential
    // that's missing only one scope must land in upgradeableCredentials so
    // the UI can offer the user a scope-upgrade flow rather than force
    // them to create a whole new credential from scratch.
    const schema = makeSchema({
      credentials_scopes: ["drive.file", "drive.metadata"],
    });
    const { savedCredentials, upgradeableCredentials } = classifyCredentials(
      [makeCred({ id: "narrow", scopes: ["drive.file"] })],
      schema,
      undefined,
    );

    expect(savedCredentials).toEqual([]);
    expect(upgradeableCredentials.map((c) => c.id)).toEqual(["narrow"]);
  });

  it("treats schemas without credentials_scopes as no-scope-required", () => {
    const schema = makeSchema({ credentials_scopes: undefined });
    const { savedCredentials, upgradeableCredentials } = classifyCredentials(
      [makeCred({ id: "anything", scopes: [] })],
      schema,
      undefined,
    );

    expect(savedCredentials.map((c) => c.id)).toEqual(["anything"]);
    expect(upgradeableCredentials).toEqual([]);
  });

  it("filters MCP OAuth2 credentials by host (discriminator) and never upgrades them", () => {
    const schema = makeSchema({
      credentials_provider: ["mcp"],
      credentials_types: ["oauth2"],
    });
    const creds = [
      makeCred({ id: "match", provider: "mcp", host: "https://mcp.example" }),
      makeCred({
        id: "different-host",
        provider: "mcp",
        host: "https://other.example",
      }),
    ];

    const matchingDiscriminator = classifyCredentials(
      creds,
      schema,
      "https://mcp.example",
    );
    expect(matchingDiscriminator.savedCredentials.map((c) => c.id)).toEqual([
      "match",
    ]);
    expect(matchingDiscriminator.upgradeableCredentials).toEqual([]);

    // A missing discriminator must drop all MCP creds (not upgrade them).
    const noDiscriminator = classifyCredentials(creds, schema, undefined);
    expect(noDiscriminator.savedCredentials).toEqual([]);
    expect(noDiscriminator.upgradeableCredentials).toEqual([]);
  });

  it("host_scoped credentials: discriminator URL is hostname-compared to c.host", () => {
    const schema = makeSchema({ credentials_types: ["host_scoped"] });
    const { savedCredentials, upgradeableCredentials } = classifyCredentials(
      [
        makeCred({
          id: "match",
          type: "host_scoped",
          host: "api.example.com",
        }),
        makeCred({
          id: "mismatch",
          type: "host_scoped",
          host: "other.example.com",
        }),
      ],
      schema,
      "https://api.example.com/something",
    );

    expect(savedCredentials.map((c) => c.id)).toEqual(["match"]);
    expect(upgradeableCredentials).toEqual([]);
  });

  it("includes api_key and user_password credentials unconditionally when type is supported", () => {
    const schema = makeSchema({
      credentials_types: ["api_key", "user_password"],
    });
    const { savedCredentials, upgradeableCredentials } = classifyCredentials(
      [
        makeCred({ id: "k", type: "api_key" }),
        makeCred({ id: "p", type: "user_password" }),
      ],
      schema,
      undefined,
    );

    expect(savedCredentials.map((c) => c.id).sort()).toEqual(["k", "p"]);
    expect(upgradeableCredentials).toEqual([]);
  });
});

function makeProviderMap(
  saved: CredentialsMetaResponse[],
): CredentialsProvidersContextType {
  return {
    google: {
      provider: "google",
      providerName: "Google",
      savedCredentials: saved,
      isSystemProvider: false,
      oAuthCallback: async () => makeCred({}),
      mcpOAuthCallback: async () => makeCred({}),
      createAPIKeyCredentials: async () => makeCred({}),
      createUserPasswordCredentials: async () => makeCred({}),
      createHostScopedCredentials: async () => makeCred({}),
      deleteCredentials: async () => ({ deleted: true, revoked: true }),
    },
  };
}

describe("useCredentials (hook)", () => {
  it("returns upgradeableCredentials for OAuth2 creds missing scopes", () => {
    const schema = makeSchema({
      credentials_provider: ["google"],
      credentials_types: ["oauth2"],
      credentials_scopes: ["drive.file", "drive.metadata"],
    });

    const full = makeCred({
      id: "full",
      scopes: ["drive.file", "drive.metadata"],
    });
    const narrow = makeCred({ id: "narrow", scopes: ["drive.file"] });
    const providers = makeProviderMap([full, narrow]);

    function Wrapper({ children }: { children: React.ReactNode }) {
      return React.createElement(
        CredentialsProvidersContext.Provider,
        { value: providers },
        children,
      );
    }

    const { result } = renderHook(() => useCredentials(schema), {
      wrapper: Wrapper,
    });

    expect(result.current).not.toBeNull();
    const data = result.current!;
    expect(data.isLoading).toBe(false);
    if (data.isLoading) return;

    expect(data.savedCredentials.map((c) => c.id)).toEqual(["full"]);
    expect(data.upgradeableCredentials.map((c) => c.id)).toEqual(["narrow"]);
  });

  it("returns empty upgradeableCredentials when all scopes match", () => {
    const schema = makeSchema({
      credentials_provider: ["google"],
      credentials_types: ["oauth2"],
      credentials_scopes: ["drive.file"],
    });

    const cred = makeCred({ id: "ok", scopes: ["drive.file", "drive.meta"] });
    const providers = makeProviderMap([cred]);

    function Wrapper({ children }: { children: React.ReactNode }) {
      return React.createElement(
        CredentialsProvidersContext.Provider,
        { value: providers },
        children,
      );
    }

    const { result } = renderHook(() => useCredentials(schema), {
      wrapper: Wrapper,
    });

    expect(result.current).not.toBeNull();
    const data = result.current!;
    expect(data.isLoading).toBe(false);
    if (data.isLoading) return;

    expect(data.savedCredentials.map((c) => c.id)).toEqual(["ok"]);
    expect(data.upgradeableCredentials).toEqual([]);
  });
});
