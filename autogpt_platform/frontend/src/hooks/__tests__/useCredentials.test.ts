import type { BlockIOCredentialsSubSchema } from "@/lib/autogpt-server-api";
import { renderHook } from "@testing-library/react";
import React from "react";
import { describe, expect, it, vi } from "vitest";
import {
  CredentialsProviderData,
  CredentialsProvidersContext,
  CredentialsProvidersContextType,
} from "@/providers/agent-credentials/credentials-provider";
import useCredentials, {
  deriveAuthMethods,
  getSupportedCredentialTypes,
} from "../useCredentials";

const schema: BlockIOCredentialsSubSchema = {
  type: "object",
  properties: {},
  credentials_provider: ["openai", "codex"],
  credentials_types: ["api_key", "oauth2"],
  discriminator: "transport",
  discriminator_mapping: {
    openai_api: "openai",
    codex_app_server: "codex",
  },
  discriminator_type_mapping: {
    openai_api: ["api_key"],
    codex_app_server: ["oauth2"],
  },
};

describe("getSupportedCredentialTypes", () => {
  it("narrows each transport to its valid credential type", () => {
    expect(getSupportedCredentialTypes(schema, "openai_api")).toEqual([
      "api_key",
    ]);
    expect(getSupportedCredentialTypes(schema, "codex_app_server")).toEqual([
      "oauth2",
    ]);
  });

  it("falls back to the schema types for legacy discriminators", () => {
    expect(getSupportedCredentialTypes(schema, "unknown")).toEqual([
      "api_key",
      "oauth2",
    ]);
  });
});

describe("deriveAuthMethods", () => {
  it("maps each credential type to its connect method", () => {
    expect(deriveAuthMethods(["api_key"])).toMatchObject({
      supportsApiKey: true,
      supportsOAuth2: false,
      supportsDeviceCode: false,
    });
    expect(deriveAuthMethods(["oauth2"])).toMatchObject({
      supportsOAuth2: true,
      supportsDeviceCode: false,
    });
    expect(deriveAuthMethods(["host_scoped"])).toMatchObject({
      supportsHostScoped: true,
    });
    expect(deriveAuthMethods(["user_password"])).toMatchObject({
      supportsUserPassword: true,
    });
  });

  // The regression: Stripe Link blocks list both types because a device-code
  // grant stores an oauth2 credential. Offering both put an OAuth tab in front
  // of a provider with no client secret, so connecting failed with
  // "Provider 'stripe_link' does not support OAuth".
  it("lets device_code shadow oauth2 so only the working method is offered", () => {
    expect(deriveAuthMethods(["oauth2", "device_code"])).toMatchObject({
      supportsDeviceCode: true,
      supportsOAuth2: false,
    });
  });

  it("still offers other methods alongside device auth", () => {
    expect(
      deriveAuthMethods(["oauth2", "device_code", "api_key"]),
    ).toMatchObject({
      supportsDeviceCode: true,
      supportsOAuth2: false,
      supportsApiKey: true,
    });
  });

  it("offers nothing when no types are supported", () => {
    expect(deriveAuthMethods([])).toEqual({
      supportsApiKey: false,
      supportsDeviceCode: false,
      supportsOAuth2: false,
      supportsUserPassword: false,
      supportsHostScoped: false,
    });
  });
});

describe("useCredentials provider list", () => {
  it("resolves a saved credential on a legacy node with no discriminator", () => {
    const savedCredential = {
      id: "codex-1",
      provider: "codex",
      type: "oauth2" as const,
      title: "ChatGPT for Codex",
    };
    const provider = {
      provider: "codex",
      providerName: "Codex",
      savedCredentials: [savedCredential],
      isSystemProvider: false,
      oAuthCallback: vi.fn(),
      mcpOAuthCallback: vi.fn(),
      createAPIKeyCredentials: vi.fn(),
      createUserPasswordCredentials: vi.fn(),
      createHostScopedCredentials: vi.fn(),
      deleteCredentials: vi.fn(),
    } satisfies CredentialsProviderData;

    const { result } = renderHook(
      () => useCredentials(schema, {}, savedCredential.provider),
      {
        wrapper: ({ children }: { children: React.ReactNode }) =>
          React.createElement(
            CredentialsProvidersContext.Provider,
            { value: { codex: provider } },
            children,
          ),
      },
    );

    expect(result.current?.provider).toBe("codex");
    expect(result.current?.savedCredentials).toEqual([savedCredential]);
  });

  it("exposes the unfiltered provider credentials alongside the filtered ones", async () => {
    const oauthOnly = {
      id: "codex-1",
      provider: "codex",
      type: "oauth2" as const,
      title: "ChatGPT for Codex",
    };
    const apiKeyToo = {
      id: "codex-2",
      provider: "codex",
      type: "api_key" as const,
      title: "A key",
    };

    const provider = {
      provider: "codex",
      providerName: "Codex",
      savedCredentials: [oauthOnly, apiKeyToo],
      isSystemProvider: false,
      oAuthCallback: vi.fn(),
      mcpOAuthCallback: vi.fn(),
      createAPIKeyCredentials: vi.fn(),
      createUserPasswordCredentials: vi.fn(),
      createHostScopedCredentials: vi.fn(),
      deleteCredentials: vi.fn(),
    } satisfies CredentialsProviderData;
    const providers: CredentialsProvidersContextType = { codex: provider };

    const codexOnlySchema = {
      type: "object",
      properties: {},
      credentials_provider: ["codex"],
      credentials_types: ["oauth2"],
    } as unknown as BlockIOCredentialsSubSchema;

    const { result } = renderHook(() => useCredentials(codexOnlySchema, {}), {
      wrapper: ({ children }: { children: React.ReactNode }) =>
        React.createElement(
          CredentialsProvidersContext.Provider,
          { value: providers },
          children,
        ),
    });

    if (result.current === null) {
      throw new Error("expected the provider to resolve");
    }

    // The api_key one is filtered out of savedCredentials by the schema's
    // supported types, but must still be present in the unfiltered list.
    expect(
      result.current.savedCredentials.map((credential) => credential.id),
    ).toEqual(["codex-1"]);
    expect(
      result.current.allProviderCredentials.map((credential) => credential.id),
    ).toEqual(["codex-1", "codex-2"]);
  });
});
