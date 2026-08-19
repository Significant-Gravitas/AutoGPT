import type { BlockIOCredentialsSubSchema } from "@/lib/autogpt-server-api";
import { describe, expect, it } from "vitest";
import { getSupportedCredentialTypes } from "../useCredentials";

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

describe("useCredentials provider list", () => {
  it("exposes the unfiltered provider credentials alongside the filtered ones", async () => {
    // `savedCredentials` is narrowed by supported type and discriminator, so it
    // cannot answer "was this credential deleted?" — a credential filtered out
    // for the current selection looks identical to one that no longer exists.
    // `providerCredentials` is the unfiltered list callers need for that.
    const { renderHook } = await import("@testing-library/react");
    const { CredentialsProvidersContext } = await import(
      "@/providers/agent-credentials/credentials-provider"
    );
    const useCredentials = (await import("../useCredentials")).default;
    const React = (await import("react")).default;

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

    const providers = {
      codex: {
        provider: "codex",
        providerName: "Codex",
        savedCredentials: [oauthOnly, apiKeyToo],
        isSystemProvider: false,
      },
    } as any;

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

    if (result.current === null || result.current.isLoading) {
      throw new Error("expected the provider to resolve");
    }

    // The api_key one is filtered out of savedCredentials by the schema's
    // supported types, but must still be present in the unfiltered list.
    expect(result.current.savedCredentials.map((c: any) => c.id)).toEqual([
      "codex-1",
    ]);
    expect(result.current.providerCredentials.map((c: any) => c.id)).toEqual([
      "codex-1",
      "codex-2",
    ]);
  });
});
