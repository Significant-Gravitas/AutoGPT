import type { BlockIOCredentialsSubSchema } from "@/lib/autogpt-server-api";
import { describe, expect, it } from "vitest";
import {
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
