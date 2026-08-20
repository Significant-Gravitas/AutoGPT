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
