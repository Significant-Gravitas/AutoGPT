import { describe, expect, it } from "vitest";
import type { BlockIOCredentialsSubSchema } from "@/lib/autogpt-server-api";
import { getCredentialProviderFromSchema } from "../helpers";

// AutoPilot's shape: one provider, but a discriminator whose `platform` value
// is deliberately unmapped because that transport needs no credential.
const autopilotSchema = {
  credentials_provider: ["codex"],
  credentials_types: ["oauth2"],
  discriminator: "transport",
  discriminator_mapping: { codex_app_server: "codex" },
} as unknown as BlockIOCredentialsSubSchema;

// A plain single-provider field with no discriminator at all.
const githubSchema = {
  credentials_provider: ["github"],
  credentials_types: ["api_key"],
} as unknown as BlockIOCredentialsSubSchema;

describe("getCredentialProviderFromSchema", () => {
  it("hides the input when a single-provider field's value is unmapped", () => {
    // Regression: this returned "codex" regardless of the transport, so
    // selecting `platform` still rendered — and auto-selected — a ChatGPT
    // connection that the platform transport never uses.
    expect(
      getCredentialProviderFromSchema(
        { transport: "platform" },
        autopilotSchema,
      ),
    ).toBeNull();
  });

  it("resolves the provider when the value is mapped", () => {
    expect(
      getCredentialProviderFromSchema(
        { transport: "codex_app_server" },
        autopilotSchema,
      ),
    ).toBe("codex");
  });

  it("still returns the lone provider when no discriminator is declared", () => {
    expect(getCredentialProviderFromSchema({}, githubSchema)).toBe("github");
  });
});
