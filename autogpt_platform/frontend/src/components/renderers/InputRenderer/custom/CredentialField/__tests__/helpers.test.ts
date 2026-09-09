import { describe, expect, it } from "vitest";
import type { BlockIOCredentialsSubSchema } from "@/lib/autogpt-server-api";
import {
  credentialNotApplicable,
  getCredentialProviderFromSchema,
} from "../helpers";

// AutoPilot's shape: one provider, but a discriminator whose `platform` value
// is deliberately unmapped because that transport needs no credential.
const autopilotSchema: BlockIOCredentialsSubSchema = {
  type: "object",
  properties: {},
  credentials_provider: ["codex"],
  credentials_types: ["oauth2"],
  discriminator: "transport",
  discriminator_mapping: { codex_app_server: "codex" },
};

// A plain single-provider field with no discriminator at all.
const githubSchema: BlockIOCredentialsSubSchema = {
  type: "object",
  properties: {},
  credentials_provider: ["github"],
  credentials_types: ["api_key"],
};

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

  it("resolves a saved legacy credential when the discriminator is unset", () => {
    expect(getCredentialProviderFromSchema({}, autopilotSchema, "codex")).toBe(
      "codex",
    );
  });

  it("still returns the lone provider when no discriminator is declared", () => {
    expect(getCredentialProviderFromSchema({}, githubSchema)).toBe("github");
  });
});

describe("credentialNotApplicable", () => {
  it("is true when the selection maps to no provider", () => {
    // AutoPilot's `platform`: needs no credential, so the row should not
    // render at all — distinct from "unavailable" and from "still loading",
    // which a bare null could not express.
    expect(
      credentialNotApplicable({ transport: "platform" }, autopilotSchema),
    ).toBe(true);
  });

  it("is false when the selection maps to a provider", () => {
    expect(
      credentialNotApplicable(
        { transport: "codex_app_server" },
        autopilotSchema,
      ),
    ).toBe(false);
  });

  it("is true when neither a selection nor a legacy credential exists", () => {
    expect(credentialNotApplicable({}, autopilotSchema)).toBe(true);
  });

  it("is false when a legacy node has a saved credential", () => {
    expect(credentialNotApplicable({}, autopilotSchema, "codex")).toBe(false);
  });

  it("is false for a field with no discriminator at all", () => {
    expect(credentialNotApplicable({}, githubSchema)).toBe(false);
  });
});
