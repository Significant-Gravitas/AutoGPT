import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import { describe, expect, it } from "vitest";
import {
  connectedIntegrationsFromCredentials,
  filterIntegrationMentions,
  insertIntegrationMention,
  integrationMentionToken,
} from "../helpers";

function credential(
  provider: string,
  overrides: Partial<CredentialsMetaResponse> = {},
): CredentialsMetaResponse {
  return {
    id: `${provider}-${overrides.title ?? "cred"}`,
    provider,
    type: "api_key",
    title: `${provider} account`,
    scopes: null,
    username: null,
    ...overrides,
  };
}

describe("integrationMentionToken", () => {
  it("prefixes the name with @ and drops the spaces so the token stays one word", () => {
    expect(integrationMentionToken("Google Maps")).toBe("@GoogleMaps");
    expect(integrationMentionToken("GitHub")).toBe("@GitHub");
  });
});

describe("connectedIntegrationsFromCredentials", () => {
  it("returns one named mention per provider, sorted by display name", () => {
    const result = connectedIntegrationsFromCredentials([
      credential("notion"),
      credential("google", { title: "work" }),
      credential("google", { title: "personal" }),
      credential("github"),
    ]);
    expect(result).toEqual([
      { provider: "github", name: "GitHub", token: "@GitHub" },
      { provider: "google", name: "Google", token: "@Google" },
      { provider: "notion", name: "Notion", token: "@Notion" },
    ]);
  });

  it("skips platform credit credentials, which are not something to address", () => {
    const result = connectedIntegrationsFromCredentials([
      credential("openai", { title: "Use Credits for OpenAI" }),
      credential("anthropic", { title: "System key" }),
      credential("slack"),
    ]);
    expect(result.map((mention) => mention.provider)).toEqual(["slack"]);
  });

  it("folds the codex login into the OpenAI integration", () => {
    const result = connectedIntegrationsFromCredentials([
      credential("codex", { type: "oauth2" }),
      credential("openai"),
    ]);
    expect(result).toEqual([
      { provider: "openai", name: "OpenAI", token: "@OpenAI" },
    ]);
  });
});

describe("filterIntegrationMentions", () => {
  const integrations = connectedIntegrationsFromCredentials([
    credential("google_maps"),
    credential("github"),
    credential("hubspot"),
  ]);

  it("returns everything for an empty query", () => {
    expect(filterIntegrationMentions(integrations, "")).toEqual(integrations);
  });

  it("matches the display name case-insensitively and ignoring spaces", () => {
    expect(
      filterIntegrationMentions(integrations, "googlem").map((m) => m.name),
    ).toEqual(["Google Maps"]);
    expect(
      filterIntegrationMentions(integrations, "HUB").map((m) => m.name),
    ).toEqual(["GitHub", "HubSpot"]);
  });

  it("also matches the provider slug", () => {
    expect(
      filterIntegrationMentions(integrations, "_maps").map((m) => m.provider),
    ).toEqual(["google_maps"]);
  });
});

describe("insertIntegrationMention", () => {
  const google = { provider: "google", name: "Google", token: "@Google" };

  it("replaces the @query with the token and parks the caret after the existing space", () => {
    expect(
      insertIntegrationMention(
        "check @goo for me",
        { start: 6, end: 10 },
        google,
      ),
    ).toEqual({
      value: "check @Google for me",
      caret: "check @Google ".length,
    });
  });

  it("adds a space when the mention ends the prompt", () => {
    expect(
      insertIntegrationMention("check @goo", { start: 6, end: 10 }, google),
    ).toEqual({ value: "check @Google ", caret: "check @Google ".length });
  });

  it("works when the mention is the whole prompt so far", () => {
    expect(insertIntegrationMention("@", { start: 0, end: 1 }, google)).toEqual(
      { value: "@Google ", caret: 8 },
    );
  });
});
