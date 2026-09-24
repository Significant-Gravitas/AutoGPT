import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import { describe, expect, it } from "vitest";
import {
  connectedIntegrationsFromCredentials,
  filterIntegrationMentions,
  insertIntegrationMention,
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

describe("connectedIntegrationsFromCredentials", () => {
  it("keeps each account using its credential name", () => {
    const result = connectedIntegrationsFromCredentials([
      credential("google", {
        title: "Work Gmail",
        username: "work@example.com",
      }),
      credential("google", {
        title: "Personal Gmail",
        username: "me@example.com",
      }),
    ]);
    expect(result).toEqual([
      {
        credentialId: "google-Personal Gmail",
        provider: "google",
        providerName: "Google",
        name: "Personal Gmail",
        username: "me@example.com",
        token: "[Personal Gmail](credential://google/google-Personal%20Gmail)",
      },
      {
        credentialId: "google-Work Gmail",
        provider: "google",
        providerName: "Google",
        name: "Work Gmail",
        username: "work@example.com",
        token: "[Work Gmail](credential://google/google-Work%20Gmail)",
      },
    ]);
  });

  it("falls back to username, then provider when the title is blank", () => {
    const result = connectedIntegrationsFromCredentials([
      credential("google", { title: "  ", username: "me@example.com" }),
      credential("github", { title: null }),
    ]);
    expect(result.map((m) => m.name)).toEqual(["GitHub", "me@example.com"]);
  });

  it("disambiguates duplicate names independently of API ordering", () => {
    const credentials = [
      credential("google", { id: "account-1", title: "Work" }),
      credential("google", { id: "account-2", title: "Work" }),
    ];
    const result = connectedIntegrationsFromCredentials(credentials);
    expect(new Set(result.map((m) => m.token)).size).toBe(2);
    expect(
      connectedIntegrationsFromCredentials([...credentials].reverse()),
    ).toEqual(result);
  });

  it("skips platform credit credentials, which are not something to address", () => {
    const result = connectedIntegrationsFromCredentials([
      credential("openai", { title: "Use Credits for OpenAI" }),
      credential("anthropic", { title: "System key" }),
      credential("slack"),
    ]);
    expect(result.map((mention) => mention.provider)).toEqual(["slack"]);
  });

  it("keeps Codex and OpenAI credentials separate with the OpenAI logo", () => {
    const result = connectedIntegrationsFromCredentials([
      credential("codex", { type: "oauth2", title: "Codex login" }),
      credential("openai", { title: "API key" }),
    ]);
    expect(result.map((m) => [m.credentialId, m.provider, m.token])).toEqual([
      [
        "openai-API key",
        "openai",
        "[API key](credential://openai/openai-API%20key)",
      ],
      [
        "codex-Codex login",
        "codex",
        "[Codex login](credential://codex/codex-Codex%20login)",
      ],
    ]);
  });
});

describe("filterIntegrationMentions", () => {
  const integrations = connectedIntegrationsFromCredentials([
    credential("google_maps", { title: "Travel", username: "me@example.com" }),
    credential("github", { title: "GitHub" }),
    credential("hubspot", { title: "HubSpot" }),
  ]);

  it("returns everything for an empty query", () => {
    expect(filterIntegrationMentions(integrations, "")).toEqual(integrations);
  });

  it("matches the display name case-insensitively and ignoring spaces", () => {
    expect(
      filterIntegrationMentions(integrations, "googlem").map((m) => m.name),
    ).toEqual(["Travel"]);
    expect(
      filterIntegrationMentions(integrations, "HUB").map((m) => m.name),
    ).toEqual(["GitHub", "HubSpot"]);
  });

  it("matches account names and usernames", () => {
    expect(
      filterIntegrationMentions(integrations, "TRAVEL").map((m) => m.name),
    ).toEqual(["Travel"]);
    expect(
      filterIntegrationMentions(integrations, "me@example").map((m) => m.name),
    ).toEqual(["Travel"]);
  });

  it("also matches the provider slug", () => {
    expect(
      filterIntegrationMentions(integrations, "_maps").map((m) => m.provider),
    ).toEqual(["google_maps"]);
  });
});

describe("insertIntegrationMention", () => {
  const google = connectedIntegrationsFromCredentials([
    credential("google", { title: "Work Gmail" }),
  ])[0];

  it("replaces the @query with the token and parks the caret after the existing space", () => {
    expect(
      insertIntegrationMention(
        "check @goo for me",
        { start: 6, end: 10 },
        google,
      ),
    ).toEqual({
      value: `check ${google.token} for me`,
      caret: `check ${google.token} `.length,
    });
  });

  it("adds a space when the mention ends the prompt", () => {
    expect(
      insertIntegrationMention("check @goo", { start: 6, end: 10 }, google),
    ).toEqual({
      value: `check ${google.token} `,
      caret: `check ${google.token} `.length,
    });
  });

  it("works when the mention is the whole prompt so far", () => {
    expect(insertIntegrationMention("@", { start: 0, end: 1 }, google)).toEqual(
      { value: `${google.token} `, caret: google.token.length + 1 },
    );
  });
});
