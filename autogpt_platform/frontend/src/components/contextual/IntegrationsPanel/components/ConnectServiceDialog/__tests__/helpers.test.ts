import { describe, expect, test } from "vitest";

import type { ProviderMetadata } from "@/app/api/__generated__/models/providerMetadata";

import {
  filterConnectableProviders,
  toConnectableProviders,
  type ConnectableProvider,
} from "../helpers";

function makeMeta(overrides: Partial<ProviderMetadata> = {}): ProviderMetadata {
  return {
    name: "github",
    description: "Issues and PRs",
    supported_auth_types: ["oauth2", "api_key"],
    ...overrides,
  };
}

describe("toConnectableProviders", () => {
  test("formats provider name and preserves description and supported types", () => {
    const result = toConnectableProviders([
      makeMeta({ name: "github", description: "Issues and PRs" }),
    ]);
    expect(result).toEqual([
      {
        id: "github",
        name: "GitHub",
        description: "Issues and PRs",
        supportedAuthTypes: ["oauth2", "api_key"],
      },
    ]);
  });

  test("dedupes by name, sorts alphabetically by display name", () => {
    const result = toConnectableProviders([
      makeMeta({ name: "openai", supported_auth_types: ["api_key"] }),
      makeMeta({ name: "github" }),
      makeMeta({ name: "openai" }), // duplicate — should be dropped
    ]);
    expect(result.map((p) => p.id)).toEqual(["github", "openai"]);
  });

  test("filters unknown auth types out and tolerates missing supported_auth_types", () => {
    const result = toConnectableProviders([
      makeMeta({
        name: "github",
        // The `as never` cast simulates an unexpected value coming over the
        // wire — we want unknowns ignored, not crashing the whole list.
        supported_auth_types: ["oauth2", "weird_thing" as never],
      }),
      makeMeta({ name: "openai", supported_auth_types: undefined }),
    ]);
    const github = result.find((p) => p.id === "github");
    const openai = result.find((p) => p.id === "openai");
    expect(github?.supportedAuthTypes).toEqual(["oauth2"]);
    expect(openai?.supportedAuthTypes).toEqual([]);
  });

  test("merges ChatGPT sign-in into OpenAI while preserving backend auth targets", () => {
    // `display_alias` and the merged description are the server's answers.
    // They used to be literals here keyed on "codex", which meant the client
    // decided which card a credential belonged on -- and could not be told
    // when a provider was added.
    const result = toConnectableProviders([
      makeMeta({
        name: "codex",
        description: "Use your ChatGPT plan",
        display_alias: "openai",
        supported_auth_types: ["oauth2"],
      }),
      makeMeta({
        name: "openai",
        description: "GPT models and embeddings or your ChatGPT subscription",
        supported_auth_types: ["api_key"],
      }),
    ]);

    expect(result).toHaveLength(1);
    expect(result[0]).toMatchObject({
      id: "openai",
      name: "OpenAI",
      description: "GPT models and embeddings or your ChatGPT subscription",
      supportedAuthTypes: ["oauth2", "api_key"],
      authProviderByType: { oauth2: "codex" },
      searchTerms: ["codex"],
    });
  });

  test("keeps a subscription that would collide as its own card", () => {
    // GitHub already owns the OAuth tab on its card. The server says so by
    // sending no alias, and this passes with no code here knowing that
    // GitHub Copilot exists -- which is the point of the alias travelling.
    const result = toConnectableProviders([
      makeMeta({
        name: "github_copilot",
        description: "Your GitHub Copilot subscription",
        supported_auth_types: ["oauth2"],
      }),
      makeMeta({
        name: "github",
        description: "Issues, pull requests, repositories",
        supported_auth_types: ["oauth2", "api_key"],
      }),
    ]);

    expect(result.map((p) => p.id)).toEqual(["github", "github_copilot"]);
    // Each card keeps its own OAuth target, so neither sign-in is unreachable.
    expect(result[0].authProviderByType?.oauth2).toBeUndefined();
    expect(result[1].authProviderByType?.oauth2).toBeUndefined();
  });
});

describe("filterConnectableProviders", () => {
  const providers: ConnectableProvider[] = [
    {
      id: "github",
      name: "GitHub",
      description: "Issues and PRs",
      supportedAuthTypes: ["oauth2", "api_key"],
    },
    {
      id: "acai-juice",
      name: "Açaí",
      description: "Bowls",
      supportedAuthTypes: ["api_key"],
    },
    {
      id: "linear",
      name: "Linear",
      description: "Project tracking",
      supportedAuthTypes: ["oauth2"],
    },
  ];

  test("returns the full list for a blank query", () => {
    expect(filterConnectableProviders(providers, "")).toEqual(providers);
    expect(filterConnectableProviders(providers, "   ")).toEqual(providers);
  });

  test("matches by name, slug, and description", () => {
    expect(
      filterConnectableProviders(providers, "github").map((p) => p.id),
    ).toEqual(["github"]);
    expect(
      filterConnectableProviders(providers, "linear").map((p) => p.id),
    ).toEqual(["linear"]);
    expect(
      filterConnectableProviders(providers, "tracking").map((p) => p.id),
    ).toEqual(["linear"]);
  });

  test("matches accented names against unaccented queries (NFKD)", () => {
    // The id is "acai-juice" so this query can only match by stripping
    // diacritics from the name "Açaí".
    expect(
      filterConnectableProviders(providers, "acai").map((p) => p.id),
    ).toEqual(["acai-juice"]);
  });

  test("returns an empty list when nothing matches", () => {
    expect(filterConnectableProviders(providers, "nope")).toEqual([]);
  });

  test("matches presentation aliases", () => {
    const openai = toConnectableProviders([
      makeMeta({
        name: "codex",
        display_alias: "openai",
        supported_auth_types: ["oauth2"],
      }),
      makeMeta({
        name: "openai",
        description: "GPT models or your ChatGPT subscription",
        supported_auth_types: ["api_key"],
      }),
    ]);

    expect(filterConnectableProviders(openai, "chatgpt")).toEqual(openai);
    expect(filterConnectableProviders(openai, "codex")).toEqual(openai);
  });
});
