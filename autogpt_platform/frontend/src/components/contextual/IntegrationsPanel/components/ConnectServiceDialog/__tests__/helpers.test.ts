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
  test("merges a vendor's block provider and its sign-in into one service", () => {
    const mcpServer = {
      server_url: "https://mcp.notion.com/mcp",
      documentation_url: "https://developers.notion.com/docs/mcp",
      setup_instructions: "Sign in to your Notion workspace.",
      connection_mode: "hosted" as const,
      auth_methods: ["oauth" as const],
      icon_id: "notion",
    };
    const block = makeMeta({
      name: "notion",
      description: "Docs and databases",
      service: "notion",
      service_name: null,
      service_icon: "notion",
    });
    const preset = {
      name: "mcp_notion",
      display_name: "Notion",
      description: "Search and edit workspace content",
      supported_auth_types: [],
      mcp_server: mcpServer,
      service: "notion",
      service_name: "Notion",
      service_icon: "notion",
    };
    const merged = {
      id: "notion",
      serviceId: "notion",
      name: "Notion",
      description: "Docs and databases",
      supportedAuthTypes: ["oauth2", "api_key"],
      mcpServer,
      iconId: "notion",
    };

    const blockFirst = toConnectableProviders([block, preset]);
    expect(blockFirst).toHaveLength(1);
    expect(blockFirst[0]).toMatchObject(merged);

    const catalogFirst = toConnectableProviders([preset, block]);
    expect(catalogFirst).toHaveLength(1);
    expect(catalogFirst[0]).toMatchObject(merged);
  });

  test("keeps a service reached only through its sign-in under its catalog name", () => {
    const result = toConnectableProviders([
      {
        name: "mcp_sentry",
        display_name: "Sentry",
        supported_auth_types: [],
        service: "sentry",
        service_name: "Sentry",
        service_icon: "sentry",
        mcp_server: {
          server_url: "https://mcp.sentry.dev/mcp",
          documentation_url: "https://docs.sentry.io",
          setup_instructions: "Sign in to Sentry.",
          connection_mode: "hosted",
          auth_methods: ["oauth"],
        },
      },
    ]);
    expect(result[0]).toMatchObject({
      id: "mcp_sentry",
      serviceId: "sentry",
      name: "Sentry",
      supportedAuthTypes: [],
    });
  });

  test("formats provider name and preserves description and supported types", () => {
    const result = toConnectableProviders([
      makeMeta({ name: "github", description: "Issues and PRs" }),
    ]);
    expect(result).toEqual([
      {
        id: "github",
        serviceId: "github",
        name: "GitHub",
        description: "Issues and PRs",
        supportedAuthTypes: ["oauth2", "api_key"],
        iconId: "github",
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
    const result = toConnectableProviders([
      makeMeta({
        name: "codex",
        description: "Use your ChatGPT plan",
        supported_auth_types: ["oauth2"],
      }),
      makeMeta({
        name: "openai",
        description: "GPT models and embeddings",
        supported_auth_types: ["api_key"],
      }),
    ]);

    expect(result).toHaveLength(1);
    expect(result[0]).toMatchObject({
      id: "openai",
      name: "OpenAI",
      description: "OpenAI models via API key or your ChatGPT subscription",
      supportedAuthTypes: ["oauth2", "api_key"],
      authProviderByType: { oauth2: "codex" },
      searchTerms: ["codex"],
    });
  });
});

describe("filterConnectableProviders", () => {
  const providers: ConnectableProvider[] = [
    {
      id: "github",
      serviceId: "github",
      name: "GitHub",
      description: "Issues and PRs",
      supportedAuthTypes: ["oauth2", "api_key"],
    },
    {
      id: "acai-juice",
      serviceId: "acai-juice",
      name: "Açaí",
      description: "Bowls",
      supportedAuthTypes: ["api_key"],
    },
    {
      id: "linear",
      serviceId: "linear",
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
      makeMeta({ name: "codex", supported_auth_types: ["oauth2"] }),
      makeMeta({ name: "openai", supported_auth_types: ["api_key"] }),
    ]);

    expect(filterConnectableProviders(openai, "chatgpt")).toEqual(openai);
    expect(filterConnectableProviders(openai, "codex")).toEqual(openai);
  });
});
