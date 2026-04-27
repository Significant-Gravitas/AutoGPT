import { describe, expect, test } from "vitest";

import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";

import {
  filterProviders,
  formatMaskedValue,
  formatProviderName,
  groupCredentialsByProvider,
  type CredentialView,
  type ProviderGroupView,
} from "../helpers";

function makeCred(
  overrides: Partial<CredentialsMetaResponse> = {},
): CredentialsMetaResponse {
  return {
    id: "id-1",
    provider: "github",
    type: "api_key",
    title: "Mock",
    scopes: null,
    username: null,
    host: null,
    is_managed: false,
    ...overrides,
  };
}

function makeView(overrides: Partial<CredentialView> = {}): CredentialView {
  return {
    id: "id-1",
    provider: "github",
    type: "api_key",
    title: "Mock",
    username: null,
    host: null,
    isManaged: false,
    ...overrides,
  };
}

describe("formatProviderName", () => {
  test("uses curated override when one exists", () => {
    expect(formatProviderName("github")).toBe("GitHub");
    expect(formatProviderName("d_id")).toBe("D-ID");
    expect(formatProviderName("twitter")).toBe("X");
  });

  test("title-cases unknown snake_case slugs", () => {
    expect(formatProviderName("foo_bar")).toBe("Foo Bar");
    expect(formatProviderName("hello-world")).toBe("Hello World");
  });

  test("returns empty string for non-string input — guards against provider.split TypeError", () => {
    expect(formatProviderName(undefined)).toBe("");
    expect(formatProviderName(null)).toBe("");
    expect(formatProviderName(42)).toBe("");
    expect(formatProviderName("")).toBe("");
  });
});

describe("formatMaskedValue", () => {
  test("prefers username when present", () => {
    expect(formatMaskedValue(makeView({ username: "abhi" }))).toBe(
      "Username: abhi",
    );
  });

  test("falls back to host when no username", () => {
    expect(formatMaskedValue(makeView({ host: "example.com" }))).toBe(
      "example.com",
    );
  });

  test("returns type-specific copy for credentials with no username/host", () => {
    expect(formatMaskedValue(makeView({ type: "api_key" }))).toBe(
      "API key configured",
    );
    expect(formatMaskedValue(makeView({ type: "oauth2" }))).toBe(
      "Connected via OAuth",
    );
    expect(formatMaskedValue(makeView({ type: "user_password" }))).toBe(
      "Username/password set",
    );
  });
});

describe("groupCredentialsByProvider", () => {
  test("groups credentials by provider id and sorts groups by display name", () => {
    const groups = groupCredentialsByProvider([
      makeCred({ id: "a", provider: "openai" }),
      makeCred({ id: "b", provider: "github", title: "Personal" }),
      makeCred({ id: "c", provider: "github", title: "Work" }),
    ]);
    expect(groups.map((g) => g.id)).toEqual(["github", "openai"]);
    expect(groups[0].credentials.map((c) => c.id)).toEqual(["b", "c"]);
  });

  test("falls back to formatted provider name when title is missing", () => {
    const groups = groupCredentialsByProvider([
      makeCred({ provider: "github", title: null }),
    ]);
    expect(groups[0].credentials[0].title).toBe("GitHub");
  });
});

describe("filterProviders", () => {
  const providers: ProviderGroupView[] = [
    {
      id: "github",
      name: "GitHub",
      credentials: [makeView({ id: "g1", title: "Personal" })],
    },
    {
      id: "acai",
      name: "Açaí",
      credentials: [makeView({ id: "a1", title: "Smoothie" })],
    },
    {
      id: "openai",
      name: "OpenAI",
      credentials: [
        makeView({ id: "o1", title: "Work key", username: "Abhi" }),
      ],
    },
  ];

  test("returns the full list when query is blank", () => {
    expect(filterProviders(providers, "")).toEqual(providers);
    expect(filterProviders(providers, "   ")).toEqual(providers);
  });

  test("matches by provider name case-insensitively", () => {
    const result = filterProviders(providers, "github");
    expect(result.map((p) => p.id)).toEqual(["github"]);
  });

  test("matches accented provider names against unaccented queries (NFKD)", () => {
    const result = filterProviders(providers, "acai");
    expect(result.map((p) => p.id)).toEqual(["acai"]);
  });

  test("matches accented queries against unaccented names (NFKD)", () => {
    const ascii: ProviderGroupView[] = [
      { id: "acai", name: "Acai", credentials: [makeView({ id: "x" })] },
    ];
    expect(filterProviders(ascii, "Açaí").map((p) => p.id)).toEqual(["acai"]);
  });

  test("matches inside credential titles when provider name does not match", () => {
    const result = filterProviders(providers, "personal");
    expect(result.map((p) => p.id)).toEqual(["github"]);
    expect(result[0].credentials.map((c) => c.id)).toEqual(["g1"]);
  });

  test("matches inside credential usernames", () => {
    const result = filterProviders(providers, "abhi");
    expect(result.map((p) => p.id)).toEqual(["openai"]);
  });

  test("returns empty list when nothing matches", () => {
    expect(filterProviders(providers, "no-such-thing")).toEqual([]);
  });
});
