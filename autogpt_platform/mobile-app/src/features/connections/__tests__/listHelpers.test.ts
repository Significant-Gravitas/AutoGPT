import type { CredentialMeta, ExpertSetupItem, ProviderInfo } from "@/lib/api/integrations";

import {
  accountRows,
  actionLabel,
  expertNeeds,
  grantableRows,
  grantedRows,
  matchesQuery,
  providerDisplayName,
  serviceFor,
  toServices,
} from "../listHelpers";

function credential(extra: Partial<CredentialMeta>): CredentialMeta {
  return {
    id: "c1",
    provider: "github",
    type: "oauth2",
    title: null,
    username: null,
    scopes: null,
    host: null,
    isSystem: false,
    serviceName: null,
    serviceIcon: null,
    ...extra,
  };
}

function provider(extra: Partial<ProviderInfo>): ProviderInfo {
  return {
    name: "github",
    displayName: null,
    description: null,
    authTypes: [],
    service: extra.name ?? "github",
    serviceName: null,
    mcpServer: null,
    ...extra,
  };
}

function setupItem(extra: Partial<ExpertSetupItem>): ExpertSetupItem {
  return {
    expertID: "e1",
    workflowName: "Weekly report",
    providers: ["hubspot"],
    resolution: "connect",
    credentialID: null,
    ...extra,
  };
}

describe("providerDisplayName", () => {
  it("uses the web's names, else title-cases the slug", () => {
    expect(providerDisplayName("github")).toBe("GitHub");
    expect(providerDisplayName("google_calendar")).toBe("Google Calendar");
  });
});

describe("accountRows", () => {
  it("drops platform keys, names the service and trims its prefix from the title", () => {
    const rows = accountRows([
      credential({ id: "a", provider: "notion", title: "Notion: Team space" }),
      credential({ id: "b", provider: "openai", title: "Use credits for OpenAI" }),
      credential({ id: "c", provider: "anthropic", isSystem: true }),
      credential({ id: "d", provider: "github", title: null, username: "octocat" }),
    ]);
    expect(rows).toEqual([
      { id: "d", name: "GitHub", detail: "octocat", logoID: "github" },
      { id: "a", name: "Notion", detail: "Team space", logoID: "notion" },
    ]);
  });

  it("prefers the backend's service name", () => {
    const [row] = accountRows([
      credential({ provider: "mcp", serviceName: "Linear", title: "MCP: mcp.linear.app" }),
    ]);
    expect(row).toMatchObject({ name: "Linear", detail: "mcp.linear.app" });
  });

  it("uses the backend's logo, else the provider or the MCP server's host", () => {
    const rows = accountRows([
      credential({ id: "a", provider: "gmail", serviceIcon: "google", serviceName: "Gmail" }),
      credential({ id: "b", provider: "mcp", host: "https://mcp.linear.app/mcp", serviceName: "Linear" }),
      credential({ id: "c", provider: "github" }),
    ]);
    expect(rows.map((row) => row.logoID)).toEqual(["github", "google", "linear"]);
  });
});

describe("grantableRows and grantedRows", () => {
  it("leaves out accounts the expert can already use", () => {
    const credentials = [credential({ id: "a" }), credential({ id: "b", provider: "linear" })];
    expect(grantableRows(credentials, ["a"]).map((row) => row.id)).toEqual(["b"]);
  });

  it("reads the expert's accounts", () => {
    expect(
      grantedRows([
        {
          credentialID: "x",
          provider: "slack",
          title: "Slack: Acme",
          type: "oauth2",
          serviceName: null,
          serviceIcon: null,
        },
      ]),
    ).toEqual([{ id: "x", name: "Slack", detail: "Acme", logoID: "slack" }]);
  });
});

describe("expertNeeds", () => {
  it("keeps this expert's connect and allow items, one row per service", () => {
    const needs = expertNeeds(
      [
        setupItem({ workflowName: "Weekly report" }),
        setupItem({ workflowName: "Daily digest" }),
        setupItem({ resolution: "allow", providers: ["github"], credentialID: "c9", workflowName: "Triage" }),
        setupItem({ resolution: "inputs" }),
        setupItem({ resolution: "workflow" }),
        setupItem({ expertID: "other" }),
        setupItem({ resolution: "allow", credentialID: null }),
      ],
      "e1",
    );
    expect(needs).toEqual([
      {
        key: "connect:hubspot",
        resolution: "connect",
        providers: ["hubspot"],
        credentialID: null,
        workflowNames: ["Weekly report", "Daily digest"],
      },
      {
        key: "allow:c9",
        resolution: "allow",
        providers: ["github"],
        credentialID: "c9",
        workflowNames: ["Triage"],
      },
    ]);
  });
});

describe("toServices", () => {
  it("picks the way in for each provider", () => {
    const services = toServices([
      provider({ name: "github", authTypes: ["oauth2", "api_key"] }),
      provider({ name: "openweathermap", authTypes: ["api_key"] }),
      provider({ name: "smtp", authTypes: ["user_password"] }),
      provider({ name: "codex", authTypes: ["oauth2"] }),
      provider({ name: "microsoft", authTypes: ["device_code"] }),
      provider({ name: "empty", authTypes: [] }),
      provider({ name: "mcp", authTypes: ["oauth2"] }),
    ]);
    expect(services.map((service) => [service.key, service.action])).toEqual([
      ["codex", { kind: "web" }],
      ["github", { kind: "oauth", provider: "github" }],
      ["microsoft", { kind: "web" }],
      ["openweathermap", { kind: "sheet", sheet: "api_key", provider: "openweathermap" }],
      ["smtp", { kind: "sheet", sheet: "user_password", provider: "smtp" }],
    ]);
  });

  it("files an MCP server under its block provider, which leads", () => {
    const services = toServices([
      provider({ name: "notion", authTypes: ["api_key"] }),
      provider({
        name: "notion_mcp",
        service: "notion",
        serviceName: "Notion",
        mcpServer: { serverURL: "https://mcp.notion.com/mcp", connectionMode: "hosted", authMethods: ["oauth"] },
      }),
    ]);
    expect(services).toEqual([
      {
        key: "notion",
        name: "Notion",
        description: null,
        providers: ["notion", "notion_mcp"],
        action: { kind: "sheet", sheet: "api_key", provider: "notion" },
      },
    ]);
  });

  it("uses a hosted server's sign-in or token, and sends the rest to the web", () => {
    const server = (name: string, mcp: Partial<NonNullable<ProviderInfo["mcpServer"]>>) =>
      provider({
        name,
        service: name,
        displayName: name.toUpperCase(),
        mcpServer: { serverURL: `https://${name}.example.com`, connectionMode: "hosted", authMethods: ["oauth"], ...mcp },
      });
    const services = toServices([
      server("aa", {}),
      server("bb", { authMethods: ["bearer"] }),
      server("cc", { connectionMode: "custom", serverURL: null }),
    ]);
    expect(services.map((service) => service.action)).toEqual([
      { kind: "mcp", serverURL: "https://aa.example.com" },
      { kind: "mcp_token", serverURL: "https://bb.example.com" },
      { kind: "web" },
    ]);
    expect(services[0].name).toBe("AA");
  });

  it("finds a setup item's service by any of its providers", () => {
    const services = toServices([provider({ name: "hubspot", authTypes: ["oauth2"] })]);
    expect(serviceFor(services, ["other", "hubspot"])?.key).toBe("hubspot");
    expect(serviceFor(services, ["other"])).toBeNull();
  });
});

describe("actionLabel", () => {
  it("names each way in", () => {
    expect(actionLabel({ kind: "oauth", provider: "github" })).toBe("Sign in");
    expect(actionLabel({ kind: "sheet", sheet: "api_key", provider: "x" })).toBe("Add key");
    expect(actionLabel({ kind: "sheet", sheet: "user_password", provider: "x" })).toBe("Add login");
    expect(actionLabel({ kind: "mcp_token", serverURL: "https://x" })).toBe("Add token");
    expect(actionLabel({ kind: "web" })).toBe("Open");
  });
});

describe("matchesQuery", () => {
  it("ignores case, accents and blank queries", () => {
    expect(matchesQuery(["Café Notes"], "cafe")).toBe(true);
    expect(matchesQuery(["GitHub", null], "HUB")).toBe(true);
    expect(matchesQuery(["GitHub"], "  ")).toBe(true);
    expect(matchesQuery(["GitHub"], "slack")).toBe(false);
  });
});
