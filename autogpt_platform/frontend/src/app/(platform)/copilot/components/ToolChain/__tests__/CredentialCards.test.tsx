import { render, screen } from "@/tests/integrations/test-utils";
import { cleanup } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import {
  credentialDetail,
  parseCredentialInventory,
  type CredentialItem,
} from "../credentialHelpers";
import type { ChainRow } from "../helpers";
import { getCatalogLabel } from "../toolCatalog";
import { ToolResult } from "../ToolResult";

function row(output: unknown, tool: string, input: unknown = {}): ChainRow {
  return {
    key: "tool",
    category: "integration",
    text: "Tool",
    state: "done",
    output,
    tool,
    input,
  };
}

const INVENTORY = {
  type: "credential_list",
  message: "The user has 2 connected credential(s) across 2 provider(s).",
  credentials: [
    {
      id: "cred-github",
      provider: "github",
      type: "oauth2",
      title: "GitHub",
      username: "octocat",
      scopes: ["repo", "read:org"],
      host: null,
      is_managed: false,
      service_name: "GitHub",
    },
    {
      id: "cred-mcp",
      provider: "mcp",
      type: "oauth2",
      title: "Linear MCP",
      username: null,
      scopes: [],
      host: "https://mcp.linear.app/mcp",
      is_managed: false,
    },
  ],
  providers: ["github", "mcp"],
  count: 2,
  provisioning_complete: true,
  ungranted_providers: [],
};

function credential(overrides: Partial<CredentialItem>): CredentialItem {
  return {
    id: "id",
    provider: "agent_mail",
    type: "api_key",
    title: null,
    username: null,
    host: null,
    scopes: [],
    isManaged: false,
    serviceName: null,
    ...overrides,
  };
}

describe("list_user_credentials result", () => {
  afterEach(cleanup);

  it("renders each connected integration with its account and auth type", () => {
    render(<ToolResult row={row(INVENTORY, "list_user_credentials")} />);

    expect(screen.getByText("GitHub")).toBeDefined();
    expect(screen.getByText("octocat · OAuth · 2 scopes")).toBeDefined();
    expect(screen.getByText("Mcp")).toBeDefined();
    expect(
      screen.getByText("Linear MCP · OAuth · https://mcp.linear.app/mcp"),
    ).toBeDefined();
  });

  it("renders the same card when reached through run_capability", () => {
    render(
      <ToolResult
        row={row(INVENTORY, "run_capability", {
          id: "tool:list_user_credentials",
          input: {},
        })}
      />,
    );

    expect(screen.getByText("octocat · OAuth · 2 scopes")).toBeDefined();
  });

  it("flags a partial inventory and lists integrations awaiting a grant", () => {
    render(
      <ToolResult
        row={row(
          {
            ...INVENTORY,
            credentials: [],
            count: 0,
            provisioning_complete: false,
            ungranted_providers: ["google_sheets"],
          },
          "list_user_credentials",
        )}
      />,
    );

    expect(screen.getByText(/may be\s+incomplete/)).toBeDefined();
    expect(screen.getByText("No connected integrations")).toBeDefined();
    expect(screen.getByText("Not granted to this expert")).toBeDefined();
    expect(screen.getByText("Google Sheets")).toBeDefined();
  });
});

describe("parseCredentialInventory", () => {
  it("rejects a payload without a credentials array", () => {
    expect(
      parseCredentialInventory({
        type: "credential_list",
        message: "Connected integrations are hidden on shared chats.",
      }),
    ).toBeNull();
    expect(parseCredentialInventory({ type: "error" })).toBeNull();
  });

  it("drops malformed entries instead of rendering them", () => {
    const inventory = parseCredentialInventory({
      type: "credential_list",
      credentials: [{ id: "x" }, "junk", INVENTORY.credentials[0]],
    });

    expect(inventory?.credentials.map((c) => c.id)).toEqual(["cred-github"]);
    expect(inventory?.provisioningComplete).toBe(true);
    expect(inventory?.ungrantedProviders).toEqual([]);
  });
});

describe("credentialDetail", () => {
  it("prefers the account name and marks managed credentials", () => {
    expect(
      credentialDetail(
        credential({ title: "AgentMail", isManaged: true, scopes: ["send"] }),
      ),
    ).toBe("AgentMail · API key · 1 scope · Managed by AutoGPT");
  });
});

describe("list_user_credentials chain label", () => {
  it("names the provider filter when one is given", () => {
    expect(
      getCatalogLabel("list_user_credentials", { provider: "github" }, "done")
        ?.text,
    ).toBe("Checked connected integrations for github");
    expect(getCatalogLabel("list_user_credentials", {}, "running")?.text).toBe(
      "Checking connected integrations…",
    );
  });
});
