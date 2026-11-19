import { beforeEach, describe, expect, test } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { http, HttpResponse } from "msw";
import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import type { ProviderMetadata } from "@/app/api/__generated__/models/providerMetadata";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import {
  getGetV1ListCredentialsMockHandler,
  getGetV1ListProvidersMockHandler,
} from "@/app/api/__generated__/endpoints/integrations/integrations.msw";
import { getPostV2StoreABearerTokenForAnMcpServerMockHandler200 } from "@/app/api/__generated__/endpoints/mcp/mcp.msw";
import SettingsIntegrationsPage from "../page";
import {
  authProviders,
  discoveryRequest,
  oauthRequest,
  setupAuthFixtures,
  tokenRequest,
} from "./mcp-auth-fixtures";

beforeEach(() => {
  setupAuthFixtures();
  server.use(
    getGetV1ListProvidersMockHandler([
      ...authProviders,
      {
        name: "github",
        description: "Issues and PRs",
        supported_auth_types: ["api_key"],
      },
      {
        name: "mcp",
        description: "Connect any MCP server",
        supported_auth_types: ["oauth2"],
      },
    ]),
  );
});

function shippedCatalogProvider(name: string): ProviderMetadata {
  const catalog: ProviderMetadata[] = JSON.parse(
    readFileSync(
      resolve("../backend/backend/integrations/mcp_catalog.json"),
      "utf8",
    ),
  );
  const entry = catalog.find((item) => item.name === name);
  expect(entry).toBeDefined();
  return {
    ...entry!,
    service: entry!.mcp_server?.provider ?? name.replace(/^mcp_/, ""),
    service_name: entry!.display_name,
    service_icon: entry!.mcp_server?.icon_id ?? null,
  };
}

async function openPicker() {
  const buttons = await screen.findAllByRole("button", {
    name: /connect.*service/i,
  });
  fireEvent.click(buttons[0]);
  return screen.findByRole("dialog");
}

describe("SettingsIntegrationsPage — MCP catalogue", () => {
  test("shows PostHog branding and connection options from the shipped catalog", async () => {
    server.use(
      getGetV1ListProvidersMockHandler([shippedCatalogProvider("mcp_posthog")]),
    );

    render(<SettingsIntegrationsPage />);
    const row = await screen.findByRole("button", { name: /posthog/i });
    expect(row.querySelector("img")?.getAttribute("src")).toBe(
      "/integrations/posthog.png",
    );
    fireEvent.click(row);

    const dialog = await screen.findByRole("dialog");
    expect(
      (await within(dialog).findByAltText("PostHog logo")).getAttribute("src"),
    ).toBe("/integrations/posthog.png");
    const input =
      await within(dialog).findByLabelText<HTMLInputElement>("Server URL");
    expect(input.value).toBe("https://mcp.posthog.com/mcp");
    expect(input.readOnly).toBe(true);
    expect(within(dialog).getByRole("tab", { name: /sign in/i })).toBeDefined();
    expect(
      within(dialog).getByRole("tab", { name: /api token/i }),
    ).toBeDefined();
  });

  test("opens OpenSEO on its cloud server and lets self-hosters change the URL", async () => {
    server.use(
      getGetV1ListProvidersMockHandler([shippedCatalogProvider("mcp_openseo")]),
    );

    render(<SettingsIntegrationsPage />);
    const row = await screen.findByRole("button", { name: /openseo/i });
    expect(within(row).queryByText("Setup required")).toBeNull();
    fireEvent.click(row);

    const dialog = await screen.findByRole("dialog");
    const input =
      await within(dialog).findByLabelText<HTMLInputElement>("Server URL");
    expect(input.value).toBe("https://app.openseo.so/mcp");
    expect(input.readOnly).toBe(false);
    fireEvent.change(input, {
      target: { value: "https://openseo.example.workers.dev/mcp" },
    });
    expect(input.value).toBe("https://openseo.example.workers.dev/mcp");
    fireEvent.click(within(dialog).getByRole("button", { name: /^connect$/i }));
    await waitFor(() => {
      expect(oauthRequest).toHaveBeenCalledWith({
        server_url: "https://openseo.example.workers.dev/mcp",
      });
    });
  });

  test("AnySearch is one row under its brand name, and a pasted token connects its server", async () => {
    const connection: CredentialsMetaResponse = {
      id: "anysearch-token",
      provider: "mcp",
      type: "oauth2",
      title: "MCP: api.anysearch.com",
      scopes: null,
      username: null,
      host: null,
      is_managed: false,
      service: "anysearch",
      service_name: "AnySearch",
      service_icon: "anysearch",
    };
    let saved = false;
    server.use(
      getGetV1ListProvidersMockHandler([
        {
          name: "anysearch",
          description: "AI-native web search",
          supported_auth_types: ["api_key"],
          service: "anysearch",
          service_icon: "anysearch",
        },
        shippedCatalogProvider("mcp_anysearch"),
      ]),
      getGetV1ListCredentialsMockHandler(() => (saved ? [connection] : [])),
      getPostV2StoreABearerTokenForAnMcpServerMockHandler200(
        async ({ request }) => {
          tokenRequest(await request.json());
          saved = true;
          return connection;
        },
      ),
    );

    render(<SettingsIntegrationsPage />);
    const rows = await screen.findAllByRole("button", { name: /^anysearch/i });
    expect(rows).toHaveLength(1);
    expect(within(rows[0]).getByText("AnySearch")).toBeDefined();
    fireEvent.click(rows[0]);

    const dialog = await screen.findByRole("dialog");
    fireEvent.click(
      await within(dialog).findByRole("button", {
        name: "Use an AnySearch token instead",
      }),
    );
    const input =
      await within(dialog).findByLabelText<HTMLInputElement>("Server URL");
    expect(input.value).toBe("https://api.anysearch.com/mcp");
    fireEvent.change(within(dialog).getByPlaceholderText("Paste API token"), {
      target: { value: "anysearch-access-token" },
    });
    fireEvent.click(
      within(dialog).getByRole("button", { name: /save token/i }),
    );

    expect(await screen.findByText("api.anysearch.com")).toBeDefined();
    expect(
      screen.getByText(
        "AnySearch blocks in workflows need their own connection.",
      ),
    ).toBeDefined();
    expect(tokenRequest).toHaveBeenCalledWith({
      server_url: "https://api.anysearch.com/mcp",
      token: "Bearer anysearch-access-token",
    });
  });

  test("lists native providers and branded MCP entries together", async () => {
    render(<SettingsIntegrationsPage />);
    expect(
      await screen.findByRole("button", { name: /github.*issues and prs/i }),
    ).toBeDefined();
    const preset = await screen.findByRole("button", {
      name: /agentmail.*account/i,
    });
    expect(within(preset).queryByText("MCP", { exact: true })).toBeNull();
    expect(preset.querySelector("img")?.getAttribute("src")).toBe(
      "/integrations/agent_mail.png",
    );
    expect(screen.queryByText("Mcp Agentmail")).toBeNull();
  });

  test("search filters native and MCP providers by display name", async () => {
    render(<SettingsIntegrationsPage />);
    await screen.findByRole("button", { name: /agentmail.*account/i });
    fireEvent.change(screen.getByLabelText(/search integrations/i), {
      target: { value: "agentmail" },
    });
    await waitFor(() => {
      expect(
        screen.queryByRole("button", { name: /github.*issues and prs/i }),
      ).toBeNull();
    });
    expect(
      screen.getByRole("button", { name: /agentmail.*account/i }),
    ).toBeDefined();
    expect(
      screen.queryByRole("button", { name: /intercom.*account/i }),
    ).toBeNull();
  });

  test("a page preset locks its URL and uses generic MCP OAuth", async () => {
    render(<SettingsIntegrationsPage />);
    fireEvent.click(
      await screen.findByRole("button", { name: /agentmail.*account/i }),
    );
    const dialog = await screen.findByRole("dialog");
    const input =
      await within(dialog).findByLabelText<HTMLInputElement>("Server URL");
    expect(input.value).toBe("https://mcp.agentmail.example.com/mcp");
    expect(input.readOnly).toBe(true);
    expect(
      within(dialog).getByAltText("AgentMail logo").getAttribute("src"),
    ).toBe("/integrations/agent_mail.png");
    fireEvent.click(within(dialog).getByRole("button", { name: /^connect$/i }));
    await waitFor(() => {
      expect(oauthRequest).toHaveBeenCalledWith({
        server_url: "https://mcp.agentmail.example.com/mcp",
      });
    });
  });

  test("the service picker opens the same named MCP presets", async () => {
    render(<SettingsIntegrationsPage />);
    const dialog = await openPicker();
    fireEvent.click(
      await within(dialog).findByRole("button", {
        name: /agentmail.*account/i,
      }),
    );
    expect(
      (await within(dialog).findByLabelText<HTMLInputElement>("Server URL"))
        .value,
    ).toBe("https://mcp.agentmail.example.com/mcp");
  });

  test("generic MCP setup retains a blank editable URL", async () => {
    render(<SettingsIntegrationsPage />);
    const dialog = await openPicker();
    fireEvent.click(await within(dialog).findByText("Connect any MCP server"));
    const input =
      await within(dialog).findByLabelText<HTMLInputElement>("Server URL");
    expect(input.value).toBe("");
    expect(input.readOnly).toBe(false);
    expect(
      within(dialog).getByRole<HTMLButtonElement>("button", {
        name: /^connect$/i,
      }).disabled,
    ).toBe(true);
  });

  test.each([0, 2])(
    "public discovery reports %i tools without saving or reusing credentials",
    async (toolCount) => {
      server.use(
        http.post("*/api/mcp/discover-tools", async ({ request }) => {
          discoveryRequest(await request.json());
          return HttpResponse.json({
            tools: Array.from({ length: toolCount }, (_, i) => ({
              name: "tool_" + i,
              description: "Tool",
              input_schema: {},
            })),
          });
        }),
      );
      render(<SettingsIntegrationsPage />);
      fireEvent.click(
        await screen.findByRole("button", { name: /parallel.*account/i }),
      );
      const dialog = await screen.findByRole("dialog");
      fireEvent.click(
        await within(dialog).findByRole("button", {
          name: /check connection/i,
        }),
      );
      expect((await within(dialog).findByRole("status")).textContent).toBe(
        toolCount
          ? "2 tools available."
          : "Connected, but this server returned no tools.",
      );
      expect(
        within(dialog).getByText(/needs no sign-in, so nothing was saved/i),
      ).toBeDefined();
      expect(discoveryRequest).toHaveBeenCalledWith({
        server_url: "https://search.parallel.ai/mcp",
        use_saved_credentials: false,
      });
      expect(tokenRequest).not.toHaveBeenCalled();
      expect(oauthRequest).not.toHaveBeenCalled();
    },
  );
});
