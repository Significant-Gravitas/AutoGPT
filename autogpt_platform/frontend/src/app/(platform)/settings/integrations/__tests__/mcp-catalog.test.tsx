import { beforeEach, describe, expect, test, vi } from "vitest";
import { http, HttpResponse } from "msw";

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
import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import type { ProviderMetadata } from "@/app/api/__generated__/models/providerMetadata";

import SettingsIntegrationsPage from "../page";

type CatalogProvider = ProviderMetadata & {
  display_name?: string;
  mcp_server?: {
    server_url: string | null;
    documentation_url: string;
    setup_instructions: string;
    connection_mode: "hosted" | "custom" | "unavailable";
    auth_mode: "oauth" | "token" | "none" | "unknown";
    icon_id: string | null;
  };
};

function preset(
  slug: string,
  name: string,
  serverURL: string | null,
  authMode: "oauth" | "token" | "none" | "unknown" = "oauth",
): CatalogProvider {
  return {
    name: `mcp_${slug}`,
    display_name: name,
    description: `${name} official MCP integration`,
    supported_auth_types: ["oauth2"],
    mcp_server: {
      server_url: serverURL,
      documentation_url: `https://docs.example.com/${slug}`,
      setup_instructions: serverURL
        ? `Connect to ${name} with your existing account.`
        : "Run this server locally and expose a supported remote endpoint.",
      connection_mode: serverURL ? "hosted" : "unavailable",
      auth_mode: authMode,
      icon_id: null,
    },
  };
}

const providers: CatalogProvider[] = [
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
  preset("notion", "Notion", "https://mcp.notion.com/mcp"),
  preset("superme", "SuperMe", "https://mcp.superme.ai", "token"),
  preset("parallel", "Parallel", "https://search.parallel.ai/mcp", "none"),
  preset("shadcn_ui", "shadcn/ui", null, "unknown"),
];

const storedCredential: CredentialsMetaResponse = {
  id: "catalog-credential",
  provider: "mcp",
  type: "oauth2",
  title: "SuperMe account",
  scopes: null,
  username: null,
  host: "https://mcp.superme.ai",
  is_managed: false,
};
const oauthRequest = vi.fn();
const tokenRequest = vi.fn();
const discoveryRequest = vi.fn();
const providerCredentialRequest = vi.fn();
let credentials: CredentialsMetaResponse[] = [];

beforeEach(() => {
  vi.clearAllMocks();
  credentials = [];
  server.use(
    getGetV1ListCredentialsMockHandler(() => credentials),
    getGetV1ListProvidersMockHandler(providers),
    http.post("*/api/mcp/oauth/login", async ({ request }) => {
      oauthRequest(await request.json());
      return HttpResponse.json(
        { detail: "Sign-in is unavailable in this test" },
        { status: 503 },
      );
    }),
    http.post("*/api/mcp/discover-tools", async ({ request }) => {
      discoveryRequest(await request.json());
      return HttpResponse.json({
        tools: [
          { name: "search", description: "Search", input_schema: {} },
          { name: "fetch", description: "Fetch", input_schema: {} },
        ],
      });
    }),
    http.post("*/api/mcp/token", async ({ request }) => {
      tokenRequest(await request.json());
      credentials = [storedCredential];
      return HttpResponse.json(storedCredential);
    }),
    http.post("*/api/integrations/:provider/credentials", ({ params }) => {
      providerCredentialRequest(params.provider);
      return HttpResponse.json(
        { detail: "Use MCP credentials" },
        { status: 400 },
      );
    }),
  );
});

async function openPreset(name: RegExp) {
  fireEvent.click(await screen.findByRole("button", { name }));
  return screen.findByRole("dialog");
}

async function openPicker() {
  const buttons = await screen.findAllByRole("button", {
    name: /connect.*service/i,
  });
  fireEvent.click(buttons[0]);
  return screen.findByRole("dialog");
}

describe("SettingsIntegrationsPage — MCP catalogue", () => {
  test("lists native providers and branded MCP entries together on the page", async () => {
    render(<SettingsIntegrationsPage />);
    expect(
      await screen.findByRole("heading", { name: "Available integrations" }),
    ).toBeDefined();
    expect(
      await screen.findByRole("button", { name: /github.*issues and prs/i }),
    ).toBeDefined();
    const notion = screen.getByRole("button", {
      name: /notion.*official mcp/i,
    });
    expect(within(notion).getByText("MCP", { exact: true })).toBeDefined();
    const local = screen.getByRole("button", { name: /shadcn\/ui/i });
    expect(within(local).getByText(/setup required/i)).toBeDefined();
    expect(screen.queryByText("Mcp Notion")).toBeNull();
  });

  test("search filters available native and MCP providers", async () => {
    render(<SettingsIntegrationsPage />);
    await screen.findByRole("button", { name: /notion.*official mcp/i });
    fireEvent.change(screen.getByLabelText(/search integrations/i), {
      target: { value: "notion" },
    });
    await waitFor(() => {
      expect(
        screen.queryByRole("button", { name: /github.*issues and prs/i }),
      ).toBeNull();
    });
    expect(
      screen.getByRole("button", { name: /notion.*official mcp/i }),
    ).toBeDefined();
    expect(
      screen.queryByRole("button", { name: /superme.*official mcp/i }),
    ).toBeNull();
  });

  test("a page preset opens its detail with a locked URL and uses generic MCP OAuth", async () => {
    render(<SettingsIntegrationsPage />);
    const dialog = await openPreset(/notion.*official mcp/i);
    const input = (await within(dialog).findByLabelText(
      /server url/i,
    )) as HTMLInputElement;
    expect(input.value).toBe("https://mcp.notion.com/mcp");
    expect(input.readOnly).toBe(true);
    fireEvent.click(within(dialog).getByRole("button", { name: /^connect$/i }));
    await waitFor(() => {
      expect(oauthRequest).toHaveBeenCalledWith({
        server_url: "https://mcp.notion.com/mcp",
      });
    });
    expect(providerCredentialRequest).not.toHaveBeenCalled();
  });

  test("the Connect Service picker includes the same named MCP presets", async () => {
    render(<SettingsIntegrationsPage />);
    const dialog = await openPicker();
    fireEvent.click(
      await within(dialog).findByRole("button", {
        name: /notion.*official mcp/i,
      }),
    );
    expect(
      (
        (await within(dialog).findByLabelText(
          /server url/i,
        )) as HTMLInputElement
      ).value,
    ).toBe("https://mcp.notion.com/mcp");
  });

  test("generic MCP setup retains a blank editable URL", async () => {
    render(<SettingsIntegrationsPage />);
    const dialog = await openPicker();
    fireEvent.click(await within(dialog).findByText("Connect any MCP server"));
    const input = (await within(dialog).findByLabelText(
      /server url/i,
    )) as HTMLInputElement;
    expect(input.value).toBe("");
    expect(input.readOnly).toBe(false);
    expect(
      (
        within(dialog).getByRole("button", {
          name: /^connect$/i,
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(true);
  });

  test("unavailable presets explain setup and link documentation without attempting OAuth", async () => {
    render(<SettingsIntegrationsPage />);
    const dialog = await openPreset(/shadcn\/ui/i);
    expect(
      await within(dialog).findByText(/run this server locally/i),
    ).toBeDefined();
    expect(
      within(dialog)
        .getByRole("link", { name: /documentation/i })
        .getAttribute("href"),
    ).toBe("https://docs.example.com/shadcn_ui");
    expect(
      within(dialog).queryByRole("button", { name: /^connect$/i }),
    ).toBeNull();
    expect(within(dialog).queryByLabelText(/server url/i)).toBeNull();
    expect(oauthRequest).not.toHaveBeenCalled();
    expect(tokenRequest).not.toHaveBeenCalled();
  });

  test("token presets probe and save under generic MCP identity with the preset URL", async () => {
    render(<SettingsIntegrationsPage />);
    const dialog = await openPreset(/superme.*official mcp/i);
    fireEvent.change(
      await within(dialog).findByPlaceholderText("Paste API token"),
      { target: { value: "test-superme-token" } },
    );
    fireEvent.click(
      within(dialog).getByRole("button", { name: /save token/i }),
    );
    await waitFor(() => {
      expect(tokenRequest).toHaveBeenCalledWith({
        server_url: "https://mcp.superme.ai",
        token: "Bearer test-superme-token",
      });
    });
    expect(discoveryRequest).toHaveBeenCalledWith({
      server_url: "https://mcp.superme.ai",
      auth_token: "Bearer test-superme-token",
    });
    expect(await screen.findByText("SuperMe account")).toBeDefined();
    expect(oauthRequest).not.toHaveBeenCalled();
    expect(providerCredentialRequest).not.toHaveBeenCalled();
  });

  test("empty public discovery explains that no tools are available", async () => {
    server.use(
      http.post("*/api/mcp/discover-tools", () =>
        HttpResponse.json({ tools: [] }),
      ),
    );
    render(<SettingsIntegrationsPage />);
    const dialog = await openPreset(/parallel.*official mcp/i);
    fireEvent.click(
      await within(dialog).findByRole("button", { name: /check connection/i }),
    );
    expect(
      await within(dialog).findByText(
        "Connected, but this server returned no tools.",
      ),
    ).toBeDefined();
    expect(within(dialog).queryByText(/ready to use/i)).toBeNull();
    expect(tokenRequest).not.toHaveBeenCalled();
    expect(credentials).toEqual([]);
  });

  test("public MCP discovery reports tools without storing a fabricated credential", async () => {
    render(<SettingsIntegrationsPage />);
    const dialog = await openPreset(/parallel.*official mcp/i);
    fireEvent.click(
      await within(dialog).findByRole("button", { name: /check connection/i }),
    );
    expect(await within(dialog).findByText(/2 tools available/i)).toBeDefined();
    expect(discoveryRequest).toHaveBeenCalledWith({
      server_url: "https://search.parallel.ai/mcp",
    });
    expect(tokenRequest).not.toHaveBeenCalled();
    expect(oauthRequest).not.toHaveBeenCalled();
    expect(providerCredentialRequest).not.toHaveBeenCalled();
    expect(credentials).toEqual([]);
  });
});
