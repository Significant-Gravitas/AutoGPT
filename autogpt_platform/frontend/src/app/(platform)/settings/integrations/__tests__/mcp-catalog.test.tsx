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
import { openOAuthPopup } from "@/lib/oauth-popup";

import SettingsIntegrationsPage from "../page";

vi.mock("@/lib/oauth-popup", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/oauth-popup")>()),
  openOAuthPopup: vi.fn(),
}));

function preset(
  slug: string,
  name: string,
  serverURL: string | null,
  authMode: "oauth" | "token" | "none" | "unknown" = "oauth",
  connectionMode: "hosted" | "custom" | "unavailable" = serverURL
    ? "hosted"
    : "unavailable",
  iconID: string | null = null,
  metadata: Partial<NonNullable<ProviderMetadata["mcp_server"]>> = {},
): ProviderMetadata {
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
        : "Follow the official setup instructions for this server.",
      connection_mode: connectionMode,
      auth_mode: authMode,
      icon_id: iconID,
      ...metadata,
    },
  };
}

const providers: ProviderMetadata[] = [
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
  preset("treg", "Treg", "https://treg.to/mcp/", "token"),
  preset(
    "parallel",
    "Parallel",
    "https://search.parallel.ai/mcp",
    "none",
    "hosted",
    null,
    {
      auth_methods: ["none", "oauth", "bearer"],
      oauth_server_url: "https://search.parallel.ai/mcp-oauth",
    },
  ),
  preset(
    "aws_knowledge",
    "AWS Knowledge",
    "https://knowledge-mcp.global.api.aws",
    "none",
  ),
  preset("langfuse", "Langfuse", null, "token", "custom", null, {
    auth_methods: ["basic"],
  }),
  preset("shadcn_ui", "shadcn/ui", null, "unknown"),
  preset(
    "azure_cosmos_db",
    "Azure Cosmos DB",
    null,
    "token",
    "custom",
    "microsoft",
  ),
];

const storedCredential: CredentialsMetaResponse = {
  id: "catalog-credential",
  provider: "mcp",
  type: "oauth2",
  title: "Treg account",
  scopes: null,
  username: null,
  host: "https://treg.to/mcp/",
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
    expect(screen.queryByRole("button", { name: /shadcn\/ui/i })).toBeNull();
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
      screen.queryByRole("button", { name: /treg.*official mcp/i }),
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

  test("the service picker hides unavailable presets even if the API returns them", async () => {
    render(<SettingsIntegrationsPage />);
    const dialog = await openPicker();
    await within(dialog).findByRole("button", {
      name: /notion.*official mcp/i,
    });
    expect(
      within(dialog).queryByRole("button", { name: /shadcn\/ui/i }),
    ).toBeNull();
    expect(oauthRequest).not.toHaveBeenCalled();
    expect(tokenRequest).not.toHaveBeenCalled();
  });

  test("token presets probe and save under generic MCP identity with the preset URL", async () => {
    render(<SettingsIntegrationsPage />);
    const dialog = await openPreset(/treg.*official mcp/i);
    fireEvent.change(
      await within(dialog).findByPlaceholderText("Paste API token"),
      { target: { value: "test-treg-token" } },
    );
    fireEvent.click(
      within(dialog).getByRole("button", { name: /save token/i }),
    );
    await waitFor(() => {
      expect(tokenRequest).toHaveBeenCalledWith({
        server_url: "https://treg.to/mcp/",
        token: "Bearer test-treg-token",
      });
    });
    expect(discoveryRequest).toHaveBeenCalledWith({
      server_url: "https://treg.to/mcp/",
      auth_token: "Bearer test-treg-token",
    });
    expect(await screen.findByText("Treg account")).toBeDefined();
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
    const dialog = await openPreset(/aws knowledge.*official mcp/i);
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
    const dialog = await openPreset(/aws knowledge.*official mcp/i);
    fireEvent.click(
      await within(dialog).findByRole("button", { name: /check connection/i }),
    );
    expect(await within(dialog).findByText(/2 tools available/i)).toBeDefined();
    expect(within(dialog).getByText(/no connection was saved/i)).toBeDefined();
    expect(within(dialog).getByText(/use this server url/i)).toBeDefined();
    expect(within(dialog).queryByText(/ready to use/i)).toBeNull();
    expect(discoveryRequest).toHaveBeenCalledWith({
      server_url: "https://knowledge-mcp.global.api.aws",
      use_saved_credentials: false,
    });
    expect(tokenRequest).not.toHaveBeenCalled();
    expect(oauthRequest).not.toHaveBeenCalled();
    expect(providerCredentialRequest).not.toHaveBeenCalled();
    expect(credentials).toEqual([]);
  });

  test("OAuth presets offer a validated manual token after the vendor popup rejects sign-in", async () => {
    server.use(
      http.post("*/api/mcp/oauth/login", () =>
        HttpResponse.json({
          login_url: "https://vendor.example.com/authorize",
          state_token: "test-state",
        }),
      ),
    );
    vi.mocked(openOAuthPopup).mockImplementation(() => ({
      promise: Promise.reject(new Error("Vendor rejected this sign-in")),
      cleanup: { abort: vi.fn(), signal: new AbortController().signal },
      popupBlocked: false,
      fallbackBlocked: false,
    }));
    render(<SettingsIntegrationsPage />);
    const dialog = await openPreset(/parallel.*official mcp/i);
    fireEvent.mouseDown(
      await within(dialog).findByRole("tab", { name: /^sign in$/i }),
      { button: 0, ctrlKey: false },
    );
    fireEvent.click(
      await within(dialog).findByRole("button", { name: /^connect$/i }),
    );
    expect(
      await within(dialog).findByText("Vendor rejected this sign-in"),
    ).toBeDefined();
    fireEvent.mouseDown(
      within(dialog).getByRole("tab", { name: /^api token$/i }),
      { button: 0, ctrlKey: false },
    );
    expect(
      within(dialog).getByText(
        /credential described in the setup instructions/i,
      ),
    ).toBeDefined();
    fireEvent.change(within(dialog).getByPlaceholderText("Paste API token"), {
      target: { value: "test-manual-token" },
    });
    fireEvent.click(
      within(dialog).getByRole("button", { name: /save token/i }),
    );
    await waitFor(() =>
      expect(tokenRequest).toHaveBeenCalledWith({
        server_url: "https://search.parallel.ai/mcp",
        token: "Bearer test-manual-token",
      }),
    );
    expect(discoveryRequest).toHaveBeenCalledWith({
      server_url: "https://search.parallel.ai/mcp",
      auth_token: "Bearer test-manual-token",
    });
    expect(discoveryRequest.mock.invocationCallOrder[0]).toBeLessThan(
      tokenRequest.mock.invocationCallOrder[0],
    );
    expect(providerCredentialRequest).not.toHaveBeenCalled();
  });

  test("custom presets accept an editable URL and validate a Basic credential before saving", async () => {
    render(<SettingsIntegrationsPage />);
    const dialog = await openPreset(/langfuse.*official mcp/i);
    const url =
      await within(dialog).findByLabelText<HTMLInputElement>(/server url/i);
    expect(url.value).toBe("");
    expect(url.readOnly).toBe(false);
    fireEvent.change(url, {
      target: { value: "https://cloud.langfuse.com/api/public/mcp" },
    });
    expect(within(dialog).queryByLabelText("Authentication type")).toBeNull();
    fireEvent.change(within(dialog).getByPlaceholderText(/paste base64/i), {
      target: { value: "public:secret" },
    });
    fireEvent.click(
      within(dialog).getByRole("button", { name: /save token/i }),
    );
    expect(
      await within(dialog).findByText(/unencoded user:password/i),
    ).toBeDefined();
    expect(discoveryRequest).not.toHaveBeenCalled();
    expect(tokenRequest).not.toHaveBeenCalled();
    fireEvent.change(within(dialog).getByPlaceholderText(/paste base64/i), {
      target: { value: "cHVibGljOnNlY3JldA==" },
    });
    fireEvent.click(
      within(dialog).getByRole("button", { name: /save token/i }),
    );
    await waitFor(() =>
      expect(tokenRequest).toHaveBeenCalledWith({
        server_url: "https://cloud.langfuse.com/api/public/mcp",
        token: "Basic cHVibGljOnNlY3JldA==",
      }),
    );
    expect(discoveryRequest).toHaveBeenCalledWith({
      server_url: "https://cloud.langfuse.com/api/public/mcp",
      auth_token: "Basic cHVibGljOnNlY3JldA==",
    });
    expect(discoveryRequest.mock.invocationCallOrder[0]).toBeLessThan(
      tokenRequest.mock.invocationCallOrder[0],
    );
    expect(oauthRequest).not.toHaveBeenCalled();
  });

  test("Microsoft presets use the existing webp icon in the list and detail", async () => {
    render(<SettingsIntegrationsPage />);
    const row = await screen.findByRole("button", {
      name: /azure cosmos db.*official mcp/i,
    });
    expect(row.querySelector("img")?.getAttribute("src")).toContain(
      "microsoft.webp",
    );
    fireEvent.click(row);
    const dialog = await screen.findByRole("dialog");
    expect(
      (await within(dialog).findByAltText("Azure Cosmos DB logo")).getAttribute(
        "src",
      ),
    ).toContain("microsoft.webp");
  });
});
