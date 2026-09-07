import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  CredentialsProvidersContext,
  type CredentialsProvidersContextType,
} from "@/providers/agent-credentials/credentials-provider";

import { MCPToolDialog } from "../MCPToolDialog";

vi.mock("@/app/api/__generated__/endpoints/mcp/mcp", () => ({
  postV2DiscoverAvailableToolsOnAnMcpServer: vi.fn(),
  postV2InitiateOauthLoginForAnMcpServer: vi.fn(),
  postV2ExchangeOauthCodeForMcpTokens: vi.fn(),
  postV2StoreABearerTokenForAnMcpServer: vi.fn(),
}));

vi.mock("@/lib/oauth-popup", () => ({
  openOAuthPopup: vi.fn(),
}));

const PRIVATE_SERVER_URL = "https://private.example.com/mcp";
const PUBLIC_SERVER_URL = "https://public.example.com/mcp";
const CREDENTIAL = {
  id: "credential-id",
  provider: "mcp",
  type: "oauth2",
  title: "private.example.com",
};
const PRIVATE_TOOL = {
  name: "private-tool",
  description: "A private tool",
  input_schema: { type: "object", properties: {} },
};
const PUBLIC_TOOL = {
  name: "public-tool",
  description: "A public tool",
  input_schema: { type: "object", properties: {} },
};

function apiResponse(status: number, data: unknown) {
  return { status, data, headers: new Headers() } as never;
}

async function connectPrivateServer() {
  const {
    postV2DiscoverAvailableToolsOnAnMcpServer,
    postV2InitiateOauthLoginForAnMcpServer,
    postV2StoreABearerTokenForAnMcpServer,
  } = await import("@/app/api/__generated__/endpoints/mcp/mcp");

  vi.mocked(postV2DiscoverAvailableToolsOnAnMcpServer)
    .mockResolvedValueOnce(
      apiResponse(401, { detail: "Authentication required" }),
    )
    .mockResolvedValueOnce(
      apiResponse(200, {
        tools: [PRIVATE_TOOL],
        server_name: "Private Server",
      }),
    );
  vi.mocked(postV2InitiateOauthLoginForAnMcpServer).mockResolvedValueOnce(
    apiResponse(400, { detail: "OAuth not supported" }),
  );
  vi.mocked(postV2StoreABearerTokenForAnMcpServer).mockResolvedValueOnce(
    apiResponse(200, CREDENTIAL),
  );

  fireEvent.change(screen.getByLabelText("Server URL"), {
    target: { value: PRIVATE_SERVER_URL },
  });
  fireEvent.click(screen.getByRole("button", { name: "Discover Tools" }));

  const tokenInput = await screen.findByLabelText("API token");
  fireEvent.change(tokenInput, { target: { value: "private-secret" } });
  fireEvent.click(screen.getByRole("button", { name: "Connect & Discover" }));

  await screen.findByRole("button", { name: /private-tool/i });
}

describe("MCPToolDialog credential binding", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("attaches a manually stored credential to a tool from the same server", async () => {
    const {
      postV2DiscoverAvailableToolsOnAnMcpServer,
      postV2StoreABearerTokenForAnMcpServer,
    } = await import("@/app/api/__generated__/endpoints/mcp/mcp");
    const onConfirm = vi.fn();
    render(<MCPToolDialog open onClose={() => {}} onConfirm={onConfirm} />);

    await connectPrivateServer();
    fireEvent.click(screen.getByRole("button", { name: /private-tool/i }));
    fireEvent.click(screen.getByRole("button", { name: "Add Block" }));

    expect(onConfirm).toHaveBeenCalledWith(
      expect.objectContaining({
        serverUrl: PRIVATE_SERVER_URL,
        selectedTool: PRIVATE_TOOL.name,
        credentials: CREDENTIAL,
      }),
    );
    expect(postV2DiscoverAvailableToolsOnAnMcpServer).toHaveBeenNthCalledWith(
      2,
      {
        server_url: PRIVATE_SERVER_URL,
        auth_token: "Bearer private-secret",
      },
    );
    expect(postV2StoreABearerTokenForAnMcpServer).toHaveBeenCalledWith({
      server_url: PRIVATE_SERVER_URL,
      token: "Bearer private-secret",
    });
  });

  it("stores a manual credential through the credentials provider", async () => {
    // Storing via the endpoint directly leaves the provider map without the
    // credential the node is about to be bound to, and the builder renders
    // that binding as "was removed" until the next page load.
    const { postV2StoreABearerTokenForAnMcpServer } = await import(
      "@/app/api/__generated__/endpoints/mcp/mcp"
    );
    const mcpStoreToken = vi.fn().mockResolvedValue(CREDENTIAL);
    const providers = {
      mcp: { mcpStoreToken, savedCredentials: [] },
    } as unknown as CredentialsProvidersContextType;
    const onConfirm = vi.fn();

    render(
      <CredentialsProvidersContext.Provider value={providers}>
        <MCPToolDialog open onClose={() => {}} onConfirm={onConfirm} />
      </CredentialsProvidersContext.Provider>,
    );

    await connectPrivateServer();
    fireEvent.click(screen.getByRole("button", { name: /private-tool/i }));
    fireEvent.click(screen.getByRole("button", { name: "Add Block" }));

    expect(mcpStoreToken).toHaveBeenCalledWith(
      PRIVATE_SERVER_URL,
      "Bearer private-secret",
    );
    expect(postV2StoreABearerTokenForAnMcpServer).not.toHaveBeenCalled();
    expect(onConfirm).toHaveBeenCalledWith(
      expect.objectContaining({ credentials: CREDENTIAL }),
    );
  });

  it("sends a Basic credential with the selected scheme", async () => {
    // Hardcoding "bearer" at the prepare call survived the whole dialog suite:
    // no Basic case existed on this surface.
    const {
      postV2DiscoverAvailableToolsOnAnMcpServer,
      postV2StoreABearerTokenForAnMcpServer,
    } = await import("@/app/api/__generated__/endpoints/mcp/mcp");
    const { postV2InitiateOauthLoginForAnMcpServer } = await import(
      "@/app/api/__generated__/endpoints/mcp/mcp"
    );

    vi.mocked(postV2DiscoverAvailableToolsOnAnMcpServer)
      .mockResolvedValueOnce(
        apiResponse(401, { detail: "Authentication required" }),
      )
      .mockResolvedValueOnce(
        apiResponse(200, {
          tools: [PRIVATE_TOOL],
          server_name: "Private Server",
        }),
      );
    vi.mocked(postV2InitiateOauthLoginForAnMcpServer).mockResolvedValueOnce(
      apiResponse(400, { detail: "OAuth not supported" }),
    );
    vi.mocked(postV2StoreABearerTokenForAnMcpServer).mockResolvedValueOnce(
      apiResponse(200, CREDENTIAL),
    );

    render(<MCPToolDialog open onClose={() => {}} onConfirm={vi.fn()} />);

    fireEvent.change(screen.getByLabelText("Server URL"), {
      target: { value: PRIVATE_SERVER_URL },
    });
    fireEvent.click(screen.getByRole("button", { name: "Discover Tools" }));

    await screen.findByLabelText("Authentication type");
    fireEvent.change(screen.getByLabelText("Authentication type"), {
      target: { value: "basic" },
    });
    fireEvent.change(screen.getByLabelText("Basic authentication token"), {
      target: { value: "cGstbGYtYWJjZA==" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Connect & Discover" }));

    await screen.findByRole("button", { name: /private-tool/i });
    expect(postV2StoreABearerTokenForAnMcpServer).toHaveBeenCalledWith({
      server_url: PRIVATE_SERVER_URL,
      token: "Basic cGstbGYtYWJjZA==",
    });
  });

  it("seeds the selector from the scheme already stored for the server", async () => {
    // This dialog was the only surface that never read `mcp_auth_scheme`, so
    // reconnecting a Basic credential silently downgraded it to Bearer.
    const {
      postV2DiscoverAvailableToolsOnAnMcpServer,
      postV2InitiateOauthLoginForAnMcpServer,
    } = await import("@/app/api/__generated__/endpoints/mcp/mcp");
    vi.mocked(postV2DiscoverAvailableToolsOnAnMcpServer).mockResolvedValueOnce(
      apiResponse(401, { detail: "Authentication required" }),
    );
    vi.mocked(postV2InitiateOauthLoginForAnMcpServer).mockResolvedValueOnce(
      apiResponse(400, { detail: "OAuth not supported" }),
    );

    const providers = {
      mcp: {
        savedCredentials: [
          {
            id: "c1",
            provider: "mcp",
            type: "oauth2",
            title: "MCP",
            host: PRIVATE_SERVER_URL,
            mcp_auth_scheme: "basic",
          },
        ],
      },
    } as unknown as CredentialsProvidersContextType;

    render(
      <CredentialsProvidersContext.Provider value={providers}>
        <MCPToolDialog open onClose={() => {}} onConfirm={vi.fn()} />
      </CredentialsProvidersContext.Provider>,
    );

    fireEvent.change(screen.getByLabelText("Server URL"), {
      target: { value: PRIVATE_SERVER_URL },
    });
    fireEvent.click(screen.getByRole("button", { name: "Discover Tools" }));

    const select = (await screen.findByLabelText(
      "Authentication type",
    )) as HTMLSelectElement;
    expect(select.value).toBe("basic");
  });

  it("does not store a manual credential rejected by discovery", async () => {
    const {
      postV2DiscoverAvailableToolsOnAnMcpServer,
      postV2InitiateOauthLoginForAnMcpServer,
      postV2StoreABearerTokenForAnMcpServer,
    } = await import("@/app/api/__generated__/endpoints/mcp/mcp");
    vi.mocked(postV2DiscoverAvailableToolsOnAnMcpServer)
      .mockResolvedValueOnce(
        apiResponse(401, { detail: "Authentication required" }),
      )
      .mockResolvedValueOnce(
        apiResponse(401, { detail: "Invalid API credential" }),
      );
    vi.mocked(postV2InitiateOauthLoginForAnMcpServer).mockResolvedValueOnce(
      apiResponse(400, { detail: "OAuth not supported" }),
    );

    render(<MCPToolDialog open onClose={() => {}} onConfirm={() => {}} />);

    fireEvent.change(screen.getByLabelText("Server URL"), {
      target: { value: PRIVATE_SERVER_URL },
    });
    fireEvent.click(screen.getByRole("button", { name: "Discover Tools" }));

    const tokenInput = await screen.findByLabelText("API token");
    fireEvent.change(tokenInput, { target: { value: "invalid-secret" } });
    fireEvent.click(screen.getByRole("button", { name: "Connect & Discover" }));

    expect(await screen.findByText("Invalid API credential")).toBeDefined();
    expect(postV2StoreABearerTokenForAnMcpServer).not.toHaveBeenCalled();
  });

  it("keeps the typed credential while the path of the same server is edited", async () => {
    const {
      postV2DiscoverAvailableToolsOnAnMcpServer,
      postV2InitiateOauthLoginForAnMcpServer,
    } = await import("@/app/api/__generated__/endpoints/mcp/mcp");
    vi.mocked(postV2DiscoverAvailableToolsOnAnMcpServer).mockResolvedValueOnce(
      apiResponse(401, { detail: "Authentication required" }),
    );
    vi.mocked(postV2InitiateOauthLoginForAnMcpServer).mockResolvedValueOnce(
      apiResponse(400, { detail: "OAuth not supported" }),
    );

    render(<MCPToolDialog open onClose={() => {}} onConfirm={() => {}} />);

    const urlInput = screen.getByLabelText("Server URL");
    fireEvent.change(urlInput, {
      target: { value: "https://private.example.com" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Discover Tools" }));

    const tokenInput = await screen.findByLabelText("API token");
    fireEvent.change(tokenInput, { target: { value: "secret-token" } });

    // Appending the `/mcp` suffix one character at a time. The reset guard
    // compared trimmed URLs, so every one of these keystrokes cleared the
    // credential and collapsed the panel, forcing another discover → 401 →
    // OAuth-probe round trip to get back to where the user already was.
    for (const url of [
      "https://private.example.com/",
      "https://private.example.com/m",
      "https://private.example.com/mc",
      PRIVATE_SERVER_URL,
    ]) {
      fireEvent.change(urlInput, { target: { value: url } });
    }

    expect((screen.getByLabelText("API token") as HTMLInputElement).value).toBe(
      "secret-token",
    );
  });

  it("does not reuse a credential after changing to a public server", async () => {
    const { postV2DiscoverAvailableToolsOnAnMcpServer } = await import(
      "@/app/api/__generated__/endpoints/mcp/mcp"
    );
    const onConfirm = vi.fn();
    render(<MCPToolDialog open onClose={() => {}} onConfirm={onConfirm} />);

    await connectPrivateServer();
    fireEvent.click(screen.getByRole("button", { name: "Back" }));
    fireEvent.change(screen.getByLabelText("Server URL"), {
      target: { value: PUBLIC_SERVER_URL },
    });
    vi.mocked(postV2DiscoverAvailableToolsOnAnMcpServer).mockResolvedValueOnce(
      apiResponse(200, {
        tools: [PUBLIC_TOOL],
        server_name: "Public Server",
      }),
    );

    fireEvent.click(screen.getByRole("button", { name: "Discover Tools" }));
    await screen.findByRole("button", { name: /public-tool/i });
    fireEvent.click(screen.getByRole("button", { name: /public-tool/i }));
    fireEvent.click(screen.getByRole("button", { name: "Add Block" }));

    await waitFor(() => {
      expect(onConfirm).toHaveBeenCalledWith(
        expect.objectContaining({
          serverUrl: PUBLIC_SERVER_URL,
          selectedTool: PUBLIC_TOOL.name,
          credentials: null,
        }),
      );
    });
  });
});
