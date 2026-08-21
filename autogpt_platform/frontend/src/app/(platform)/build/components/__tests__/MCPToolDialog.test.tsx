import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";

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
        auth_token: "private-secret",
      },
    );
    expect(postV2StoreABearerTokenForAnMcpServer).toHaveBeenCalledWith({
      server_url: PRIVATE_SERVER_URL,
      token: "private-secret",
    });
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
