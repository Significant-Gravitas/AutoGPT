import {
  render,
  screen,
  fireEvent,
  waitFor,
  cleanup,
} from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { MCPSetupCard } from "../MCPSetupCard";

// Mock the copilot chat actions used by MCPSetupCard
const mockOnSend = vi.fn();
vi.mock(
  "../../../../../components/CopilotChatActionsProvider/useCopilotChatActions",
  () => ({
    useCopilotChatActions: () => ({ onSend: mockOnSend }),
  }),
);

// Mock the OAuth popup utility
vi.mock("@/lib/oauth-popup", () => ({
  openOAuthPopup: vi.fn(),
}));

// Mock the generated API functions
vi.mock("@/app/api/__generated__/endpoints/mcp/mcp", () => ({
  postV2InitiateOauthLoginForAnMcpServer: vi.fn(),
  postV2ExchangeOauthCodeForMcpTokens: vi.fn(),
  postV2StoreABearerTokenForAnMcpServer: vi.fn(),
}));

function makeSetupOutput(serverUrl = "https://mcp.example.com/mcp") {
  return {
    type: "setup_requirements" as const,
    message: "The MCP server at mcp.example.com requires authentication.",
    session_id: "test-session",
    setup_info: {
      agent_id: serverUrl,
      agent_name: "MCP: mcp.example.com",
      user_readiness: {
        has_all_credentials: false,
        missing_credentials: {},
        ready_to_run: false,
      },
      requirements: {
        credentials: [],
        inputs: [],
        execution_modes: ["immediate"],
      },
    },
    graph_id: null,
    graph_version: null,
  };
}

describe("MCPSetupCard", () => {
  afterEach(() => cleanup());

  it("renders setup message and connect button", () => {
    render(<MCPSetupCard output={makeSetupOutput()} />);
    expect(screen.getByText(/requires authentication/)).toBeDefined();
    expect(
      screen.getByRole("button", { name: /connect to mcp.example.com/i }),
    ).toBeDefined();
  });

  it("shows manual token input after OAuth 400", async () => {
    const { postV2InitiateOauthLoginForAnMcpServer } = await import(
      "@/app/api/__generated__/endpoints/mcp/mcp"
    );
    vi.mocked(postV2InitiateOauthLoginForAnMcpServer).mockResolvedValueOnce({
      status: 400,
      data: { detail: "No OAuth support" },
      headers: new Headers(),
    } as never);

    render(<MCPSetupCard output={makeSetupOutput()} />);
    fireEvent.click(
      screen.getByRole("button", { name: /connect to mcp.example.com/i }),
    );

    await waitFor(() => {
      expect(screen.getByPlaceholderText("Paste API token")).toBeDefined();
    });
    expect(screen.getByText(/does not support OAuth/)).toBeDefined();
  });

  it("shows connected state after manual token", async () => {
    const {
      postV2InitiateOauthLoginForAnMcpServer,
      postV2StoreABearerTokenForAnMcpServer,
    } = await import("@/app/api/__generated__/endpoints/mcp/mcp");

    // First click: OAuth fails with 400 → shows manual token input
    vi.mocked(postV2InitiateOauthLoginForAnMcpServer).mockResolvedValueOnce({
      status: 400,
      data: { detail: "No OAuth" },
      headers: new Headers(),
    } as never);

    render(<MCPSetupCard output={makeSetupOutput()} />);
    fireEvent.click(
      screen.getByRole("button", { name: /connect to mcp.example.com/i }),
    );

    await waitFor(() => {
      expect(screen.getByPlaceholderText("Paste API token")).toBeDefined();
    });

    // Mock the token store endpoint
    vi.mocked(postV2StoreABearerTokenForAnMcpServer).mockResolvedValueOnce({
      status: 200,
      data: {
        id: "cred-1",
        provider: "mcp",
        type: "oauth2",
        title: "MCP: mcp.example.com",
        scopes: [],
      },
      headers: new Headers(),
    } as never);

    // Enter token and submit
    fireEvent.change(screen.getByPlaceholderText("Paste API token"), {
      target: { value: "my-secret-token" },
    });
    fireEvent.click(screen.getByRole("button", { name: /use token/i }));

    await waitFor(() => {
      expect(screen.getByText(/connected to mcp.example.com/i)).toBeDefined();
    });
  });
});
