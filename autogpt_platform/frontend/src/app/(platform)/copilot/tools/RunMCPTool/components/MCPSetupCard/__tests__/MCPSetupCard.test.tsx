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

// Mock the credentials list hook used for the on-mount live-cred re-sync.
// Default: no stored creds → ``liveHasCred=false`` matches the persisted
// ``has_all_credentials=false`` snapshot so the existing tests don't have
// to thread a connected state through MSW.  ``setMockLiveCreds`` lets
// individual tests override the live state to verify the refresh path.
let mockLiveCreds: Array<{ provider: string; host?: string | null }> = [];
function setMockLiveCreds(
  next: Array<{ provider: string; host?: string | null }>,
) {
  mockLiveCreds = next;
}
vi.mock("@/app/api/__generated__/endpoints/integrations/integrations", () => ({
  useGetV1ListCredentials: () => ({
    data: mockLiveCreds,
    isLoading: false,
  }),
}));

function makeSetupOutput(
  serverUrl = "https://mcp.example.com/mcp",
  hasAllCredentials = false,
) {
  return {
    type: "setup_requirements" as const,
    message: "To continue, sign in to example.com and approve access.",
    session_id: "test-session",
    setup_info: {
      agent_id: serverUrl,
      agent_name: "example.com",
      user_readiness: {
        has_all_credentials: hasAllCredentials,
        missing_credentials: {},
        ready_to_run: hasAllCredentials,
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
  afterEach(() => {
    cleanup();
    setMockLiveCreds([]);
  });

  it("renders setup message and connect button", () => {
    render(<MCPSetupCard output={makeSetupOutput()} />);
    expect(screen.getByText(/sign in to example\.com/i)).toBeDefined();
    expect(
      screen.getByRole("button", { name: /connect example\.com/i }),
    ).toBeDefined();
  });

  it("renders Connected/Reconnect when live creds say the server is connected even if the persisted snapshot was disconnected", () => {
    // Persisted card snapshot was emitted while the cred was missing (e.g.
    // John's stale-cred 401 path), but on chat refresh the cred now exists.
    // Card should render the connected pill, not the bare Connect button.
    setMockLiveCreds([
      { provider: "mcp", host: "https://mcp.example.com/mcp" },
    ]);
    render(<MCPSetupCard output={makeSetupOutput()} />);
    expect(screen.getByText(/connected to example\.com/i)).toBeDefined();
    expect(screen.getByRole("button", { name: /reconnect/i })).toBeDefined();
  });

  it("matches live creds across a trailing slash on the server URL", () => {
    // Card was emitted with no trailing slash; stored cred has one.
    // The frontend ``normalizeMcpUrl`` mirrors the backend so they match.
    setMockLiveCreds([
      { provider: "mcp", host: "https://mcp.example.com/mcp/" },
    ]);
    render(
      <MCPSetupCard output={makeSetupOutput("https://mcp.example.com/mcp")} />,
    );
    expect(screen.getByText(/connected to example\.com/i)).toBeDefined();
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
      screen.getByRole("button", { name: /connect example\.com/i }),
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
      screen.getByRole("button", { name: /connect example\.com/i }),
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
      expect(screen.getByText(/connected to example\.com/i)).toBeDefined();
    });
  });

  it("drops Connected state and surfaces manual token input when Reconnect hits HTTP 400", async () => {
    const { postV2InitiateOauthLoginForAnMcpServer } = await import(
      "@/app/api/__generated__/endpoints/mcp/mcp"
    );
    vi.mocked(postV2InitiateOauthLoginForAnMcpServer).mockResolvedValueOnce({
      status: 400,
      data: { detail: "No OAuth support" },
      headers: new Headers(),
    } as never);

    render(<MCPSetupCard output={makeSetupOutput(undefined, true)} />);

    // Starts in Connected state with Reconnect button visible.
    const reconnectBtn = screen.getByRole("button", { name: /reconnect/i });
    fireEvent.click(reconnectBtn);

    // After 400, the not-connected branch must render: error banner + manual
    // token input. The Connected/Reconnect banner must be gone.
    await waitFor(() => {
      expect(screen.getByPlaceholderText("Paste API token")).toBeDefined();
    });
    expect(screen.getByText(/does not support OAuth/)).toBeDefined();
    expect(screen.queryByText(/connected to example\.com/i)).toBeNull();
  });
});
