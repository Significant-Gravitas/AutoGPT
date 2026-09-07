import {
  act,
  render,
  screen,
  fireEvent,
  waitFor,
  cleanup,
} from "@/tests/integrations/test-utils";
import {
  CredentialsProvidersContext,
  type CredentialsProvidersContextType,
} from "@/providers/agent-credentials/credentials-provider";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  ChainActionsContext,
  type ChainActionEntry,
} from "../../../../../components/ToolChain/chainActions";
import { MCPSetupCard } from "../MCPSetupCard";

// Mock the copilot chat actions used by MCPSetupCard
const mockOnSend = vi.fn();
let currentOnSend = mockOnSend;
vi.mock(
  "../../../../../components/CopilotChatActionsProvider/useCopilotChatActions",
  () => ({
    useCopilotChatActions: () => ({ onSend: currentOnSend }),
  }),
);

// Mock the OAuth popup utility
vi.mock("@/lib/oauth-popup", () => ({
  openOAuthPopup: vi.fn(),
}));

// Mock the generated API functions
vi.mock("@/app/api/__generated__/endpoints/mcp/mcp", () => ({
  postV2DiscoverAvailableToolsOnAnMcpServer: vi.fn(),
  postV2InitiateOauthLoginForAnMcpServer: vi.fn(),
  postV2ExchangeOauthCodeForMcpTokens: vi.fn(),
  postV2StoreABearerTokenForAnMcpServer: vi.fn(),
}));

// Mock the credentials list hook used for the on-mount live-cred re-sync.
// Default: no stored creds → ``liveHasCred=false`` matches the persisted
// ``has_all_credentials=false`` snapshot so the existing tests don't have
// to thread a connected state through MSW.  ``setMockLiveCreds`` lets
// individual tests override the live state to verify the refresh path.
let mockLiveCreds: Array<{
  provider: string;
  host?: string | null;
  mcp_auth_scheme?: "basic" | "bearer" | null;
}> = [];
function setMockLiveCreds(
  next: Array<{
    provider: string;
    host?: string | null;
    mcp_auth_scheme?: "basic" | "bearer" | null;
  }>,
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

// The placeholder tracks the selected scheme: "Paste API token" under Bearer,
// and the Base64 wording under Basic, so it stops restating the mistake the
// hint below it exists to prevent. These queries match either.
const manualTokenPlaceholder = /paste (api token|base64 of user:password)/i;

describe("MCPSetupCard", () => {
  // Storing a manual credential probes the server first, so the default is an
  // accepting server; the tests that care override it.
  beforeEach(async () => {
    const { postV2DiscoverAvailableToolsOnAnMcpServer } = await import(
      "@/app/api/__generated__/endpoints/mcp/mcp"
    );
    vi.mocked(postV2DiscoverAvailableToolsOnAnMcpServer).mockResolvedValue({
      status: 200,
      data: { tools: [], server_name: "Example" },
      headers: new Headers(),
    } as never);
  });

  afterEach(() => {
    cleanup();
    // Without this, call history leaks between tests and any
    // `not.toHaveBeenCalled()` assertion silently depends on declaration order.
    vi.clearAllMocks();
    setMockLiveCreds([]);
    currentOnSend = mockOnSend;
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

  it("drops Connected state when live API confirms the cred is gone, even if persisted snapshot said connected", () => {
    // Sentry-flagged sticky-localConnected regression: the card was
    // emitted with ``has_all_credentials=true`` (persisted snapshot)
    // but the cred was deleted server-side before the card re-mounted
    // on chat refresh.  Live API returns ``liveHasCred=false`` (cred
    // gone).  The pill must flip to "Connect", not stay on "Reconnect".
    setMockLiveCreds([]); // live truth: no cred
    render(<MCPSetupCard output={makeSetupOutput(undefined, true)} />);
    expect(
      screen.getByRole("button", { name: /connect example\.com/i }),
    ).toBeDefined();
    expect(screen.queryByText(/connected to example\.com/i)).toBeNull();
  });

  it("falls back to persisted snapshot when the live cred API fails", () => {
    // ``useGetV1ListCredentials`` returns ``null`` via ``select`` on a
    // 401/5xx response.  Treating that as "no creds" would override a
    // still-valid persisted snapshot; the card must keep the
    // ``initiallyConnected`` truth instead.
    setMockLiveCreds(null as unknown as Array<{ provider: string }>);
    render(<MCPSetupCard output={makeSetupOutput(undefined, true)} />);
    expect(screen.getByText(/connected to example\.com/i)).toBeDefined();
    expect(screen.getByRole("button", { name: /reconnect/i })).toBeDefined();
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
      expect(screen.getByPlaceholderText(manualTokenPlaceholder)).toBeDefined();
    });
    expect(screen.getByText(/does not support OAuth/)).toBeDefined();
  });

  it("uses a unique manual credential input id for each mounted card", async () => {
    const { postV2InitiateOauthLoginForAnMcpServer } = await import(
      "@/app/api/__generated__/endpoints/mcp/mcp"
    );
    vi.mocked(postV2InitiateOauthLoginForAnMcpServer)
      .mockResolvedValueOnce({
        status: 400,
        data: { detail: "No OAuth support" },
        headers: new Headers(),
      } as never)
      .mockResolvedValueOnce({
        status: 400,
        data: { detail: "No OAuth support" },
        headers: new Headers(),
      } as never);

    render(
      <>
        <MCPSetupCard output={makeSetupOutput()} />
        <MCPSetupCard output={makeSetupOutput()} />
      </>,
    );
    fireEvent.click(
      screen.getAllByRole("button", { name: /connect example\.com/i })[0],
    );
    await waitFor(() => {
      expect(
        screen.getAllByPlaceholderText(manualTokenPlaceholder),
      ).toHaveLength(1);
    });
    fireEvent.click(
      screen.getAllByRole("button", { name: /connect example\.com/i })[1],
    );
    await waitFor(() => {
      expect(
        screen.getAllByPlaceholderText(manualTokenPlaceholder),
      ).toHaveLength(2);
    });

    const inputs = screen.getAllByPlaceholderText(manualTokenPlaceholder);
    expect(inputs[0].id).toBeTruthy();
    expect(inputs[1].id).toBeTruthy();
    expect(inputs[0].id).not.toBe(inputs[1].id);
    for (const input of inputs) {
      expect(document.querySelector(`label[for="${input.id}"]`)).not.toBeNull();
    }
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
      expect(screen.getByPlaceholderText(manualTokenPlaceholder)).toBeDefined();
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
    fireEvent.change(screen.getByPlaceholderText(manualTokenPlaceholder), {
      target: { value: "my-secret-token" },
    });
    fireEvent.click(screen.getByRole("button", { name: /use token/i }));

    await waitFor(() => {
      expect(screen.getByText(/connected to example\.com/i)).toBeDefined();
    });
    expect(postV2StoreABearerTokenForAnMcpServer).toHaveBeenCalledWith({
      server_url: "https://mcp.example.com/mcp",
      token: "Bearer my-secret-token",
    });
  });

  it("stores a selected Basic credential with an explicit prefix", async () => {
    const {
      postV2InitiateOauthLoginForAnMcpServer,
      postV2StoreABearerTokenForAnMcpServer,
    } = await import("@/app/api/__generated__/endpoints/mcp/mcp");

    vi.mocked(postV2InitiateOauthLoginForAnMcpServer).mockResolvedValueOnce({
      status: 400,
      data: { detail: "No OAuth" },
      headers: new Headers(),
    } as never);
    vi.mocked(postV2StoreABearerTokenForAnMcpServer).mockResolvedValueOnce({
      status: 200,
      data: {
        id: "cred-basic",
        provider: "mcp",
        type: "oauth2",
        title: "MCP: mcp.example.com",
        scopes: [],
      },
      headers: new Headers(),
    } as never);

    render(<MCPSetupCard output={makeSetupOutput()} />);
    fireEvent.click(
      screen.getByRole("button", { name: /connect example\.com/i }),
    );
    await waitFor(() => {
      expect(screen.getByPlaceholderText(manualTokenPlaceholder)).toBeDefined();
    });

    fireEvent.change(
      screen.getByLabelText("Authentication type for example.com"),
      { target: { value: "basic" } },
    );
    fireEvent.change(
      screen.getByLabelText("Basic authentication token for example.com"),
      { target: { value: "  cGstbGYtYWJjZA==  " } },
    );
    fireEvent.click(screen.getByRole("button", { name: /use token/i }));

    await waitFor(() => {
      expect(postV2StoreABearerTokenForAnMcpServer).toHaveBeenCalledWith({
        server_url: "https://mcp.example.com/mcp",
        token: "Basic cGstbGYtYWJjZA==",
      });
    });
  });

  it("restores Basic for a saved manual credential during reconnect", async () => {
    setMockLiveCreds([
      {
        provider: "mcp",
        host: "https://mcp.example.com/mcp",
        mcp_auth_scheme: "basic",
      },
    ]);
    const {
      postV2InitiateOauthLoginForAnMcpServer,
      postV2StoreABearerTokenForAnMcpServer,
    } = await import("@/app/api/__generated__/endpoints/mcp/mcp");
    vi.mocked(postV2InitiateOauthLoginForAnMcpServer).mockResolvedValueOnce({
      status: 400,
      data: { detail: "No OAuth" },
      headers: new Headers(),
    } as never);
    vi.mocked(postV2StoreABearerTokenForAnMcpServer).mockResolvedValueOnce({
      status: 200,
      data: {
        id: "cred-basic",
        provider: "mcp",
        type: "oauth2",
        title: "MCP: mcp.example.com",
        scopes: [],
      },
      headers: new Headers(),
    } as never);

    render(<MCPSetupCard output={makeSetupOutput(undefined, true)} />);
    fireEvent.click(screen.getByRole("button", { name: /reconnect/i }));
    await waitFor(() => {
      expect(screen.getByLabelText(/authentication type/i)).toBeDefined();
    });

    expect(
      (screen.getByLabelText(/authentication type/i) as HTMLSelectElement)
        .value,
    ).toBe("basic");
    fireEvent.change(screen.getByPlaceholderText(manualTokenPlaceholder), {
      target: { value: "new-encoded-value" },
    });
    fireEvent.click(screen.getByRole("button", { name: /use token/i }));

    await waitFor(() => {
      expect(postV2StoreABearerTokenForAnMcpServer).toHaveBeenCalledWith({
        server_url: "https://mcp.example.com/mcp",
        token: "Basic new-encoded-value",
      });
    });
  });

  it("drops Connected state and surfaces manual token input when Reconnect hits HTTP 400", async () => {
    // Live creds must report the server connected so the card starts in
    // Connected/Reconnect — ``localConnected`` is no longer seeded from
    // the persisted snapshot (sentry bug: live false would have been
    // shadowed by a sticky ``localConnected=true`` from initialization).
    setMockLiveCreds([
      { provider: "mcp", host: "https://mcp.example.com/mcp" },
    ]);
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
      expect(screen.getByPlaceholderText(manualTokenPlaceholder)).toBeDefined();
    });
    expect(screen.getByText(/does not support OAuth/)).toBeDefined();
    expect(screen.queryByText(/connected to example\.com/i)).toBeNull();
  });

  it("drops Connected state on Reconnect failure even when live creds still report the server as connected", async () => {
    // Sentry-flagged bug: when a stored cred exists (``liveHasCred=true``)
    // and the user clicks Reconnect → OAuth 400, the previous
    // ``connected = localConnected || liveHasCred`` logic would keep the
    // Connected pill rendered because liveHasCred was still true.  The
    // user couldn't see the error or the manual-token input.  Fix:
    // ``forceDisconnected`` overrides both states on any catch.
    setMockLiveCreds([
      { provider: "mcp", host: "https://mcp.example.com/mcp" },
    ]);
    const { postV2InitiateOauthLoginForAnMcpServer } = await import(
      "@/app/api/__generated__/endpoints/mcp/mcp"
    );
    vi.mocked(postV2InitiateOauthLoginForAnMcpServer).mockResolvedValueOnce({
      status: 400,
      data: { detail: "No OAuth support" },
      headers: new Headers(),
    } as never);

    render(<MCPSetupCard output={makeSetupOutput()} />);
    // Live creds say connected → starts in Connected state.
    const reconnectBtn = screen.getByRole("button", { name: /reconnect/i });
    fireEvent.click(reconnectBtn);

    await waitFor(() => {
      expect(screen.getByPlaceholderText(manualTokenPlaceholder)).toBeDefined();
    });
    expect(screen.queryByText(/connected to example\.com/i)).toBeNull();
  });

  it("re-entrancy guard prevents handleConnect from firing twice on rapid double-click", async () => {
    // Without ``if (loading) return;`` the second click would race the
    // first's in-flight popup — abort it and reject the first's await
    // with OAUTH_ERROR_FLOW_CANCELED, even though the second attempt is
    // still alive. The guard keeps the second click a no-op so the
    // first attempt runs to completion.
    const { postV2InitiateOauthLoginForAnMcpServer } = await import(
      "@/app/api/__generated__/endpoints/mcp/mcp"
    );
    // Reset call counter — prior tests in this file also invoke the same mock.
    vi.mocked(postV2InitiateOauthLoginForAnMcpServer).mockClear();
    let resolveLogin: ((value: unknown) => void) | undefined;
    vi.mocked(postV2InitiateOauthLoginForAnMcpServer).mockReturnValueOnce(
      new Promise((res) => {
        resolveLogin = res;
      }) as never,
    );

    render(<MCPSetupCard output={makeSetupOutput()} />);
    const btn = screen.getByRole("button", { name: /connect example\.com/i });
    // Rapid double-click before the first call resolves.
    fireEvent.click(btn);
    fireEvent.click(btn);

    // Only one network call despite two clicks.
    expect(
      vi.mocked(postV2InitiateOauthLoginForAnMcpServer).mock.calls.length,
    ).toBe(1);

    // Drain the in-flight promise so React doesn't warn on unmount.
    resolveLogin?.({
      status: 400,
      data: { detail: "No OAuth" },
      headers: new Headers(),
    });
    await waitFor(() => {
      expect(screen.getByPlaceholderText(manualTokenPlaceholder)).toBeDefined();
    });
  });

  it("shows timeout-specific error message when OAuth popup times out", async () => {
    const { postV2InitiateOauthLoginForAnMcpServer } = await import(
      "@/app/api/__generated__/endpoints/mcp/mcp"
    );
    vi.mocked(postV2InitiateOauthLoginForAnMcpServer).mockResolvedValueOnce({
      status: 200,
      data: { login_url: "https://example.com/oauth", state_token: "s1" },
      headers: new Headers(),
    } as never);

    const { openOAuthPopup } = await import("@/lib/oauth-popup");
    vi.mocked(openOAuthPopup).mockReturnValueOnce({
      promise: Promise.reject(new Error("OAuth flow timed out")),
      cleanup: { abort: vi.fn(), signal: new AbortController().signal },
      popupBlocked: false,
      fallbackBlocked: false,
    });

    render(<MCPSetupCard output={makeSetupOutput()} />);
    fireEvent.click(
      screen.getByRole("button", { name: /connect example\.com/i }),
    );

    await waitFor(() => {
      expect(screen.getByText(/oauth sign-in timed out/i)).toBeDefined();
    });
  });

  it("shows generic error message when OAuth callback fails with a non-400 status", async () => {
    const { postV2InitiateOauthLoginForAnMcpServer } = await import(
      "@/app/api/__generated__/endpoints/mcp/mcp"
    );
    // Login itself returns 500 (not 400, not timeout) → catch hits the
    // "generic error" branch with the server's ``detail``.
    vi.mocked(postV2InitiateOauthLoginForAnMcpServer).mockResolvedValueOnce({
      status: 500,
      data: { detail: "Upstream OAuth registration failed" },
      headers: new Headers(),
    } as never);

    render(<MCPSetupCard output={makeSetupOutput()} />);
    fireEvent.click(
      screen.getByRole("button", { name: /connect example\.com/i }),
    );

    await waitFor(() => {
      expect(
        screen.getByText(/upstream oauth registration failed/i),
      ).toBeDefined();
    });
    // Manual-token input must NOT appear — that's the 400-only branch.
    expect(screen.queryByPlaceholderText(manualTokenPlaceholder)).toBeNull();
  });

  it("submits manual token via Enter key", async () => {
    // The token input has an onKeyDown that fires handleManualToken on
    // Enter — covers the keyboard path that's an alternative to the Use
    // Token button click.
    const {
      postV2InitiateOauthLoginForAnMcpServer,
      postV2StoreABearerTokenForAnMcpServer,
    } = await import("@/app/api/__generated__/endpoints/mcp/mcp");
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
      expect(screen.getByPlaceholderText(manualTokenPlaceholder)).toBeDefined();
    });

    vi.mocked(postV2StoreABearerTokenForAnMcpServer).mockResolvedValueOnce({
      status: 200,
      data: {
        id: "cred-enter",
        provider: "mcp",
        type: "oauth2",
        title: "MCP: mcp.example.com",
        scopes: [],
      },
      headers: new Headers(),
    } as never);

    const input = screen.getByPlaceholderText(manualTokenPlaceholder);
    fireEvent.change(input, { target: { value: "my-token" } });
    fireEvent.keyDown(input, { key: "Enter" });

    await waitFor(() => {
      expect(screen.getByText(/connected to example\.com/i)).toBeDefined();
    });
  });

  it("shows why a stored token was refused instead of the never-connected card", () => {
    render(
      <MCPSetupCard
        output={{
          ...makeSetupOutput(),
          message: "example.com rejected the saved credential (HTTP 401).",
          rejection: {
            provider: "mcp",
            detail: "HTTP 401 Error: Unauthorized",
            status_code: 401,
            credential_id: "cred-1",
            credential_title: "Sentry token",
          },
        }}
      />,
    );

    expect(screen.getByRole("alert").textContent).toContain(
      "HTTP 401 Error: Unauthorized",
    );
    expect(
      screen.getByRole("button", { name: /connect example\.com/i }),
    ).toBeDefined();
  });

  it("keeps the Connect affordance when a stale cred list still lists the refused token", () => {
    setMockLiveCreds([
      { provider: "mcp", host: "https://mcp.example.com/mcp" },
    ]);
    render(
      <MCPSetupCard
        output={{
          ...makeSetupOutput(),
          rejection: {
            provider: "mcp",
            detail: "HTTP 401 Error: Unauthorized",
            status_code: 401,
            credential_id: "cred-1",
            credential_title: null,
          },
        }}
      />,
    );

    expect(screen.queryByText(/connected to example\.com/i)).toBeNull();
    expect(
      screen.getByRole("button", { name: /connect example\.com/i }),
    ).toBeDefined();
  });

  it("does not report Connected for a credential the server rejects", async () => {
    // A 2xx from ``/mcp/token`` only says the row was written. Without a probe
    // the card claims Connected, the very next copilot call 401s, and the card,
    // the database and the assistant all disagree about the state.
    const {
      postV2DiscoverAvailableToolsOnAnMcpServer,
      postV2InitiateOauthLoginForAnMcpServer,
      postV2StoreABearerTokenForAnMcpServer,
    } = await import("@/app/api/__generated__/endpoints/mcp/mcp");
    vi.mocked(postV2InitiateOauthLoginForAnMcpServer).mockResolvedValueOnce({
      status: 400,
      data: { detail: "No OAuth" },
      headers: new Headers(),
    } as never);
    vi.mocked(postV2DiscoverAvailableToolsOnAnMcpServer).mockResolvedValue({
      status: 401,
      data: { detail: "Server rejected the credential" },
      headers: new Headers(),
    } as never);

    render(<MCPSetupCard output={makeSetupOutput(undefined, true)} />);
    fireEvent.click(screen.getByRole("button", { name: /connect example/i }));
    await waitFor(() => {
      expect(screen.getByPlaceholderText(manualTokenPlaceholder)).toBeDefined();
    });

    fireEvent.change(screen.getByPlaceholderText(manualTokenPlaceholder), {
      target: { value: "wrong-token" },
    });
    fireEvent.click(screen.getByRole("button", { name: /use token/i }));

    await waitFor(() => {
      expect(screen.getByText(/server rejected the credential/i)).toBeDefined();
    });
    expect(postV2StoreABearerTokenForAnMcpServer).not.toHaveBeenCalled();
    expect(screen.queryByText(/connected to example\.com/i)).toBeNull();
  });

  it("refuses an unencoded user:password before any request", async () => {
    const {
      postV2DiscoverAvailableToolsOnAnMcpServer,
      postV2InitiateOauthLoginForAnMcpServer,
      postV2StoreABearerTokenForAnMcpServer,
    } = await import("@/app/api/__generated__/endpoints/mcp/mcp");
    vi.mocked(postV2InitiateOauthLoginForAnMcpServer).mockResolvedValueOnce({
      status: 400,
      data: { detail: "No OAuth" },
      headers: new Headers(),
    } as never);

    render(<MCPSetupCard output={makeSetupOutput(undefined, true)} />);
    fireEvent.click(screen.getByRole("button", { name: /connect example/i }));
    await waitFor(() => {
      expect(screen.getByPlaceholderText(manualTokenPlaceholder)).toBeDefined();
    });

    fireEvent.change(screen.getByLabelText(/authentication type/i), {
      target: { value: "basic" },
    });
    fireEvent.change(screen.getByPlaceholderText(manualTokenPlaceholder), {
      target: { value: "pk-lf-abc:sk-lf-xyz" },
    });
    fireEvent.click(screen.getByRole("button", { name: /use token/i }));

    await waitFor(() => {
      expect(screen.getByText(/unencoded user:password/i)).toBeDefined();
    });
    expect(postV2DiscoverAvailableToolsOnAnMcpServer).not.toHaveBeenCalled();
    expect(postV2StoreABearerTokenForAnMcpServer).not.toHaveBeenCalled();
  });

  it("re-renders not-connected branch when manual token POST fails (forceDisconnected flips on)", async () => {
    // ``handleManualToken`` catch must flip ``forceDisconnected=true`` —
    // otherwise an existing live cred would re-show the Connected pill
    // even though the just-attempted manual-token store failed.
    setMockLiveCreds([
      { provider: "mcp", host: "https://mcp.example.com/mcp" },
    ]);
    const {
      postV2InitiateOauthLoginForAnMcpServer,
      postV2StoreABearerTokenForAnMcpServer,
    } = await import("@/app/api/__generated__/endpoints/mcp/mcp");
    vi.mocked(postV2InitiateOauthLoginForAnMcpServer).mockResolvedValueOnce({
      status: 400,
      data: { detail: "No OAuth" },
      headers: new Headers(),
    } as never);

    render(<MCPSetupCard output={makeSetupOutput(undefined, true)} />);
    fireEvent.click(screen.getByRole("button", { name: /reconnect/i }));
    await waitFor(() => {
      expect(screen.getByPlaceholderText(manualTokenPlaceholder)).toBeDefined();
    });

    // Token endpoint rejects with non-2xx → catch fires → forceDisconnected stays on.
    vi.mocked(postV2StoreABearerTokenForAnMcpServer).mockResolvedValueOnce({
      status: 422,
      data: { detail: "Invalid token format" },
      headers: new Headers(),
    } as never);

    fireEvent.change(screen.getByPlaceholderText(manualTokenPlaceholder), {
      target: { value: "bad-token" },
    });
    fireEvent.click(screen.getByRole("button", { name: /use token/i }));

    await waitFor(() => {
      expect(screen.getByText(/invalid token format/i)).toBeDefined();
    });
    // Crucial: live creds say "connected" but the failed token attempt
    // must keep the not-connected branch rendered so the user can retry.
    expect(screen.queryByText(/connected to example\.com/i)).toBeNull();
  });

  it("uses current provider and chat actions from a previously registered chain callback", async () => {
    const { postV2InitiateOauthLoginForAnMcpServer } = await import(
      "@/app/api/__generated__/endpoints/mcp/mcp"
    );
    const { openOAuthPopup } = await import("@/lib/oauth-popup");
    vi.mocked(postV2InitiateOauthLoginForAnMcpServer).mockResolvedValueOnce({
      status: 200,
      data: {
        login_url: "https://example.com/oauth",
        state_token: "latest-state",
      },
      headers: new Headers(),
    } as never);
    vi.mocked(openOAuthPopup).mockReturnValueOnce({
      promise: Promise.resolve({ code: "latest-code", state: "latest-state" }),
      cleanup: { abort: vi.fn(), signal: new AbortController().signal },
      popupBlocked: false,
      fallbackBlocked: false,
    });

    const initialProviderCallback = vi.fn().mockResolvedValue({});
    const latestProviderCallback = vi.fn().mockResolvedValue({});
    const initialOnSend = vi.fn();
    const latestOnSend = vi.fn();
    currentOnSend = initialOnSend;

    const initialProviders = {
      mcp: { mcpOAuthCallback: initialProviderCallback },
    } as unknown as CredentialsProvidersContextType;
    const latestProviders = {
      mcp: { mcpOAuthCallback: latestProviderCallback },
    } as unknown as CredentialsProvidersContextType;
    let registeredEntry: ChainActionEntry | null = null;
    const chainActions = {
      register: vi.fn((entry: ChainActionEntry) => {
        registeredEntry = entry;
      }),
      unregister: vi.fn((_id: string) => {}),
    };

    function setupCard(
      providers: CredentialsProvidersContextType,
      retryInstruction: string,
    ) {
      return (
        <CredentialsProvidersContext.Provider value={providers}>
          <ChainActionsContext.Provider value={chainActions}>
            <MCPSetupCard
              output={makeSetupOutput()}
              retryInstruction={retryInstruction}
            />
          </ChainActionsContext.Provider>
        </CredentialsProvidersContext.Provider>
      );
    }

    const { rerender } = render(
      setupCard(initialProviders, "Initial retry instruction"),
    );
    await waitFor(() => expect(registeredEntry?.mcp).toBeDefined());
    const previouslyRegisteredOnConnect = registeredEntry!.mcp!.onConnect;

    currentOnSend = latestOnSend;
    rerender(setupCard(latestProviders, "Latest retry instruction"));
    expect(chainActions.register).toHaveBeenCalledOnce();

    await act(async () => {
      previouslyRegisteredOnConnect();
    });

    await waitFor(() => {
      expect(latestProviderCallback).toHaveBeenCalledWith(
        "latest-code",
        "latest-state",
      );
      expect(latestOnSend).toHaveBeenCalledWith("Latest retry instruction");
    });
    expect(initialProviderCallback).not.toHaveBeenCalled();
    expect(initialOnSend).not.toHaveBeenCalled();
  });

  it("does not reinterpret an already prepared Basic chain credential", async () => {
    const { postV2StoreABearerTokenForAnMcpServer } = await import(
      "@/app/api/__generated__/endpoints/mcp/mcp"
    );
    vi.mocked(postV2StoreABearerTokenForAnMcpServer).mockResolvedValueOnce({
      status: 200,
      data: {
        id: "cred-chain-basic",
        provider: "mcp",
        type: "oauth2",
        title: "MCP: mcp.example.com",
        scopes: [],
      },
      headers: new Headers(),
    } as never);

    let registeredEntry: ChainActionEntry | null = null;
    const chainActions = {
      register: vi.fn((entry: ChainActionEntry) => {
        registeredEntry = entry;
      }),
      unregister: vi.fn(),
    };
    render(
      <ChainActionsContext.Provider value={chainActions}>
        <MCPSetupCard output={makeSetupOutput()} />
      </ChainActionsContext.Provider>,
    );
    await waitFor(() => expect(registeredEntry?.mcp).toBeDefined());

    await act(async () => {
      registeredEntry!.mcp!.onUseToken("Basic encoded-chain-value");
    });

    await waitFor(() => {
      expect(postV2StoreABearerTokenForAnMcpServer).toHaveBeenCalledWith({
        server_url: "https://mcp.example.com/mcp",
        token: "Basic encoded-chain-value",
      });
    });
  });
});
