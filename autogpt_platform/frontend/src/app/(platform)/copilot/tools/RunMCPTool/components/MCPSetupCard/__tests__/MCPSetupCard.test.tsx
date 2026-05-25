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
      expect(screen.getByPlaceholderText("Paste API token")).toBeDefined();
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
      expect(screen.getByPlaceholderText("Paste API token")).toBeDefined();
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
      expect(screen.getByPlaceholderText("Paste API token")).toBeDefined();
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
    expect(screen.queryByPlaceholderText("Paste API token")).toBeNull();
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
      expect(screen.getByPlaceholderText("Paste API token")).toBeDefined();
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

    const input = screen.getByPlaceholderText("Paste API token");
    fireEvent.change(input, { target: { value: "my-token" } });
    fireEvent.keyDown(input, { key: "Enter" });

    await waitFor(() => {
      expect(screen.getByText(/connected to example\.com/i)).toBeDefined();
    });
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
      expect(screen.getByPlaceholderText("Paste API token")).toBeDefined();
    });

    // Token endpoint rejects with non-2xx → catch fires → forceDisconnected stays on.
    vi.mocked(postV2StoreABearerTokenForAnMcpServer).mockResolvedValueOnce({
      status: 422,
      data: { detail: "Invalid token format" },
      headers: new Headers(),
    } as never);

    fireEvent.change(screen.getByPlaceholderText("Paste API token"), {
      target: { value: "bad-token" },
    });
    fireEvent.click(screen.getByRole("button", { name: /use token/i }));

    await waitFor(() => {
      // ``handleManualToken`` throws ``new Error("Failed to store token")``
      // on non-2xx, which becomes the displayed error message.
      expect(screen.getByText(/failed to store token/i)).toBeDefined();
    });
    // Crucial: live creds say "connected" but the failed token attempt
    // must keep the not-connected branch rendered so the user can retry.
    expect(screen.queryByText(/connected to example\.com/i)).toBeNull();
  });
});
