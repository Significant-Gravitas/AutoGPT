import {
  render,
  screen,
  cleanup,
  fireEvent,
  waitFor,
} from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { BlockIOCredentialsSubSchema } from "@/lib/autogpt-server-api";
import { CredentialsInput } from "../CredentialsInput";

vi.mock("@/hooks/useCredentials", () => ({ default: vi.fn() }));
vi.mock("@/lib/autogpt-server-api/context", () => ({
  useBackendAPI: vi.fn(),
  BackendAPIProvider: ({ children }: { children: React.ReactNode }) => children,
}));
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: vi.fn(),
  useToast: () => ({ toast: vi.fn(), dismiss: vi.fn(), toasts: [] }),
}));
vi.mock("@/lib/oauth-popup", () => ({
  openOAuthPopup: vi.fn(),
  preOpenOAuthPopup: vi.fn(() => null),
  OAUTH_ERROR_WINDOW_CLOSED: "Sign-in window was closed",
  OAUTH_ERROR_FLOW_CANCELED: "OAuth flow was canceled",
  OAUTH_ERROR_FLOW_TIMED_OUT: "OAuth flow timed out",
  OAUTH_ERROR_POPUP_BLOCKED:
    "Popup blocked — the sign-in window opened in a new tab instead. If you don't see it, allow popups for this site and retry.",
}));
vi.mock("@/app/api/__generated__/endpoints/mcp/mcp", () => ({
  postV2InitiateOauthLoginForAnMcpServer: vi.fn(),
}));

import { toast } from "@/components/molecules/Toast/use-toast";
import useCredentials from "@/hooks/useCredentials";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { openOAuthPopup, preOpenOAuthPopup } from "@/lib/oauth-popup";

const mockUseCredentials = useCredentials as unknown as ReturnType<
  typeof vi.fn
>;
const mockUseBackendAPI = useBackendAPI as unknown as ReturnType<typeof vi.fn>;
const mockOpenOAuthPopup = openOAuthPopup as unknown as ReturnType<
  typeof vi.fn
>;
const mockPreOpenOAuthPopup = preOpenOAuthPopup as unknown as ReturnType<
  typeof vi.fn
>;
const mockToast = toast as unknown as ReturnType<typeof vi.fn>;

const baseSchema: BlockIOCredentialsSubSchema = {
  credentials_provider: ["google"],
  credentials_types: ["oauth2"],
  credentials_scopes: ["drive.file", "drive.metadata"],
} as BlockIOCredentialsSubSchema;

type CredentialsReturn = ReturnType<typeof useCredentials>;
type BackendAPI = ReturnType<typeof useBackendAPI>;

function makeCredentialsReturn(overrides: Partial<CredentialsReturn> = {}) {
  return {
    provider: "google",
    providerName: "Google",
    savedCredentials: [],
    upgradeableCredentials: [],
    supportsApiKey: false,
    supportsOAuth2: true,
    supportsUserPassword: false,
    supportsHostScoped: false,
    isLoading: false,
    isSystemProvider: false,
    schema: baseSchema,
    oAuthCallback: vi.fn().mockResolvedValue({
      id: "new-cred",
      type: "oauth2",
      title: "Connected Google",
      provider: "google",
      scopes: ["drive.file", "drive.metadata"],
    }),
    mcpOAuthCallback: vi.fn(),
    createAPIKeyCredentials: vi.fn(),
    createUserPasswordCredentials: vi.fn(),
    createHostScopedCredentials: vi.fn(),
    deleteCredentials: vi.fn(),
    discriminatorValue: undefined,
    ...overrides,
  };
}

function makeBackendAPI(overrides: Partial<BackendAPI> = {}) {
  return {
    oAuthLogin: vi.fn().mockResolvedValue({
      login_url: "https://accounts.google.com/o/oauth2/auth",
      state_token: "state-xyz",
    }),
    onWebSocketMessage: vi.fn().mockReturnValue(() => {}),
    connectWebSocket: vi.fn().mockResolvedValue(undefined),
    sendWebSocketMessage: vi.fn(),
    ...overrides,
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  mockUseBackendAPI.mockReturnValue(makeBackendAPI());
});

afterEach(() => {
  cleanup();
});

describe("CredentialsInput – OAuth flow", () => {
  it("clicking the Add account button calls oAuthLogin without a credentialID", async () => {
    const oAuthLoginMock = vi.fn().mockResolvedValue({
      login_url: "https://accounts.google.com/o/oauth2/auth",
      state_token: "state-xyz",
    });
    mockUseBackendAPI.mockReturnValue(
      makeBackendAPI({ oAuthLogin: oAuthLoginMock }),
    );

    mockUseCredentials.mockReturnValue(makeCredentialsReturn());

    mockOpenOAuthPopup.mockReturnValue({
      promise: Promise.resolve({ code: "code-2", state: "state-xyz" }),
      cleanup: { abort: vi.fn() },
    });

    render(
      <CredentialsInput
        schema={baseSchema}
        onSelectCredentials={vi.fn()}
        showTitle={false}
      />,
    );

    const addAccountButton = await screen.findByRole("button", {
      name: /add account/i,
    });
    fireEvent.click(addAccountButton);

    await waitFor(() => {
      expect(oAuthLoginMock).toHaveBeenCalledWith(
        "google",
        ["drive.file", "drive.metadata"],
        undefined,
      );
    });
  });

  it("pre-opens the window before the oAuthLogin await and passes it to openOAuthPopup", async () => {
    const fakeWindow = { closed: false, close: vi.fn() };
    const callOrder: string[] = [];
    mockPreOpenOAuthPopup.mockImplementation(() => {
      callOrder.push("preOpen");
      return fakeWindow;
    });

    const oAuthLoginMock = vi.fn().mockImplementation(async () => {
      callOrder.push("initiate");
      return {
        login_url: "https://accounts.google.com/o/oauth2/auth",
        state_token: "state-xyz",
      };
    });
    mockUseBackendAPI.mockReturnValue(
      makeBackendAPI({ oAuthLogin: oAuthLoginMock }),
    );
    mockUseCredentials.mockReturnValue(makeCredentialsReturn());
    mockOpenOAuthPopup.mockReturnValue({
      promise: Promise.resolve({ code: "code-2", state: "state-xyz" }),
      cleanup: { abort: vi.fn() },
      popupBlocked: false,
      fallbackBlocked: false,
    });

    render(
      <CredentialsInput
        schema={baseSchema}
        onSelectCredentials={vi.fn()}
        showTitle={false}
      />,
    );

    fireEvent.click(
      await screen.findByRole("button", { name: /add account/i }),
    );
    await waitFor(() => expect(mockOpenOAuthPopup).toHaveBeenCalled());

    // The window must be opened synchronously, before the login-URL request —
    // after an await iOS Safari blocks every window.open().
    expect(callOrder).toEqual(["preOpen", "initiate"]);
    expect(mockOpenOAuthPopup).toHaveBeenCalledWith(
      "https://accounts.google.com/o/oauth2/auth",
      expect.objectContaining({
        stateToken: "state-xyz",
        preOpenedWindow: fakeWindow,
        useCrossOriginListeners: true,
      }),
    );
  });

  it("shows the blocked-popup toast and modal copy when the popup is blocked", async () => {
    mockPreOpenOAuthPopup.mockReturnValue(null);
    mockUseCredentials.mockReturnValue(makeCredentialsReturn());
    mockOpenOAuthPopup.mockReturnValue({
      promise: new Promise(() => {}), // flow stays in flight
      cleanup: { abort: vi.fn() },
      popupBlocked: true,
      fallbackBlocked: false,
    });

    render(
      <CredentialsInput
        schema={baseSchema}
        onSelectCredentials={vi.fn()}
        showTitle={false}
      />,
    );

    fireEvent.click(
      await screen.findByRole("button", { name: /add account/i }),
    );

    await waitFor(() =>
      expect(mockToast).toHaveBeenCalledWith(
        expect.objectContaining({ title: "Popup blocked" }),
      ),
    );
    // The waiting modal must direct the user to the fallback tab instead of
    // to a popup that doesn't exist.
    expect(
      await screen.findByText(/blocked the sign-in window/i),
    ).toBeDefined();
  });

  it("closes the pre-opened window when the login-URL request fails", async () => {
    const fakeWindow = { closed: false, close: vi.fn() };
    mockPreOpenOAuthPopup.mockReturnValue(fakeWindow);

    const oAuthLoginMock = vi
      .fn()
      .mockRejectedValue(new Error("provider not configured"));
    mockUseBackendAPI.mockReturnValue(
      makeBackendAPI({ oAuthLogin: oAuthLoginMock }),
    );
    mockUseCredentials.mockReturnValue(makeCredentialsReturn());

    render(
      <CredentialsInput
        schema={baseSchema}
        onSelectCredentials={vi.fn()}
        showTitle={false}
      />,
    );

    fireEvent.click(
      await screen.findByRole("button", { name: /add account/i }),
    );

    // The failure happens before openOAuthPopup adopts the window, so the
    // flow still owns the dangling about:blank window and must close it.
    await waitFor(() => expect(fakeWindow.close).toHaveBeenCalled());
    expect(mockOpenOAuthPopup).not.toHaveBeenCalled();
  });

  it("closes the pre-opened window and skips adoption when unmounted mid-initiation", async () => {
    const fakeWindow = { closed: false, close: vi.fn() };
    mockPreOpenOAuthPopup.mockReturnValue(fakeWindow);

    let resolveLogin: (value: unknown) => void = () => {};
    const oAuthLoginMock = vi.fn().mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveLogin = resolve;
        }),
    );
    mockUseBackendAPI.mockReturnValue(
      makeBackendAPI({ oAuthLogin: oAuthLoginMock }),
    );
    mockUseCredentials.mockReturnValue(makeCredentialsReturn());
    mockOpenOAuthPopup.mockReturnValue({
      promise: Promise.resolve({ code: "code-2", state: "state-xyz" }),
      cleanup: { abort: vi.fn() },
    });

    const { unmount } = render(
      <CredentialsInput
        schema={baseSchema}
        onSelectCredentials={vi.fn()}
        showTitle={false}
      />,
    );

    fireEvent.click(
      await screen.findByRole("button", { name: /add account/i }),
    );
    await waitFor(() => expect(oAuthLoginMock).toHaveBeenCalled());

    // Unmount while the login-URL request is still in flight — the cleanup
    // must close the pre-opened window immediately.
    unmount();
    expect(fakeWindow.close).toHaveBeenCalled();

    // When the request resolves, the stale continuation must not adopt the
    // window into a new OAuth popup.
    resolveLogin({
      login_url: "https://accounts.google.com/o/oauth2/auth",
      state_token: "state-xyz",
    });
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(mockOpenOAuthPopup).not.toHaveBeenCalled();
  });

  it("a superseded flow does not clear the newer flow's abort handler", async () => {
    const windowA = { closed: false, close: vi.fn() };
    const windowB = { closed: false, close: vi.fn() };
    mockPreOpenOAuthPopup
      .mockReturnValueOnce(windowA)
      .mockReturnValueOnce(windowB);

    let resolveA: (value: unknown) => void = () => {};
    let resolveB: (value: unknown) => void = () => {};
    const oAuthLoginMock = vi
      .fn()
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveA = resolve;
          }),
      )
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveB = resolve;
          }),
      );
    mockUseBackendAPI.mockReturnValue(
      makeBackendAPI({ oAuthLogin: oAuthLoginMock }),
    );
    mockUseCredentials.mockReturnValue(makeCredentialsReturn());

    const abortB = vi.fn();
    mockOpenOAuthPopup.mockReturnValue({
      promise: new Promise(() => {}), // flow stays in flight
      cleanup: { abort: abortB },
    });

    const { unmount } = render(
      <CredentialsInput
        schema={baseSchema}
        onSelectCredentials={vi.fn()}
        showTitle={false}
      />,
    );

    const addAccountButton = await screen.findByRole("button", {
      name: /add account/i,
    });

    // Flow A starts, then flow B supersedes it while A's request is pending.
    fireEvent.click(addAccountButton);
    await waitFor(() => expect(oAuthLoginMock).toHaveBeenCalledTimes(1));
    fireEvent.click(addAccountButton);
    await waitFor(() => expect(oAuthLoginMock).toHaveBeenCalledTimes(2));

    // Starting B closes A's still-pending pre-opened window.
    expect(windowA.close).toHaveBeenCalled();

    // B's request resolves first and registers its abort handler.
    resolveB({
      login_url: "https://accounts.google.com/o/oauth2/auth",
      state_token: "state-xyz",
    });
    await waitFor(() => expect(mockOpenOAuthPopup).toHaveBeenCalledTimes(1));

    // A's request resolves late — its continuation must bail without
    // nulling B's abort handler in its finally block.
    resolveA({
      login_url: "https://accounts.google.com/o/oauth2/auth",
      state_token: "state-xyz",
    });
    await new Promise((resolve) => setTimeout(resolve, 0));

    // Unmount must still reach B's abort — i.e. A's finally did not clear it.
    unmount();
    expect(abortB).toHaveBeenCalled();
  });
});

describe("CredentialsInput – MCP API key creation", () => {
  const mcpSchema: BlockIOCredentialsSubSchema = {
    credentials_provider: ["mcp"],
    credentials_types: ["oauth2", "api_key"],
    discriminator: "server_url",
  } as BlockIOCredentialsSubSchema;

  function renderMCPApiKeyTab(
    createAPIKeyCredentials: ReturnType<typeof vi.fn>,
  ) {
    mockUseCredentials.mockReturnValue(
      makeCredentialsReturn({
        provider: "mcp",
        providerName: "MCP",
        schema: mcpSchema,
        supportsApiKey: true,
        supportsOAuth2: true,
        discriminatorValue: "https://mcp.datafa.st/mcp/",
        createAPIKeyCredentials,
      } as unknown as Partial<CredentialsReturn>) as unknown as CredentialsReturn,
    );

    render(
      <CredentialsInput
        schema={mcpSchema}
        onSelectCredentials={vi.fn()}
        siblingInputs={{ server_url: "https://mcp.datafa.st/mcp/" }}
        showTitle={false}
      />,
    );
  }

  it("tags a new API key with the normalized MCP server URL", async () => {
    // Without `metadata.mcp_server_url` the backend returns `host: null`, the
    // picker filters the credential out, and the selection made on create is
    // wiped again — the user sees "adding a credential does nothing".
    const createAPIKeyCredentials = vi.fn().mockResolvedValue({
      id: "new-mcp-cred",
      title: "My token",
      provider: "mcp",
      type: "api_key",
    });
    renderMCPApiKeyTab(createAPIKeyCredentials);

    fireEvent.click(
      await screen.findByRole("button", { name: /add .*credential/i }),
    );
    // Radix Tabs activate on mousedown, not click.
    fireEvent.mouseDown(await screen.findByRole("tab", { name: /api key/i }));

    fireEvent.change(await screen.findByLabelText(/^name$/i), {
      target: { value: "My token" },
    });
    fireEvent.change(
      screen.getByLabelText(/^api key$/i, { selector: "input" }),
      {
        target: { value: "wrong-token" },
      },
    );
    fireEvent.click(screen.getByRole("button", { name: /add api key/i }));

    await waitFor(() => {
      expect(createAPIKeyCredentials).toHaveBeenCalledWith(
        expect.objectContaining({
          api_key: "wrong-token",
          metadata: { mcp_server_url: "https://mcp.datafa.st/mcp" },
        }),
      );
    });
  });
});
