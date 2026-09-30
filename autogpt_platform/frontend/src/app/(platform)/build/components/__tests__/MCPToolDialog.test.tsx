import {
  act,
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
  // Defaults to null — the browser-blocked case — so every cell that does not
  // care about the sign-in window behaves as it did before the window was
  // pre-opened at all.
  preOpenOAuthPopup: vi.fn(() => null),
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
    apiResponse(400, {
      detail: { code: "no_oauth", message: "OAuth not supported" },
    }),
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

  it("ignores an old initiation after closing and reopening the same dialog", async () => {
    const {
      postV2DiscoverAvailableToolsOnAnMcpServer,
      postV2InitiateOauthLoginForAnMcpServer,
    } = await import("@/app/api/__generated__/endpoints/mcp/mcp");
    const { openOAuthPopup, preOpenOAuthPopup } = await import(
      "@/lib/oauth-popup"
    );
    type LoginResponse = Awaited<
      ReturnType<typeof postV2InitiateOauthLoginForAnMcpServer>
    >;
    let firstResolve!: (response: LoginResponse) => void;
    let secondResolve!: (response: LoginResponse) => void;
    const first = new Promise<LoginResponse>((resolve) => {
      firstResolve = resolve;
    });
    const second = new Promise<LoginResponse>((resolve) => {
      secondResolve = resolve;
    });
    const firstWindow = { closed: false, close: vi.fn() };
    const secondWindow = { closed: false, close: vi.fn() };
    vi.mocked(preOpenOAuthPopup)
      .mockReturnValueOnce(firstWindow as unknown as Window)
      .mockReturnValueOnce(secondWindow as unknown as Window);
    vi.mocked(postV2DiscoverAvailableToolsOnAnMcpServer).mockResolvedValueOnce(
      apiResponse(401, { detail: "Authentication required" }),
    );
    vi.mocked(postV2InitiateOauthLoginForAnMcpServer)
      .mockReturnValueOnce(first)
      .mockReturnValueOnce(second);
    vi.mocked(openOAuthPopup).mockReturnValue({
      promise: new Promise(() => {}),
      cleanup: { abort: vi.fn(), signal: new AbortController().signal },
      popupBlocked: false,
      fallbackBlocked: false,
    });
    const props = { onClose: vi.fn(), onConfirm: vi.fn() };
    const view = render(<MCPToolDialog open {...props} />);
    fireEvent.change(screen.getByLabelText("Server URL"), {
      target: { value: PRIVATE_SERVER_URL },
    });
    fireEvent.click(screen.getByRole("button", { name: "Discover Tools" }));
    await waitFor(() =>
      expect(postV2InitiateOauthLoginForAnMcpServer).toHaveBeenCalledTimes(1),
    );

    view.rerender(<MCPToolDialog open={false} {...props} />);
    expect(firstWindow.close).toHaveBeenCalledOnce();
    view.rerender(<MCPToolDialog open {...props} />);
    fireEvent.click(screen.getByRole("button", { name: "Sign in & Connect" }));
    await waitFor(() =>
      expect(postV2InitiateOauthLoginForAnMcpServer).toHaveBeenCalledTimes(2),
    );
    await act(async () => {
      firstResolve(
        apiResponse(400, {
          detail: { code: "no_oauth", message: "Old attempt" },
        }),
      );
    });
    expect(secondWindow.close).not.toHaveBeenCalled();
    expect(screen.queryByLabelText("API token")).toBeNull();
    expect(
      screen.getByRole<HTMLButtonElement>("button", {
        name: "Waiting for sign-in...",
      }).disabled,
    ).toBe(true);

    await act(async () => {
      secondResolve(
        apiResponse(200, {
          login_url: "https://login.example.com/new",
          state_token: "new",
        }),
      );
    });
    await waitFor(() => expect(openOAuthPopup).toHaveBeenCalledOnce());
    expect(openOAuthPopup).toHaveBeenCalledWith(
      "https://login.example.com/new",
      expect.objectContaining({ preOpenedWindow: secondWindow }),
    );
  });

  it("surfaces a rejected authorization response instead of offering a token", async () => {
    // The callback answers 400 when the RFC 9207 ``iss`` is missing or does
    // not match the issuer bound at login.  That is a blocked mix-up, not an
    // unsupported server, so it must not fall back to manual token entry.
    const {
      postV2DiscoverAvailableToolsOnAnMcpServer,
      postV2InitiateOauthLoginForAnMcpServer,
      postV2ExchangeOauthCodeForMcpTokens,
    } = await import("@/app/api/__generated__/endpoints/mcp/mcp");
    const { openOAuthPopup } = await import("@/lib/oauth-popup");

    vi.mocked(postV2DiscoverAvailableToolsOnAnMcpServer).mockResolvedValueOnce(
      apiResponse(401, { detail: "Authentication required" }),
    );
    vi.mocked(postV2InitiateOauthLoginForAnMcpServer).mockResolvedValueOnce(
      apiResponse(200, {
        login_url: "https://auth.example.com/authorize",
        state_token: "st",
      }),
    );
    vi.mocked(openOAuthPopup).mockReturnValueOnce({
      promise: Promise.resolve({ code: "auth-code", state: "st" }),
      cleanup: { abort: vi.fn(), signal: new AbortController().signal },
      popupBlocked: false,
      fallbackBlocked: false,
    });
    vi.mocked(postV2ExchangeOauthCodeForMcpTokens).mockResolvedValueOnce(
      apiResponse(400, {
        detail:
          "Authorization response issuer does not match the authorization server this login was started with.",
      }),
    );

    render(<MCPToolDialog open onClose={() => {}} onConfirm={vi.fn()} />);
    fireEvent.change(screen.getByLabelText("Server URL"), {
      target: { value: PRIVATE_SERVER_URL },
    });
    fireEvent.click(screen.getByRole("button", { name: "Discover Tools" }));

    expect(await screen.findByText(/issuer does not match/i)).toBeDefined();
    expect(screen.queryByLabelText("API token")).toBeNull();
    expect(screen.queryByText(/does not support OAuth/)).toBeNull();
  });

  // #14532: the sign-in window has to be opened before the initiate request is
  // awaited — after an await iOS Safari blocks window.open() outright. Both
  // cells drive the auto-start path (discovery answers 401), which is the only
  // one this file's harness reaches; the button path shares the same code.
  it("opens the sign-in window before the initiate await and hands it over", async () => {
    const callOrder: string[] = [];
    const fakeWindow = { closed: false, close: vi.fn() };
    const {
      postV2DiscoverAvailableToolsOnAnMcpServer,
      postV2InitiateOauthLoginForAnMcpServer,
    } = await import("@/app/api/__generated__/endpoints/mcp/mcp");
    const { openOAuthPopup, preOpenOAuthPopup } = await import(
      "@/lib/oauth-popup"
    );

    vi.mocked(postV2DiscoverAvailableToolsOnAnMcpServer).mockResolvedValueOnce(
      apiResponse(401, { detail: "Authentication required" }),
    );
    vi.mocked(preOpenOAuthPopup).mockImplementation(() => {
      callOrder.push("preOpen");
      return fakeWindow as unknown as Window;
    });
    vi.mocked(postV2InitiateOauthLoginForAnMcpServer).mockImplementation(
      async () => {
        callOrder.push("initiate");
        return apiResponse(200, {
          login_url: "https://auth.example.com/authorize",
          state_token: "st",
        });
      },
    );
    vi.mocked(openOAuthPopup).mockReturnValue({
      promise: new Promise(() => {}),
      cleanup: { abort: vi.fn(), signal: new AbortController().signal },
      popupBlocked: false,
      fallbackBlocked: false,
    });

    render(<MCPToolDialog open onClose={() => {}} onConfirm={vi.fn()} />);
    fireEvent.change(screen.getByLabelText("Server URL"), {
      target: { value: PRIVATE_SERVER_URL },
    });
    fireEvent.click(screen.getByRole("button", { name: "Discover Tools" }));

    await waitFor(() => expect(vi.mocked(openOAuthPopup)).toHaveBeenCalled());
    // The ordering IS the fix — asserting only that it was called would pass
    // on a version that called it after the await, which is the bug.
    expect(callOrder).toEqual(["preOpen", "initiate"]);
    expect(vi.mocked(openOAuthPopup)).toHaveBeenCalledWith(
      "https://auth.example.com/authorize",
      expect.objectContaining({ preOpenedWindow: fakeWindow }),
    );
    expect(fakeWindow.close).not.toHaveBeenCalled();
  });

  it("closes the sign-in window when the server has no OAuth", async () => {
    const fakeWindow = { closed: false, close: vi.fn() };
    const {
      postV2DiscoverAvailableToolsOnAnMcpServer,
      postV2InitiateOauthLoginForAnMcpServer,
    } = await import("@/app/api/__generated__/endpoints/mcp/mcp");
    const { preOpenOAuthPopup } = await import("@/lib/oauth-popup");

    vi.mocked(postV2DiscoverAvailableToolsOnAnMcpServer).mockResolvedValueOnce(
      apiResponse(401, { detail: "Authentication required" }),
    );
    vi.mocked(preOpenOAuthPopup).mockReturnValue(
      fakeWindow as unknown as Window,
    );
    vi.mocked(postV2InitiateOauthLoginForAnMcpServer).mockResolvedValueOnce(
      apiResponse(400, {
        detail: { code: "no_oauth", message: "OAuth not supported" },
      }),
    );

    render(<MCPToolDialog open onClose={() => {}} onConfirm={vi.fn()} />);
    fireEvent.change(screen.getByLabelText("Server URL"), {
      target: { value: PRIVATE_SERVER_URL },
    });
    fireEvent.click(screen.getByRole("button", { name: "Discover Tools" }));

    // openOAuthPopup never runs on this path, so nothing else can reach the
    // about:blank window it left behind.
    await waitFor(() => expect(fakeWindow.close).toHaveBeenCalled());
    expect(await screen.findByLabelText("API token")).toBeDefined();
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
      apiResponse(400, {
        detail: { code: "no_oauth", message: "OAuth not supported" },
      }),
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
      apiResponse(400, {
        detail: { code: "no_oauth", message: "OAuth not supported" },
      }),
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
      apiResponse(400, {
        detail: { code: "no_oauth", message: "OAuth not supported" },
      }),
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
      apiResponse(400, {
        detail: { code: "no_oauth", message: "OAuth not supported" },
      }),
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
