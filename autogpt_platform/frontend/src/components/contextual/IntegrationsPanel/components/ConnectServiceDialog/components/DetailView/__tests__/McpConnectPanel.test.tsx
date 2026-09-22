import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { McpConnectPanel } from "../McpConnectPanel";

vi.mock("@/lib/oauth-popup", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/oauth-popup")>()),
  openOAuthPopup: vi.fn(),
  // Defaults to null — the browser-blocked case — so every cell that does not
  // care about the sign-in window behaves as it did before the window was
  // pre-opened at all.
  preOpenOAuthPopup: vi.fn(() => null),
}));

vi.mock("@/app/api/__generated__/endpoints/mcp/mcp", () => ({
  postV2DiscoverAvailableToolsOnAnMcpServer: vi.fn(),
  postV2InitiateOauthLoginForAnMcpServer: vi.fn(),
  postV2ExchangeOauthCodeForMcpTokens: vi.fn(),
  postV2StoreABearerTokenForAnMcpServer: vi.fn(),
}));

let mockSavedCredentials: Array<{
  provider: string;
  host?: string | null;
  mcp_auth_scheme?: "basic" | "bearer" | null;
}> = [];
vi.mock("@/app/api/__generated__/endpoints/integrations/integrations", () => ({
  getGetV1ListCredentialsQueryKey: () => ["credentials"],
  useGetV1ListCredentials: () => ({ data: mockSavedCredentials }),
}));

// What the login route actually answers when a server has no OAuth at all:
// prose for the user, plus the code the panel branches on. Every other 400 it
// writes is a different failure and must not offer the manual-token form.
const noOAuthDetail = {
  detail: {
    code: "no_oauth",
    message: "This MCP server does not advertise OAuth support.",
  },
};

function makeApiError(status: number, detail: unknown = "boom"): Error {
  // `parseApiError` already unwraps a structured detail into the message, so
  // mirror that here: the prose on `message`, the whole body on `response`.
  const message =
    typeof detail === "string"
      ? detail
      : ((detail as { message?: string })?.message ?? "boom");
  const err = new Error(message) as Error & {
    status: number;
    response: unknown;
  };
  err.name = "ApiError";
  err.status = status;
  err.response = { detail };
  return err;
}

// The placeholder tracks the selected scheme: "Paste API token" under Bearer,
// and the Base64 wording under Basic, so it stops restating the mistake the
// hint below it exists to prevent. These queries match either.
const manualTokenPlaceholder = /paste (api token|base64 of user:password)/i;

describe("McpConnectPanel", () => {
  // Saving a manual credential probes the server first, so the default is an
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
    vi.resetAllMocks();
    mockSavedCredentials = [];
    cleanup();
  });

  it("disables Connect until a valid http(s) URL is entered", () => {
    render(<McpConnectPanel onSuccess={() => {}} />);

    const connectButton = screen.getByRole("button", { name: /connect/i });
    expect((connectButton as HTMLButtonElement).disabled).toBe(true);

    const urlInput = screen.getByLabelText(/server url/i);

    fireEvent.change(urlInput, { target: { value: "not a url" } });
    expect((connectButton as HTMLButtonElement).disabled).toBe(true);

    fireEvent.change(urlInput, {
      target: { value: "javascript:alert(1)" },
    });
    expect((connectButton as HTMLButtonElement).disabled).toBe(true);

    fireEvent.change(urlInput, {
      target: { value: "https://mcp.example.com" },
    });
    expect((connectButton as HTMLButtonElement).disabled).toBe(false);
  });

  // #14532: the sign-in window has to be opened inside the tap, before the
  // initiate request is awaited. iOS Safari discards the gesture context at the
  // first async break and then blocks window.open() outright — including the
  // new-tab fallback — so on mobile nothing opened at all.
  describe("user activation (#14532)", () => {
    async function mockInitiateOk(record?: string[]) {
      const { postV2InitiateOauthLoginForAnMcpServer } = await import(
        "@/app/api/__generated__/endpoints/mcp/mcp"
      );
      vi.mocked(postV2InitiateOauthLoginForAnMcpServer).mockImplementation(
        async () => {
          record?.push("initiate");
          return {
            status: 200,
            data: {
              login_url: "https://login.example.com",
              state_token: "tok",
            },
            headers: new Headers(),
          } as never;
        },
      );
    }

    function clickConnect() {
      fireEvent.change(screen.getByLabelText(/server url/i), {
        target: { value: "https://mcp.example.com" },
      });
      fireEvent.click(screen.getByRole("button", { name: /connect/i }));
    }

    it("opens the window before the initiate await and hands it to openOAuthPopup", async () => {
      const callOrder: string[] = [];
      const fakeWindow = { closed: false, close: vi.fn() };
      const { openOAuthPopup, preOpenOAuthPopup } = await import(
        "@/lib/oauth-popup"
      );
      vi.mocked(preOpenOAuthPopup).mockImplementation(() => {
        callOrder.push("preOpen");
        return fakeWindow as unknown as Window;
      });
      vi.mocked(openOAuthPopup).mockReturnValue({
        promise: new Promise(() => {}),
        cleanup: { abort: vi.fn() },
      } as never);
      await mockInitiateOk(callOrder);

      render(<McpConnectPanel onSuccess={() => {}} />);
      clickConnect();

      await waitFor(() => expect(vi.mocked(openOAuthPopup)).toHaveBeenCalled());
      // The ordering IS the fix. Asserting only that preOpenOAuthPopup was
      // called would pass on a version that called it after the await, which
      // is the bug.
      expect(callOrder).toEqual(["preOpen", "initiate"]);
      expect(vi.mocked(openOAuthPopup)).toHaveBeenCalledWith(
        "https://login.example.com",
        expect.objectContaining({
          stateToken: "tok",
          preOpenedWindow: fakeWindow,
          useCrossOriginListeners: true,
        }),
      );
      // Ownership moved to the helper, which closes it on abort.
      expect(fakeWindow.close).not.toHaveBeenCalled();
    });

    it("closes the window when the server turns out not to support OAuth", async () => {
      const fakeWindow = { closed: false, close: vi.fn() };
      const { preOpenOAuthPopup } = await import("@/lib/oauth-popup");
      vi.mocked(preOpenOAuthPopup).mockReturnValue(
        fakeWindow as unknown as Window,
      );
      const { postV2InitiateOauthLoginForAnMcpServer } = await import(
        "@/app/api/__generated__/endpoints/mcp/mcp"
      );
      vi.mocked(postV2InitiateOauthLoginForAnMcpServer).mockRejectedValueOnce(
        makeApiError(400, noOAuthDetail.detail),
      );

      render(<McpConnectPanel onSuccess={() => {}} />);
      clickConnect();

      // The 400 returns early from inside the try, before openOAuthPopup ever
      // runs — an about:blank window would otherwise sit there for good.
      await waitFor(() => expect(fakeWindow.close).toHaveBeenCalled());
      expect(screen.getByPlaceholderText(/paste api token/i)).toBeDefined();
    });

    it("closes the window when the initiate request fails outright", async () => {
      const fakeWindow = { closed: false, close: vi.fn() };
      const { preOpenOAuthPopup } = await import("@/lib/oauth-popup");
      vi.mocked(preOpenOAuthPopup).mockReturnValue(
        fakeWindow as unknown as Window,
      );
      const { postV2InitiateOauthLoginForAnMcpServer } = await import(
        "@/app/api/__generated__/endpoints/mcp/mcp"
      );
      vi.mocked(postV2InitiateOauthLoginForAnMcpServer).mockRejectedValueOnce(
        makeApiError(500, "server exploded"),
      );

      render(<McpConnectPanel onSuccess={() => {}} />);
      clickConnect();

      await waitFor(() => expect(fakeWindow.close).toHaveBeenCalled());
    });

    it("closes the window when the panel unmounts mid-initiation", async () => {
      const fakeWindow = { closed: false, close: vi.fn() };
      const { openOAuthPopup, preOpenOAuthPopup } = await import(
        "@/lib/oauth-popup"
      );
      vi.mocked(preOpenOAuthPopup).mockReturnValue(
        fakeWindow as unknown as Window,
      );
      const { postV2InitiateOauthLoginForAnMcpServer } = await import(
        "@/app/api/__generated__/endpoints/mcp/mcp"
      );
      let release: (() => void) | undefined;
      vi.mocked(postV2InitiateOauthLoginForAnMcpServer).mockImplementation(
        () =>
          new Promise((resolve) => {
            release = () =>
              resolve({
                status: 200,
                data: {
                  login_url: "https://login.example.com",
                  state_token: "tok",
                },
                headers: new Headers(),
              } as never);
          }),
      );

      const { unmount } = render(<McpConnectPanel onSuccess={() => {}} />);
      clickConnect();
      await waitFor(() =>
        expect(vi.mocked(preOpenOAuthPopup)).toHaveBeenCalled(),
      );

      // The dialog closes while the login URL is still in flight. Nothing has
      // registered an abort yet, so the unmount cleanup is the only thing that
      // can reach this window.
      unmount();
      expect(fakeWindow.close).toHaveBeenCalled();

      release?.();
      await Promise.resolve();
      // And the continuation must not adopt a window that is already gone.
      expect(vi.mocked(openOAuthPopup)).not.toHaveBeenCalled();
    });

    it("a double tap starts one flow, not two", async () => {
      const fakeWindow = { closed: false, close: vi.fn() };
      const { openOAuthPopup, preOpenOAuthPopup } = await import(
        "@/lib/oauth-popup"
      );
      vi.mocked(preOpenOAuthPopup).mockReturnValue(
        fakeWindow as unknown as Window,
      );
      vi.mocked(openOAuthPopup).mockReturnValue({
        promise: new Promise(() => {}),
        cleanup: { abort: vi.fn() },
      } as never);
      await mockInitiateOk();

      render(<McpConnectPanel onSuccess={() => {}} />);
      fireEvent.change(screen.getByLabelText(/server url/i), {
        target: { value: "https://mcp.example.com" },
      });
      const button = screen.getByRole("button", { name: /connect/i });
      // Both taps inside ONE act() batch. fireEvent flushes React state
      // between events, so two fireEvent.click calls let the second one see
      // `isSubmitting === true` and the disabled-button guard already holds —
      // which is exactly the flush a real double-tap inside one frame does not
      // get. Dispatching both before the flush is what reproduces it.
      await act(async () => {
        button.click();
        button.click();
      });

      await waitFor(() => expect(vi.mocked(openOAuthPopup)).toHaveBeenCalled());
      // Two windows would mean the second flow overwrote the first's handle
      // and the first one could never be closed.
      expect(vi.mocked(preOpenOAuthPopup)).toHaveBeenCalledTimes(1);
    });
  });

  it("allows manual authentication without first attempting OAuth", async () => {
    const { postV2InitiateOauthLoginForAnMcpServer } = await import(
      "@/app/api/__generated__/endpoints/mcp/mcp"
    );
    render(
      <McpConnectPanel
        onSuccess={() => {}}
        initialServerURL="https://mcp.example.com"
      />,
    );
    fireEvent.click(
      screen.getByRole("button", { name: /use an api token instead/i }),
    );
    expect(screen.getByPlaceholderText(manualTokenPlaceholder)).toBeDefined();
    expect(postV2InitiateOauthLoginForAnMcpServer).not.toHaveBeenCalled();
  });

  it("can cancel a pending OAuth popup and use a manual token", async () => {
    const {
      postV2InitiateOauthLoginForAnMcpServer,
      postV2ExchangeOauthCodeForMcpTokens,
    } = await import("@/app/api/__generated__/endpoints/mcp/mcp");
    const { openOAuthPopup } = await import("@/lib/oauth-popup");
    vi.mocked(postV2InitiateOauthLoginForAnMcpServer).mockResolvedValueOnce({
      status: 200,
      data: { login_url: "https://login.example.com", state_token: "tok" },
      headers: new Headers(),
    } as never);
    const controller = new AbortController();
    const abort = vi.fn(() => controller.abort());
    vi.mocked(openOAuthPopup).mockImplementation(() => ({
      promise: new Promise((_resolve, reject) =>
        controller.signal.addEventListener("abort", () =>
          reject(new Error("OAuth flow was canceled")),
        ),
      ),
      cleanup: { abort, signal: controller.signal },
      popupBlocked: false,
      fallbackBlocked: false,
    }));
    render(
      <McpConnectPanel
        onSuccess={() => {}}
        initialServerURL="https://mcp.example.com"
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /^connect$/i }));
    await waitFor(() => expect(openOAuthPopup).toHaveBeenCalled());
    const manual = screen.getByRole<HTMLButtonElement>("button", {
      name: /use an api token instead/i,
    });
    expect(manual.disabled).toBe(false);
    fireEvent.click(manual);
    await waitFor(() =>
      expect(
        screen.getByPlaceholderText<HTMLInputElement>(manualTokenPlaceholder)
          .disabled,
      ).toBe(false),
    );
    expect(abort).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole("alert")).toBeNull();
    expect(postV2ExchangeOauthCodeForMcpTokens).not.toHaveBeenCalled();
  });

  it("falls back to manual-token form when initiate returns 400", async () => {
    const { postV2InitiateOauthLoginForAnMcpServer } = await import(
      "@/app/api/__generated__/endpoints/mcp/mcp"
    );

    vi.mocked(postV2InitiateOauthLoginForAnMcpServer).mockResolvedValueOnce({
      status: 400,
      data: noOAuthDetail,
      headers: new Headers(),
    } as never);

    render(<McpConnectPanel onSuccess={() => {}} />);

    fireEvent.change(screen.getByLabelText(/server url/i), {
      target: { value: "https://mcp.example.com" },
    });
    fireEvent.click(screen.getByRole("button", { name: /connect/i }));

    await waitFor(() => {
      expect(screen.getByPlaceholderText(manualTokenPlaceholder)).toBeDefined();
    });
    // The route's own wording, not a generic stand-in: it names the reason
    // this particular server cannot be signed into.
    expect(screen.getByText(/does not advertise oauth support/i)).toBeDefined();
  });

  it("offers the manual form for a catalog service that has no OAuth", async () => {
    const { postV2InitiateOauthLoginForAnMcpServer } = await import(
      "@/app/api/__generated__/endpoints/mcp/mcp"
    );

    // A catalog entry without OAuth is rejected in quite different words from
    // the "does not advertise OAuth" case above. Both mean the same thing to
    // this panel, and only the code says so.
    vi.mocked(postV2InitiateOauthLoginForAnMcpServer).mockResolvedValueOnce({
      status: 400,
      data: {
        detail: {
          code: "no_oauth",
          message:
            "Brevo uses bearer / basic authentication. Create an API key in " +
            "your Brevo dashboard.",
        },
      },
      headers: new Headers(),
    } as never);

    render(<McpConnectPanel onSuccess={() => {}} />);

    fireEvent.change(screen.getByLabelText(/server url/i), {
      target: { value: "https://mcp.example.com" },
    });
    fireEvent.click(screen.getByRole("button", { name: /connect/i }));

    await waitFor(() => {
      expect(screen.getByPlaceholderText(manualTokenPlaceholder)).toBeDefined();
    });
    expect(screen.getByText(/create an api key/i)).toBeDefined();
  });

  it("keeps the OAuth form for a 400 that is not about missing OAuth", async () => {
    const { postV2InitiateOauthLoginForAnMcpServer } = await import(
      "@/app/api/__generated__/endpoints/mcp/mcp"
    );

    // A failed client registration is a problem with this attempt, on a server
    // that does support OAuth. Sending the user off to find an API token would
    // be the wrong advice.
    vi.mocked(postV2InitiateOauthLoginForAnMcpServer).mockResolvedValueOnce({
      status: 400,
      data: {
        detail: "Could not register an OAuth client with this MCP server.",
      },
      headers: new Headers(),
    } as never);

    render(<McpConnectPanel onSuccess={() => {}} />);

    fireEvent.change(screen.getByLabelText(/server url/i), {
      target: { value: "https://mcp.example.com" },
    });
    fireEvent.click(screen.getByRole("button", { name: /connect/i }));

    await waitFor(() => {
      expect(
        screen.getByText(/could not register an oauth client/i),
      ).toBeDefined();
    });
    expect(screen.queryByPlaceholderText(manualTokenPlaceholder)).toBeNull();
  });

  it("does NOT switch to manual-token on a 400 from token exchange", async () => {
    const {
      postV2InitiateOauthLoginForAnMcpServer,
      postV2ExchangeOauthCodeForMcpTokens,
    } = await import("@/app/api/__generated__/endpoints/mcp/mcp");
    const { openOAuthPopup } = await import("@/lib/oauth-popup");

    vi.mocked(postV2InitiateOauthLoginForAnMcpServer).mockResolvedValueOnce({
      status: 200,
      data: { login_url: "https://login.example.com", state_token: "tok" },
      headers: new Headers(),
    } as never);

    vi.mocked(openOAuthPopup).mockReturnValueOnce({
      promise: Promise.resolve({ code: "abc" }),
      cleanup: { abort: vi.fn() },
    } as never);

    vi.mocked(postV2ExchangeOauthCodeForMcpTokens).mockResolvedValueOnce({
      status: 400,
      data: { detail: "bad code" },
      headers: new Headers(),
    } as never);

    render(<McpConnectPanel onSuccess={() => {}} />);

    fireEvent.change(screen.getByLabelText(/server url/i), {
      target: { value: "https://mcp.example.com" },
    });
    fireEvent.click(screen.getByRole("button", { name: /connect/i }));

    await waitFor(() => {
      expect(screen.getByText(/bad code/i)).toBeDefined();
    });

    expect(screen.queryByPlaceholderText(manualTokenPlaceholder)).toBeNull();
  });

  it("submits a bearer token then calls onSuccess", async () => {
    const {
      postV2InitiateOauthLoginForAnMcpServer,
      postV2StoreABearerTokenForAnMcpServer,
    } = await import("@/app/api/__generated__/endpoints/mcp/mcp");

    vi.mocked(postV2InitiateOauthLoginForAnMcpServer).mockRejectedValueOnce(
      makeApiError(400, noOAuthDetail.detail),
    );

    const onSuccess = vi.fn();
    render(<McpConnectPanel onSuccess={onSuccess} />);

    fireEvent.change(screen.getByLabelText(/server url/i), {
      target: { value: "https://mcp.example.com" },
    });
    fireEvent.click(screen.getByRole("button", { name: /connect/i }));

    await waitFor(() => {
      expect(screen.getByPlaceholderText(manualTokenPlaceholder)).toBeDefined();
    });

    vi.mocked(postV2StoreABearerTokenForAnMcpServer).mockResolvedValueOnce({
      status: 200,
      data: { ok: true },
      headers: new Headers(),
    } as never);

    fireEvent.change(screen.getByPlaceholderText(manualTokenPlaceholder), {
      target: { value: "secret-bearer-token" },
    });
    fireEvent.click(screen.getByRole("button", { name: /save token/i }));

    await waitFor(() => {
      expect(onSuccess).toHaveBeenCalledTimes(1);
    });
    expect(postV2StoreABearerTokenForAnMcpServer).toHaveBeenCalledWith(
      {
        server_url: "https://mcp.example.com",
        token: "Bearer secret-bearer-token",
      },
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );
  });

  it("submits a selected Basic credential with an explicit prefix", async () => {
    const {
      postV2InitiateOauthLoginForAnMcpServer,
      postV2StoreABearerTokenForAnMcpServer,
    } = await import("@/app/api/__generated__/endpoints/mcp/mcp");

    vi.mocked(postV2InitiateOauthLoginForAnMcpServer).mockResolvedValueOnce({
      status: 400,
      data: noOAuthDetail,
      headers: new Headers(),
    } as never);
    vi.mocked(postV2StoreABearerTokenForAnMcpServer).mockResolvedValueOnce({
      status: 200,
      data: { ok: true },
      headers: new Headers(),
    } as never);

    render(<McpConnectPanel onSuccess={() => {}} />);
    fireEvent.change(screen.getByLabelText(/server url/i), {
      target: { value: "https://mcp.example.com" },
    });
    fireEvent.click(screen.getByRole("button", { name: /connect/i }));
    await waitFor(() => {
      expect(screen.getByLabelText("Authentication type")).toBeDefined();
    });

    fireEvent.change(screen.getByLabelText("Authentication type"), {
      target: { value: "basic" },
    });
    expect(screen.getByText("Basic authentication token")).toBeDefined();
    fireEvent.change(screen.getByPlaceholderText(manualTokenPlaceholder), {
      target: { value: "  cGstbGYtYWJjZA==  " },
    });
    fireEvent.click(screen.getByRole("button", { name: /save token/i }));

    await waitFor(() => {
      expect(postV2StoreABearerTokenForAnMcpServer).toHaveBeenCalledWith(
        {
          server_url: "https://mcp.example.com",
          token: "Basic cGstbGYtYWJjZA==",
        },
        expect.objectContaining({ signal: expect.any(AbortSignal) }),
      );
    });
  });

  it("restores the saved Basic scheme for the same server", async () => {
    mockSavedCredentials = [
      {
        provider: "mcp",
        host: "https://mcp.example.com",
        mcp_auth_scheme: "basic",
      },
    ];
    const {
      postV2InitiateOauthLoginForAnMcpServer,
      postV2StoreABearerTokenForAnMcpServer,
    } = await import("@/app/api/__generated__/endpoints/mcp/mcp");
    vi.mocked(postV2InitiateOauthLoginForAnMcpServer).mockResolvedValueOnce({
      status: 400,
      data: noOAuthDetail,
      headers: new Headers(),
    } as never);
    vi.mocked(postV2StoreABearerTokenForAnMcpServer).mockResolvedValueOnce({
      status: 200,
      data: { ok: true },
      headers: new Headers(),
    } as never);

    render(<McpConnectPanel onSuccess={() => {}} />);
    fireEvent.change(screen.getByLabelText(/server url/i), {
      target: { value: "https://mcp.example.com/" },
    });
    fireEvent.click(screen.getByRole("button", { name: /connect/i }));
    await waitFor(() => {
      expect(screen.getByLabelText("Authentication type")).toBeDefined();
    });
    expect(
      (screen.getByLabelText("Authentication type") as HTMLSelectElement).value,
    ).toBe("basic");

    fireEvent.change(screen.getByPlaceholderText(manualTokenPlaceholder), {
      target: { value: "new-encoded-value" },
    });
    fireEvent.click(screen.getByRole("button", { name: /save token/i }));

    await waitFor(() => {
      expect(postV2StoreABearerTokenForAnMcpServer).toHaveBeenCalledWith(
        {
          server_url: "https://mcp.example.com/",
          token: "Basic new-encoded-value",
        },
        expect.objectContaining({ signal: expect.any(AbortSignal) }),
      );
    });
  });

  it("hands the stored credential to onSuccess", async () => {
    // The payload, not just the fact of being called: the dialog binds what it
    // receives here, so returning undefined silently unbinds the connection.
    const {
      postV2InitiateOauthLoginForAnMcpServer,
      postV2StoreABearerTokenForAnMcpServer,
    } = await import("@/app/api/__generated__/endpoints/mcp/mcp");
    const stored = {
      id: "cred-1",
      provider: "mcp",
      type: "oauth2",
      title: "MCP: mcp.example.com",
      host: "https://mcp.example.com",
      mcp_auth_scheme: "basic",
    };
    vi.mocked(postV2InitiateOauthLoginForAnMcpServer).mockResolvedValueOnce({
      status: 400,
      data: noOAuthDetail,
      headers: new Headers(),
    } as never);
    vi.mocked(postV2StoreABearerTokenForAnMcpServer).mockResolvedValueOnce({
      status: 200,
      data: stored,
      headers: new Headers(),
    } as never);

    const onSuccess = vi.fn();
    render(<McpConnectPanel onSuccess={onSuccess} />);
    fireEvent.change(screen.getByLabelText(/server url/i), {
      target: { value: "https://mcp.example.com" },
    });
    fireEvent.click(screen.getByRole("button", { name: /connect/i }));
    await waitFor(() => {
      expect(screen.getByLabelText("Authentication type")).toBeDefined();
    });

    fireEvent.change(screen.getByPlaceholderText(manualTokenPlaceholder), {
      target: { value: "cGstbGYtYWJjZA==" },
    });
    fireEvent.click(screen.getByRole("button", { name: /save token/i }));

    await waitFor(() => {
      expect(onSuccess).toHaveBeenCalledWith(stored);
    });
  });

  it("rejects an unencoded user:password before any request", async () => {
    const {
      postV2DiscoverAvailableToolsOnAnMcpServer,
      postV2InitiateOauthLoginForAnMcpServer,
      postV2StoreABearerTokenForAnMcpServer,
    } = await import("@/app/api/__generated__/endpoints/mcp/mcp");
    vi.mocked(postV2InitiateOauthLoginForAnMcpServer).mockResolvedValueOnce({
      status: 400,
      data: noOAuthDetail,
      headers: new Headers(),
    } as never);

    render(<McpConnectPanel onSuccess={() => {}} />);
    fireEvent.change(screen.getByLabelText(/server url/i), {
      target: { value: "https://mcp.example.com" },
    });
    fireEvent.click(screen.getByRole("button", { name: /connect/i }));
    await waitFor(() => {
      expect(screen.getByLabelText("Authentication type")).toBeDefined();
    });

    fireEvent.change(screen.getByLabelText("Authentication type"), {
      target: { value: "basic" },
    });
    fireEvent.change(screen.getByPlaceholderText(manualTokenPlaceholder), {
      target: { value: "pk-lf-abc:sk-lf-xyz" },
    });
    fireEvent.click(screen.getByRole("button", { name: /save token/i }));

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toMatch(/unencoded user:password/i);
    expect(postV2DiscoverAvailableToolsOnAnMcpServer).not.toHaveBeenCalled();
    expect(postV2StoreABearerTokenForAnMcpServer).not.toHaveBeenCalled();
  });

  it("does not store a credential the server rejects", async () => {
    const {
      postV2DiscoverAvailableToolsOnAnMcpServer,
      postV2InitiateOauthLoginForAnMcpServer,
      postV2StoreABearerTokenForAnMcpServer,
    } = await import("@/app/api/__generated__/endpoints/mcp/mcp");

    vi.mocked(postV2InitiateOauthLoginForAnMcpServer).mockResolvedValueOnce({
      status: 400,
      data: noOAuthDetail,
      headers: new Headers(),
    } as never);
    vi.mocked(postV2DiscoverAvailableToolsOnAnMcpServer).mockResolvedValue({
      status: 401,
      data: { detail: "Bad credential" },
      headers: new Headers(),
    } as never);

    const onSuccess = vi.fn();
    render(<McpConnectPanel onSuccess={onSuccess} />);
    fireEvent.change(screen.getByLabelText(/server url/i), {
      target: { value: "https://mcp.example.com" },
    });
    fireEvent.click(screen.getByRole("button", { name: /connect/i }));
    await waitFor(() => {
      expect(screen.getByLabelText("Authentication type")).toBeDefined();
    });

    fireEvent.change(screen.getByPlaceholderText(manualTokenPlaceholder), {
      target: { value: "wrong-token" },
    });
    fireEvent.click(screen.getByRole("button", { name: /save token/i }));

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("Bad credential");
    expect(postV2StoreABearerTokenForAnMcpServer).not.toHaveBeenCalled();
    expect(onSuccess).not.toHaveBeenCalled();
  });

  it("rejects a Basic credential containing whitespace before any request", async () => {
    const {
      postV2DiscoverAvailableToolsOnAnMcpServer,
      postV2InitiateOauthLoginForAnMcpServer,
      postV2StoreABearerTokenForAnMcpServer,
    } = await import("@/app/api/__generated__/endpoints/mcp/mcp");

    vi.mocked(postV2InitiateOauthLoginForAnMcpServer).mockResolvedValueOnce({
      status: 400,
      data: noOAuthDetail,
      headers: new Headers(),
    } as never);

    render(<McpConnectPanel onSuccess={() => {}} />);
    fireEvent.change(screen.getByLabelText(/server url/i), {
      target: { value: "https://mcp.example.com" },
    });
    fireEvent.click(screen.getByRole("button", { name: /connect/i }));
    await waitFor(() => {
      expect(screen.getByLabelText("Authentication type")).toBeDefined();
    });

    fireEvent.change(screen.getByLabelText("Authentication type"), {
      target: { value: "basic" },
    });
    fireEvent.change(screen.getByPlaceholderText(manualTokenPlaceholder), {
      target: { value: "user pass" },
    });
    fireEvent.click(screen.getByRole("button", { name: /save token/i }));

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("cannot contain spaces");
    expect(postV2DiscoverAvailableToolsOnAnMcpServer).not.toHaveBeenCalled();
    expect(postV2StoreABearerTokenForAnMcpServer).not.toHaveBeenCalled();
  });

  it("lets the user switch from manual-token back to OAuth", async () => {
    const { postV2InitiateOauthLoginForAnMcpServer } = await import(
      "@/app/api/__generated__/endpoints/mcp/mcp"
    );

    vi.mocked(postV2InitiateOauthLoginForAnMcpServer).mockRejectedValueOnce(
      makeApiError(400, noOAuthDetail.detail),
    );

    render(<McpConnectPanel onSuccess={() => {}} />);

    fireEvent.change(screen.getByLabelText(/server url/i), {
      target: { value: "https://mcp.example.com" },
    });
    fireEvent.click(screen.getByRole("button", { name: /connect/i }));

    await waitFor(() => {
      expect(screen.getByPlaceholderText(manualTokenPlaceholder)).toBeDefined();
    });

    fireEvent.change(screen.getByPlaceholderText(manualTokenPlaceholder), {
      target: { value: "stale-token" },
    });

    fireEvent.click(screen.getByRole("button", { name: /try oauth/i }));

    expect(screen.queryByPlaceholderText(manualTokenPlaceholder)).toBeNull();
    expect(screen.getByRole("button", { name: /connect/i })).toBeDefined();
  });

  it("discards manual credentials when the server URL changes", async () => {
    const {
      postV2InitiateOauthLoginForAnMcpServer,
      postV2StoreABearerTokenForAnMcpServer,
    } = await import("@/app/api/__generated__/endpoints/mcp/mcp");

    vi.mocked(postV2InitiateOauthLoginForAnMcpServer)
      .mockResolvedValueOnce({
        status: 400,
        data: noOAuthDetail,
        headers: new Headers(),
      } as never)
      .mockResolvedValueOnce({
        status: 400,
        data: noOAuthDetail,
        headers: new Headers(),
      } as never);

    render(<McpConnectPanel onSuccess={() => {}} />);

    const urlInput = screen.getByLabelText(/server url/i);
    fireEvent.change(urlInput, {
      target: { value: "https://server-a.example.com" },
    });
    fireEvent.click(screen.getByRole("button", { name: /connect/i }));

    await waitFor(() => {
      expect(screen.getByLabelText("Authentication type")).toBeDefined();
    });
    fireEvent.change(screen.getByLabelText("Authentication type"), {
      target: { value: "basic" },
    });
    fireEvent.change(screen.getByPlaceholderText(manualTokenPlaceholder), {
      target: { value: "credential-for-server-a" },
    });

    fireEvent.change(urlInput, {
      target: { value: "https://server-b.example.com" },
    });

    expect(screen.queryByLabelText("Authentication type")).toBeNull();
    expect(screen.queryByPlaceholderText(manualTokenPlaceholder)).toBeNull();
    expect(screen.queryByRole("alert")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: /connect/i }));

    await waitFor(() => {
      expect(screen.getByLabelText("Authentication type")).toBeDefined();
    });
    expect(
      (screen.getByLabelText("Authentication type") as HTMLSelectElement).value,
    ).toBe("bearer");
    expect(
      (screen.getByPlaceholderText(manualTokenPlaceholder) as HTMLInputElement)
        .value,
    ).toBe("");
    expect(
      (
        screen.getByRole("button", {
          name: /save token/i,
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(true);
    expect(postV2StoreABearerTokenForAnMcpServer).not.toHaveBeenCalled();
  });

  it("surfaces an error when bearer-token submission fails", async () => {
    const {
      postV2InitiateOauthLoginForAnMcpServer,
      postV2StoreABearerTokenForAnMcpServer,
    } = await import("@/app/api/__generated__/endpoints/mcp/mcp");

    vi.mocked(postV2InitiateOauthLoginForAnMcpServer).mockRejectedValueOnce(
      makeApiError(400, noOAuthDetail.detail),
    );

    const onSuccess = vi.fn();
    render(<McpConnectPanel onSuccess={onSuccess} />);

    fireEvent.change(screen.getByLabelText(/server url/i), {
      target: { value: "https://mcp.example.com" },
    });
    fireEvent.click(screen.getByRole("button", { name: /connect/i }));

    await waitFor(() => {
      expect(screen.getByPlaceholderText(manualTokenPlaceholder)).toBeDefined();
    });

    vi.mocked(postV2StoreABearerTokenForAnMcpServer).mockResolvedValueOnce({
      status: 401,
      data: { detail: "invalid token" },
      headers: new Headers(),
    } as never);

    fireEvent.change(screen.getByPlaceholderText(manualTokenPlaceholder), {
      target: { value: "wrong-token" },
    });
    fireEvent.click(screen.getByRole("button", { name: /save token/i }));

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("invalid token");
    expect(onSuccess).not.toHaveBeenCalled();
  });

  it("renders an aria-live error region when a non-400 error occurs", async () => {
    const { postV2InitiateOauthLoginForAnMcpServer } = await import(
      "@/app/api/__generated__/endpoints/mcp/mcp"
    );

    vi.mocked(postV2InitiateOauthLoginForAnMcpServer).mockRejectedValueOnce(
      makeApiError(500, "internal server error"),
    );

    render(<McpConnectPanel onSuccess={() => {}} />);

    fireEvent.change(screen.getByLabelText(/server url/i), {
      target: { value: "https://mcp.example.com" },
    });
    fireEvent.click(screen.getByRole("button", { name: /connect/i }));

    const alert = await screen.findByRole("alert");
    expect(alert.getAttribute("aria-live")).toBe("polite");
    expect(alert.textContent).toContain("internal server error");
  });
});
