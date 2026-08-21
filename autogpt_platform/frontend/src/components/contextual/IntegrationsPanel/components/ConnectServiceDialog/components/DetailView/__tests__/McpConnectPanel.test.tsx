import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";

import { McpConnectPanel } from "../McpConnectPanel";

vi.mock("@/lib/oauth-popup", () => ({
  openOAuthPopup: vi.fn(),
}));

vi.mock("@/app/api/__generated__/endpoints/mcp/mcp", () => ({
  postV2InitiateOauthLoginForAnMcpServer: vi.fn(),
  postV2ExchangeOauthCodeForMcpTokens: vi.fn(),
  postV2StoreABearerTokenForAnMcpServer: vi.fn(),
}));

function makeApiError(status: number, detail = "boom"): Error {
  const err = new Error(detail) as Error & {
    status: number;
    response: unknown;
  };
  err.name = "ApiError";
  err.status = status;
  err.response = { detail };
  return err;
}

describe("McpConnectPanel", () => {
  afterEach(() => {
    vi.resetAllMocks();
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

  it("falls back to manual-token form when initiate returns 400", async () => {
    const { postV2InitiateOauthLoginForAnMcpServer } = await import(
      "@/app/api/__generated__/endpoints/mcp/mcp"
    );

    vi.mocked(postV2InitiateOauthLoginForAnMcpServer).mockResolvedValueOnce({
      status: 400,
      data: { detail: "OAuth not supported" },
      headers: new Headers(),
    } as never);

    render(<McpConnectPanel onSuccess={() => {}} />);

    fireEvent.change(screen.getByLabelText(/server url/i), {
      target: { value: "https://mcp.example.com" },
    });
    fireEvent.click(screen.getByRole("button", { name: /connect/i }));

    await waitFor(() => {
      expect(screen.getByPlaceholderText(/paste api token/i)).toBeDefined();
    });
    expect(
      screen.getByText(/server doesn't support oauth sign-in/i),
    ).toBeDefined();
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

    expect(screen.queryByPlaceholderText(/paste api token/i)).toBeNull();
  });

  it("submits a bearer token then calls onSuccess", async () => {
    const {
      postV2InitiateOauthLoginForAnMcpServer,
      postV2StoreABearerTokenForAnMcpServer,
    } = await import("@/app/api/__generated__/endpoints/mcp/mcp");

    vi.mocked(postV2InitiateOauthLoginForAnMcpServer).mockRejectedValueOnce(
      makeApiError(400, "OAuth not supported"),
    );

    const onSuccess = vi.fn();
    render(<McpConnectPanel onSuccess={onSuccess} />);

    fireEvent.change(screen.getByLabelText(/server url/i), {
      target: { value: "https://mcp.example.com" },
    });
    fireEvent.click(screen.getByRole("button", { name: /connect/i }));

    await waitFor(() => {
      expect(screen.getByPlaceholderText(/paste api token/i)).toBeDefined();
    });

    vi.mocked(postV2StoreABearerTokenForAnMcpServer).mockResolvedValueOnce({
      status: 200,
      data: { ok: true },
      headers: new Headers(),
    } as never);

    fireEvent.change(screen.getByPlaceholderText(/paste api token/i), {
      target: { value: "secret-bearer-token" },
    });
    fireEvent.click(screen.getByRole("button", { name: /save token/i }));

    await waitFor(() => {
      expect(onSuccess).toHaveBeenCalledTimes(1);
    });
    expect(postV2StoreABearerTokenForAnMcpServer).toHaveBeenCalledWith({
      server_url: "https://mcp.example.com",
      token: "secret-bearer-token",
    });
  });

  it("submits a selected Basic credential with an explicit prefix", async () => {
    const {
      postV2InitiateOauthLoginForAnMcpServer,
      postV2StoreABearerTokenForAnMcpServer,
    } = await import("@/app/api/__generated__/endpoints/mcp/mcp");

    vi.mocked(postV2InitiateOauthLoginForAnMcpServer).mockResolvedValueOnce({
      status: 400,
      data: { detail: "OAuth not supported" },
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
    fireEvent.change(screen.getByPlaceholderText(/paste api token/i), {
      target: { value: "  cGstbGYtYWJjZA==  " },
    });
    fireEvent.click(screen.getByRole("button", { name: /save token/i }));

    await waitFor(() => {
      expect(postV2StoreABearerTokenForAnMcpServer).toHaveBeenCalledWith({
        server_url: "https://mcp.example.com",
        token: "Basic cGstbGYtYWJjZA==",
      });
    });
  });

  it("lets the user switch from manual-token back to OAuth", async () => {
    const { postV2InitiateOauthLoginForAnMcpServer } = await import(
      "@/app/api/__generated__/endpoints/mcp/mcp"
    );

    vi.mocked(postV2InitiateOauthLoginForAnMcpServer).mockRejectedValueOnce(
      makeApiError(400, "OAuth not supported"),
    );

    render(<McpConnectPanel onSuccess={() => {}} />);

    fireEvent.change(screen.getByLabelText(/server url/i), {
      target: { value: "https://mcp.example.com" },
    });
    fireEvent.click(screen.getByRole("button", { name: /connect/i }));

    await waitFor(() => {
      expect(screen.getByPlaceholderText(/paste api token/i)).toBeDefined();
    });

    fireEvent.change(screen.getByPlaceholderText(/paste api token/i), {
      target: { value: "stale-token" },
    });

    fireEvent.click(screen.getByRole("button", { name: /try oauth/i }));

    expect(screen.queryByPlaceholderText(/paste api token/i)).toBeNull();
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
        data: { detail: "OAuth not supported" },
        headers: new Headers(),
      } as never)
      .mockResolvedValueOnce({
        status: 400,
        data: { detail: "OAuth not supported" },
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
    fireEvent.change(screen.getByPlaceholderText(/paste api token/i), {
      target: { value: "credential-for-server-a" },
    });

    fireEvent.change(urlInput, {
      target: { value: "https://server-b.example.com" },
    });

    expect(screen.queryByLabelText("Authentication type")).toBeNull();
    expect(screen.queryByPlaceholderText(/paste api token/i)).toBeNull();
    expect(screen.queryByRole("alert")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: /connect/i }));

    await waitFor(() => {
      expect(screen.getByLabelText("Authentication type")).toBeDefined();
    });
    expect(
      (screen.getByLabelText("Authentication type") as HTMLSelectElement).value,
    ).toBe("bearer");
    expect(
      (screen.getByPlaceholderText(/paste api token/i) as HTMLInputElement)
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
      makeApiError(400, "OAuth not supported"),
    );

    const onSuccess = vi.fn();
    render(<McpConnectPanel onSuccess={onSuccess} />);

    fireEvent.change(screen.getByLabelText(/server url/i), {
      target: { value: "https://mcp.example.com" },
    });
    fireEvent.click(screen.getByRole("button", { name: /connect/i }));

    await waitFor(() => {
      expect(screen.getByPlaceholderText(/paste api token/i)).toBeDefined();
    });

    vi.mocked(postV2StoreABearerTokenForAnMcpServer).mockResolvedValueOnce({
      status: 401,
      data: { detail: "invalid token" },
      headers: new Headers(),
    } as never);

    fireEvent.change(screen.getByPlaceholderText(/paste api token/i), {
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
