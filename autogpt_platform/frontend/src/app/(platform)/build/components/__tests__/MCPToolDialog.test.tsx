import {
  render,
  screen,
  fireEvent,
  waitFor,
  cleanup,
} from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { MCPToolDialog } from "../MCPToolDialog";

vi.mock("@/lib/oauth-popup", () => ({
  openOAuthPopup: vi.fn(),
}));

const mockDiscover = vi.fn();
const mockStoreToken = vi.fn();
const mockOAuthLogin = vi.fn();
const mockExchangeCode = vi.fn();
vi.mock("@/app/api/__generated__/endpoints/mcp/mcp", () => ({
  postV2DiscoverAvailableToolsOnAnMcpServer: (...args: unknown[]) =>
    mockDiscover(...args),
  postV2InitiateOauthLoginForAnMcpServer: (...args: unknown[]) =>
    mockOAuthLogin(...args),
  postV2ExchangeOauthCodeForMcpTokens: (...args: unknown[]) =>
    mockExchangeCode(...args),
  postV2StoreABearerTokenForAnMcpServer: (...args: unknown[]) =>
    mockStoreToken(...args),
}));

const SERVER_URL = "https://mcp.datafa.st/mcp";

function discoverOk() {
  return {
    status: 200,
    data: {
      tools: [
        {
          name: "get_analytics",
          description: "Fetch analytics",
          input_schema: { type: "object", properties: {}, required: [] },
        },
      ],
      server_name: "datafa.st",
    },
  };
}

describe("MCPToolDialog — static API key / bearer token", () => {
  afterEach(() => {
    cleanup();
    mockDiscover.mockReset();
    mockStoreToken.mockReset();
    mockOAuthLogin.mockReset();
    mockExchangeCode.mockReset();
  });

  it("persists a manually entered token and attaches it as the block credential", async () => {
    mockDiscover.mockResolvedValue(discoverOk());
    mockStoreToken.mockResolvedValue({
      status: 200,
      data: {
        id: "cred-abc",
        provider: "mcp",
        type: "api_key",
        title: "MCP: datafa.st",
        host: SERVER_URL,
      },
    });
    const onConfirm = vi.fn();

    render(<MCPToolDialog open onClose={() => {}} onConfirm={onConfirm} />);

    fireEvent.change(screen.getByLabelText(/server url/i), {
      target: { value: SERVER_URL },
    });

    // Proactively choose the token flow without triggering a failed OAuth
    // round-trip first.
    fireEvent.click(
      screen.getByText(/use an api key \/ bearer token instead/i),
    );
    fireEvent.change(screen.getByLabelText(/api key \/ bearer token/i), {
      target: { value: "df_live_secret" },
    });
    fireEvent.click(
      screen.getByRole("button", { name: /connect with token/i }),
    );

    // The token is persisted via the /token endpoint so the block can
    // authenticate at runtime (discovery alone would discard it).
    await waitFor(() =>
      expect(mockStoreToken).toHaveBeenCalledWith({
        server_url: SERVER_URL,
        token: "df_live_secret",
      }),
    );

    // Land on the tool-selection step, pick a tool, add the block.
    const toolButton = await screen.findByText("get_analytics");
    fireEvent.click(toolButton);
    fireEvent.click(screen.getByRole("button", { name: /add block/i }));

    expect(onConfirm).toHaveBeenCalledTimes(1);
    const result = onConfirm.mock.calls[0][0];
    expect(result.credentials).toEqual({
      id: "cred-abc",
      provider: "mcp",
      type: "api_key",
      title: "MCP: datafa.st",
    });
    expect(result.selectedTool).toBe("get_analytics");
  });

  it("normalizes the server URL so it matches the one credentials are stored under", async () => {
    mockDiscover.mockResolvedValue(discoverOk());
    const onConfirm = vi.fn();

    render(<MCPToolDialog open onClose={() => {}} onConfirm={onConfirm} />);

    fireEvent.change(screen.getByLabelText(/server url/i), {
      target: { value: `  ${SERVER_URL}/  ` },
    });
    fireEvent.click(screen.getByRole("button", { name: /discover tools/i }));
    fireEvent.click(await screen.findByText("get_analytics"));
    fireEvent.click(screen.getByRole("button", { name: /add block/i }));

    expect(onConfirm.mock.calls[0][0].serverUrl).toBe(SERVER_URL);
  });

  it("does not attach credentials for a public server (no token)", async () => {
    mockDiscover.mockResolvedValue(discoverOk());
    const onConfirm = vi.fn();

    render(<MCPToolDialog open onClose={() => {}} onConfirm={onConfirm} />);

    fireEvent.change(screen.getByLabelText(/server url/i), {
      target: { value: SERVER_URL },
    });
    fireEvent.click(screen.getByRole("button", { name: /discover tools/i }));

    const toolButton = await screen.findByText("get_analytics");
    fireEvent.click(toolButton);
    fireEvent.click(screen.getByRole("button", { name: /add block/i }));

    expect(mockStoreToken).not.toHaveBeenCalled();
    expect(onConfirm.mock.calls[0][0].credentials).toBeNull();
  });

  it("shows an invalid-token error instead of bouncing into OAuth when a manual token is rejected", async () => {
    // The generated client throws an ApiError (carrying .status) on non-2xx.
    mockDiscover.mockRejectedValue({ status: 401, detail: "invalid token" });
    const onConfirm = vi.fn();

    render(<MCPToolDialog open onClose={() => {}} onConfirm={onConfirm} />);

    fireEvent.change(screen.getByLabelText(/server url/i), {
      target: { value: SERVER_URL },
    });
    fireEvent.click(
      screen.getByText(/use an api key \/ bearer token instead/i),
    );
    fireEvent.change(screen.getByLabelText(/api key \/ bearer token/i), {
      target: { value: "wrong-token" },
    });
    fireEvent.click(
      screen.getByRole("button", { name: /connect with token/i }),
    );

    expect(await screen.findByText(/authentication failed/i)).toBeDefined();
    // A rejected manual token must NOT trigger the OAuth login flow.
    expect(mockOAuthLogin).not.toHaveBeenCalled();
    expect(mockStoreToken).not.toHaveBeenCalled();
  });

  it("surfaces a clear error and adds no block when persisting the token fails", async () => {
    mockDiscover.mockResolvedValue(discoverOk());
    // Non-2xx from /token: the dialog must show a token-specific message, not
    // a misleading "failed to connect" one.
    mockStoreToken.mockResolvedValue({ status: 500, data: {} });
    const onConfirm = vi.fn();

    render(<MCPToolDialog open onClose={() => {}} onConfirm={onConfirm} />);

    fireEvent.change(screen.getByLabelText(/server url/i), {
      target: { value: SERVER_URL },
    });
    fireEvent.click(
      screen.getByText(/use an api key \/ bearer token instead/i),
    );
    fireEvent.change(screen.getByLabelText(/api key \/ bearer token/i), {
      target: { value: "df_live_secret" },
    });
    fireEvent.click(
      screen.getByRole("button", { name: /connect with token/i }),
    );

    // Discovery succeeded but persistence failed → error shown, no tool step,
    // no block added.
    expect(
      await screen.findByText(/saving your api token failed/i),
    ).toBeDefined();
    expect(screen.queryByText("get_analytics")).toBeNull();
    expect(onConfirm).not.toHaveBeenCalled();
  });

  it("can switch back from token entry to OAuth sign-in", () => {
    render(<MCPToolDialog open onClose={() => {}} onConfirm={vi.fn()} />);

    fireEvent.click(
      screen.getByText(/use an api key \/ bearer token instead/i),
    );
    expect(screen.getByLabelText(/api key \/ bearer token/i)).toBeDefined();

    fireEvent.click(screen.getByText(/use oauth sign-in instead/i));
    expect(screen.queryByLabelText(/api key \/ bearer token/i)).toBeNull();
    expect(
      screen.getByRole("button", { name: /discover tools/i }),
    ).toBeDefined();
  });
});

describe("MCPToolDialog — tool cards", () => {
  afterEach(() => {
    cleanup();
    mockDiscover.mockReset();
  });

  function discoverWithParams() {
    return {
      status: 200,
      data: {
        tools: [
          {
            name: "ask_question",
            description: "Ask a question about the data",
            input_schema: {
              type: "object",
              properties: { question: { type: "string" } },
              required: ["question"],
            },
          },
        ],
        server_name: "datafa.st",
      },
    };
  }

  async function discoverAndGetToggle() {
    mockDiscover.mockResolvedValue(discoverWithParams());
    render(<MCPToolDialog open onClose={() => {}} onConfirm={vi.fn()} />);

    fireEvent.change(screen.getByLabelText(/server url/i), {
      target: { value: SERVER_URL },
    });
    fireEvent.click(screen.getByRole("button", { name: /discover tools/i }));

    // Exact name: the card's own accessible name also *contains* "Show
    // details" because it is computed from its descendants.
    return await screen.findByRole("button", { name: "Show details" });
  }

  it("renders the details toggle outside of any button ancestor", async () => {
    const toggle = await discoverAndGetToggle();

    // A <button> nested inside a <button> is invalid HTML and breaks
    // hydration, so the card itself must not be a native button.
    expect(toggle.parentElement?.closest("button")).toBeNull();
  });

  it("expands parameter details without selecting the tool", async () => {
    const toggle = await discoverAndGetToggle();

    fireEvent.click(toggle);

    expect(await screen.findByText("Parameter")).toBeDefined();
    expect(screen.getByRole("button", { name: /add block/i })).toHaveProperty(
      "disabled",
      true,
    );
  });
});
