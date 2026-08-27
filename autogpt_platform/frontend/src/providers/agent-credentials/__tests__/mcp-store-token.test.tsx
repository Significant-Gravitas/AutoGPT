import { render, screen, waitFor } from "@testing-library/react";
import { useContext } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import CredentialsProvider, {
  CredentialsProvidersContext,
} from "../credentials-provider";

const mockStoreToken = vi.fn();
const mockExchangeCode = vi.fn();
vi.mock("@/app/api/__generated__/endpoints/mcp/mcp", () => ({
  postV2StoreABearerTokenForAnMcpServer: (...args: unknown[]) =>
    mockStoreToken(...args),
  postV2ExchangeOauthCodeForMcpTokens: (...args: unknown[]) =>
    mockExchangeCode(...args),
}));

vi.mock("@/app/api/__generated__/endpoints/integrations/integrations", () => ({
  getGetV1ListCredentialsQueryKey: () => ["credentials"],
}));

vi.mock("@tanstack/react-query", () => ({
  useQueryClient: () => ({
    getQueryCache: () => ({ subscribe: () => () => {} }),
  }),
  hashKey: () => "key",
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ isLoggedIn: true, isUserLoading: false }),
}));

const onFailToast = vi.fn(() => vi.fn());
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToastOnFail: () => onFailToast,
}));

const EXISTING = {
  id: "old-cred",
  provider: "mcp",
  type: "api_key",
  title: "MCP: datafa.st",
  host: "https://mcp.datafa.st/mcp",
};

vi.mock("@/lib/autogpt-server-api/context", () => ({
  useBackendAPI: () => ({
    listProviders: async () => ["mcp"],
    listSystemProviders: async () => [],
    listCredentials: async () => [EXISTING],
  }),
}));

/** Renders the saved MCP credential IDs and exposes `mcpStoreToken`. */
function Probe({
  onReady,
  pick = "mcpStoreToken",
}: {
  onReady: (fn: unknown) => void;
  pick?: "mcpStoreToken" | "mcpOAuthCallback";
}) {
  const providers = useContext(CredentialsProvidersContext);
  const mcp = providers?.["mcp"];
  if (mcp) onReady(mcp[pick]);
  return (
    <div data-testid="ids">
      {(mcp?.savedCredentials ?? []).map((c) => c.id).join(",")}
    </div>
  );
}

function setup(pick?: "mcpStoreToken" | "mcpOAuthCallback") {
  let fn: ((a: string, b: string) => Promise<unknown>) | null = null;
  render(
    <CredentialsProvider>
      <Probe
        pick={pick}
        onReady={(ready) => {
          fn = ready as typeof fn;
        }}
      />
    </CredentialsProvider>,
  );
  return () => fn;
}

afterEach(() => {
  vi.clearAllMocks();
});

describe("CredentialsProvider — mcpStoreToken", () => {
  it("replaces the server's cached credential with the newly stored one", async () => {
    // The backend deletes the previous credential for the server, so leaving
    // it in the cached list makes the picker re-select a deleted ID.
    mockStoreToken.mockResolvedValue({
      status: 200,
      data: {
        id: "new-cred",
        provider: "mcp",
        type: "api_key",
        title: "MCP: datafa.st",
        host: "https://mcp.datafa.st/mcp",
        scopes: null,
        username: null,
      },
    });

    const getStoreToken = setup();
    await waitFor(() =>
      expect(screen.getByTestId("ids").textContent).toBe("old-cred"),
    );

    const result = await getStoreToken()!(
      "https://mcp.datafa.st/mcp",
      "df_live_secret",
    );

    expect(mockStoreToken).toHaveBeenCalledWith({
      server_url: "https://mcp.datafa.st/mcp",
      token: "df_live_secret",
    });
    expect((result as { id: string }).id).toBe("new-cred");
    await waitFor(() =>
      expect(screen.getByTestId("ids").textContent).toBe("new-cred"),
    );
  });

  it("falls back to the requested server URL when the response has no host", async () => {
    mockStoreToken.mockResolvedValue({
      status: 200,
      data: {
        id: "new-cred",
        provider: "mcp",
        type: "api_key",
        title: null,
        host: null,
        scopes: null,
        username: null,
      },
    });

    const getStoreToken = setup();
    await waitFor(() => expect(getStoreToken()).not.toBeNull());

    await getStoreToken()!("https://mcp.datafa.st/mcp", "tok");

    await waitFor(() =>
      expect(screen.getByTestId("ids").textContent).toBe("new-cred"),
    );
  });

  it("surfaces a toast and rethrows when the request fails", async () => {
    mockStoreToken.mockResolvedValue({ status: 500, data: { detail: "boom" } });

    const getStoreToken = setup();
    await waitFor(() => expect(getStoreToken()).not.toBeNull());

    await expect(
      getStoreToken()!("https://mcp.datafa.st/mcp", "tok"),
    ).rejects.toBeDefined();
    expect(onFailToast).toHaveBeenCalledWith("save MCP API token");
    // The stale credential must survive a failed store.
    expect(screen.getByTestId("ids").textContent).toBe("old-cred");
  });
});

describe("CredentialsProvider — mcpOAuthCallback", () => {
  it("replaces the server's cached credential when the response carries a host", async () => {
    mockExchangeCode.mockResolvedValue({
      status: 200,
      data: {
        id: "oauth-cred",
        provider: "mcp",
        type: "oauth2",
        title: "MCP: datafa.st",
        host: "https://mcp.datafa.st/mcp",
        scopes: ["read"],
        username: "abhi",
      },
    });

    const getCallback = setup("mcpOAuthCallback");
    await waitFor(() =>
      expect(screen.getByTestId("ids").textContent).toBe("old-cred"),
    );

    await getCallback()!("code", "state");

    await waitFor(() =>
      expect(screen.getByTestId("ids").textContent).toBe("oauth-cred"),
    );
  });

  it("falls back to a plain upsert when the response has no host", async () => {
    mockExchangeCode.mockResolvedValue({
      status: 200,
      data: {
        id: "oauth-cred",
        provider: "mcp",
        type: "oauth2",
        title: null,
        host: null,
        scopes: null,
        username: null,
      },
    });

    const getCallback = setup("mcpOAuthCallback");
    await waitFor(() =>
      expect(screen.getByTestId("ids").textContent).toBe("old-cred"),
    );

    await getCallback()!("code", "state");

    // Nothing identifies the server, so the existing row must survive.
    await waitFor(() =>
      expect(screen.getByTestId("ids").textContent).toBe("old-cred,oauth-cred"),
    );
  });
});
