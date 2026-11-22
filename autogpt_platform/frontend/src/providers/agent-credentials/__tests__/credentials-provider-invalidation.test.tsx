import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import { useContext, type ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { getGetV2ListChatConnectionsQueryKey } from "@/app/api/__generated__/endpoints/chat/chat";
import { getGetV1ListCredentialsQueryKey } from "@/app/api/__generated__/endpoints/integrations/integrations";

// Settings → Integrations and the chat connection lists read the user's
// credentials through React Query. Every way the credentials field adds or
// removes one goes through this provider, so each has to mark those lists
// stale (invalidateConnectionQueries), or they show the old set for a minute.

const savedCredential = {
  id: "cred-new",
  provider: "github",
  type: "api_key",
  title: "New key",
};

const { apiMock, mcpMock, onFailToastMock } = vi.hoisted(() => ({
  // Stable, as the real hook's is: the provider's callbacks depend on it.
  onFailToastMock: vi.fn(() => () => {}),
  apiMock: {
    listCredentials: vi.fn(),
    listProviders: vi.fn(),
    listSystemProviders: vi.fn(),
    oAuthCallback: vi.fn(),
    createAPIKeyCredentials: vi.fn(),
    createUserPasswordCredentials: vi.fn(),
    createHostScopedCredentials: vi.fn(),
    deleteCredentials: vi.fn(),
  },
  mcpMock: {
    postV2ExchangeOauthCodeForMcpTokens: vi.fn(),
    postV2StoreABearerTokenForAnMcpServer: vi.fn(),
  },
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ isLoggedIn: true, isUserLoading: false }),
}));

vi.mock("@/lib/autogpt-server-api/context", () => ({
  useBackendAPI: () => apiMock,
}));

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToastOnFail: () => onFailToastMock,
}));

vi.mock("@/app/api/__generated__/endpoints/mcp/mcp", () => mcpMock);

import CredentialsProvider, {
  CredentialsProvidersContext,
  type CredentialsProviderData,
} from "../credentials-provider";

type Change = (provider: CredentialsProviderData) => Promise<unknown>;

const changes: [string, Change][] = [
  [
    "adding an API key",
    (p) => p.createAPIKeyCredentials({ api_key: "k", title: "New key" }),
  ],
  [
    "adding a username and password",
    (p) =>
      p.createUserPasswordCredentials({
        username: "u",
        password: "p",
        title: "New login",
      }),
  ],
  [
    "adding host-scoped headers",
    (p) =>
      p.createHostScopedCredentials({
        host: "api.example.com",
        headers: { Authorization: "Bearer x" },
        title: "New headers",
      }),
  ],
  ["finishing an OAuth sign-in", (p) => p.oAuthCallback("code", "state")],
  [
    "finishing an MCP sign-in",
    (p) => p.mcpOAuthCallback("code", "state", undefined),
  ],
  [
    "saving an MCP token",
    (p) => p.mcpStoreToken("https://mcp.example.com", "token"),
  ],
  ["deleting a key", (p) => p.deleteCredentials("cred-old")],
];

let current: CredentialsProviderData | undefined;

function Probe() {
  const providers = useContext(CredentialsProvidersContext);
  current = providers?.github;
  return <span data-testid="loaded">{current ? "yes" : "no"}</span>;
}

function renderProvider(queryClient: QueryClient) {
  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );
  }
  render(
    <CredentialsProvider>
      <Probe />
    </CredentialsProvider>,
    { wrapper: Wrapper },
  );
}

describe("CredentialsProvider keeps the connection lists fresh", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    current = undefined;
    apiMock.listCredentials.mockResolvedValue([]);
    apiMock.listProviders.mockResolvedValue(["github"]);
    apiMock.listSystemProviders.mockResolvedValue([]);
    apiMock.oAuthCallback.mockResolvedValue({
      ...savedCredential,
      type: "oauth2",
    });
    apiMock.createAPIKeyCredentials.mockResolvedValue(savedCredential);
    apiMock.createUserPasswordCredentials.mockResolvedValue({
      ...savedCredential,
      type: "user_password",
    });
    apiMock.createHostScopedCredentials.mockResolvedValue({
      ...savedCredential,
      type: "host_scoped",
    });
    apiMock.deleteCredentials.mockResolvedValue({
      deleted: true,
      revoked: null,
    });
    const mcpCredential = {
      status: 200,
      data: { ...savedCredential, provider: "mcp", type: "oauth2" },
    };
    mcpMock.postV2ExchangeOauthCodeForMcpTokens.mockResolvedValue(
      mcpCredential,
    );
    mcpMock.postV2StoreABearerTokenForAnMcpServer.mockResolvedValue(
      mcpCredential,
    );
  });

  it.each(changes)("%s marks them stale", async (_, change) => {
    const queryClient = new QueryClient({
      defaultOptions: { queries: { staleTime: 60_000, retry: false } },
    });
    const lists = [
      getGetV1ListCredentialsQueryKey(),
      getGetV2ListChatConnectionsQueryKey(),
    ];
    for (const key of lists) queryClient.setQueryData(key, { data: [] });
    renderProvider(queryClient);
    await waitFor(() =>
      expect(screen.getByTestId("loaded").textContent).toBe("yes"),
    );
    for (const key of lists) {
      expect(queryClient.getQueryState(key)?.isInvalidated).toBe(false);
    }

    await act(async () => {
      await change(current!);
    });

    for (const key of lists) {
      expect(
        queryClient.getQueryState(key)?.isInvalidated,
        `${JSON.stringify(key)} should be stale`,
      ).toBe(true);
    }
  });

  it("leaves them alone when a delete still needs confirmation", async () => {
    apiMock.deleteCredentials.mockResolvedValue({
      deleted: false,
      need_confirmation: true,
      message: "In use by a webhook",
    });
    const queryClient = new QueryClient({
      defaultOptions: { queries: { staleTime: 60_000, retry: false } },
    });
    const lists = [
      getGetV1ListCredentialsQueryKey(),
      getGetV2ListChatConnectionsQueryKey(),
    ];
    for (const key of lists) queryClient.setQueryData(key, { data: [] });
    renderProvider(queryClient);
    await waitFor(() =>
      expect(screen.getByTestId("loaded").textContent).toBe("yes"),
    );

    await act(async () => {
      await current!.deleteCredentials("cred-old");
    });

    for (const key of lists) {
      expect(queryClient.getQueryState(key)?.isInvalidated).toBe(false);
    }
  });

  it("leaves them alone when adding a key fails", async () => {
    apiMock.createAPIKeyCredentials.mockRejectedValue(new Error("refused"));
    const queryClient = new QueryClient({
      defaultOptions: { queries: { staleTime: 60_000, retry: false } },
    });
    const lists = [
      getGetV1ListCredentialsQueryKey(),
      getGetV2ListChatConnectionsQueryKey(),
    ];
    for (const key of lists) queryClient.setQueryData(key, { data: [] });
    renderProvider(queryClient);
    await waitFor(() =>
      expect(screen.getByTestId("loaded").textContent).toBe("yes"),
    );

    await act(async () => {
      await expect(
        current!.createAPIKeyCredentials({ api_key: "k", title: "New key" }),
      ).rejects.toThrow("refused");
    });

    for (const key of lists) {
      expect(queryClient.getQueryState(key)?.isInvalidated).toBe(false);
    }
  });
});
