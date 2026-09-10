import type { CredentialsProviderName } from "@/lib/autogpt-server-api";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import * as React from "react";
import { useContext, type ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import CredentialsProvider, {
  CredentialsProvidersContext,
} from "../credentials-provider";

const storedCredential = {
  id: "cred-stored",
  provider: "mcp",
  type: "oauth2",
  title: "MCP: mcp.example.com",
  scopes: null,
  username: null,
  host: "https://mcp.example.com/mcp",
  mcp_auth_scheme: "basic",
};

const mocks = vi.hoisted(() => ({
  storeToken: vi.fn(),
  listCredentials: vi.fn(),
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ isLoggedIn: true, isUserLoading: false }),
}));

vi.mock("@/app/api/__generated__/endpoints/mcp/mcp", () => ({
  postV2StoreABearerTokenForAnMcpServer: mocks.storeToken,
  postV2ExchangeOauthCodeForMcpTokens: vi.fn(),
}));

vi.mock("@/lib/autogpt-server-api/context", async (importOriginal) => ({
  ...(await importOriginal<object>()),
  useBackendAPI: () => ({
    listCredentials: mocks.listCredentials,
    listProviders: async () => ["mcp"],
    listSystemProviders: async () => [],
  }),
}));

// Deliberately not `tests/integrations/test-utils`: its wrapper mounts
// `OnboardingProvider`, which reaches for `BackendAPI` methods this file stubs
// out. `CredentialsProvider` itself only needs a query client.
function Wrapper({ children }: { children: ReactNode }) {
  const [queryClient] = React.useState(
    () =>
      new QueryClient({
        defaultOptions: { queries: { retry: false } },
      }),
  );
  return (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

function McpConsumer() {
  const providers = useContext(CredentialsProvidersContext);
  const mcp = providers?.["mcp" as CredentialsProviderName];

  if (!mcp) return <span>loading</span>;
  return (
    <div>
      <button
        onClick={() =>
          mcp
            .mcpStoreToken("https://mcp.example.com/mcp", "Basic dXNlcjpwdw==")
            // The refusal case rejects; the assertion is on what the provider
            // map does *not* gain, so swallow it rather than let it surface as
            // an unhandled rejection.
            .catch(() => undefined)
        }
      >
        store credential
      </button>
      <span data-testid="saved-ids">
        {mcp.savedCredentials.map((cred) => cred.id).join(",")}
      </span>
    </div>
  );
}

describe("CredentialsProvider.mcpStoreToken", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.listCredentials.mockResolvedValue([]);
    mocks.storeToken.mockResolvedValue({ status: 200, data: storedCredential });
  });

  // The half this covers that `upsertProviderCredentials` cannot: that
  // `mcpStoreToken` actually feeds the endpoint's result back into the provider
  // map. Binding a node to a credential the map has never seen renders it as
  // "MCP: … was removed" until a full reload, which is the bug this exists to
  // fix — and deleting the `upsertCredentials` call left the suite green.
  it("publishes the stored credential to the provider map", async () => {
    render(
      <CredentialsProvider>
        <McpConsumer />
      </CredentialsProvider>,
      { wrapper: Wrapper },
    );

    await waitFor(() =>
      expect(screen.getByTestId("saved-ids").textContent).toBe(""),
    );

    fireEvent.click(screen.getByRole("button", { name: "store credential" }));

    await waitFor(() =>
      expect(screen.getByTestId("saved-ids").textContent).toBe("cred-stored"),
    );
    expect(mocks.storeToken).toHaveBeenCalledWith({
      server_url: "https://mcp.example.com/mcp",
      token: "Basic dXNlcjpwdw==",
    });
  });

  it("does not publish anything when the endpoint refuses the credential", async () => {
    mocks.storeToken.mockResolvedValue({
      status: 422,
      data: {
        detail: "Basic authentication expects the Base64 of user:password",
      },
    });

    render(
      <CredentialsProvider>
        <McpConsumer />
      </CredentialsProvider>,
      { wrapper: Wrapper },
    );

    await waitFor(() =>
      expect(screen.getByTestId("saved-ids").textContent).toBe(""),
    );

    fireEvent.click(screen.getByRole("button", { name: "store credential" }));

    await waitFor(() => expect(mocks.storeToken).toHaveBeenCalled());
    expect(screen.getByTestId("saved-ids").textContent).toBe("");
  });
});
