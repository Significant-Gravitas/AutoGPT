import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import { useContext, type ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const { apiMock, errorHandlerMock, isLoggedIn, onFailToastMock, useAuthMock } =
  vi.hoisted(() => {
    const errorHandler = vi.fn();
    return {
      apiMock: {
        listCredentials: vi.fn().mockResolvedValue([]),
        deleteCredentials: vi.fn().mockResolvedValue({ deleted: true }),
        listProviders: vi.fn().mockResolvedValue(["google"]),
        listSystemProviders: vi.fn().mockResolvedValue([]),
      },
      errorHandlerMock: errorHandler,
      isLoggedIn: { value: false },
      onFailToastMock: vi.fn(() => errorHandler),
      useAuthMock: vi.fn(),
    };
  });

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: useAuthMock,
}));

vi.mock("@/lib/autogpt-server-api/context", () => ({
  useBackendAPI: () => apiMock,
}));

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToastOnFail: () => onFailToastMock,
}));

import CredentialsProvider, {
  CredentialsActionsContext,
  CredentialsProvidersContext,
} from "../credentials-provider";

const queryClient = new QueryClient();

interface Props {
  children: ReactNode;
}

function TestWrapper({ children }: Props) {
  return (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

function ProviderState() {
  const providers = useContext(CredentialsProvidersContext);
  return (
    <span data-testid="provider-state">
      {providers === null ? "loading" : Object.keys(providers).join(",")}
    </span>
  );
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise;
  });
  return { promise, resolve };
}

/**
 * Exposes `upsert` and the resulting saved-credential ids, so a test can drive
 * the same sequence device auth does: mint a credential outside a list
 * response, then let a reload land.
 */
function UpsertProbe({ provider }: { provider: string }) {
  const providers = useContext(CredentialsProvidersContext);
  const actions = useContext(CredentialsActionsContext);
  const saved = providers?.[provider]?.savedCredentials ?? [];
  return (
    <div>
      <button
        data-testid="do-upsert"
        onClick={() =>
          actions?.upsert(provider, {
            id: "cred-device",
            provider,
            type: "oauth2",
            title: "Device Auth Credential",
          } as never)
        }
      >
        upsert
      </button>
      <button data-testid="do-reload" onClick={() => actions?.reload()}>
        reload
      </button>
      <button
        data-testid="do-delete"
        onClick={() =>
          providers?.[provider]?.deleteCredentials("cred-device", true)
        }
      >
        delete
      </button>
      <span data-testid="saved-ids">{saved.map((c) => c.id).join(",")}</span>
    </div>
  );
}

describe("CredentialsProvider authentication", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    queryClient.clear();
    isLoggedIn.value = false;
    apiMock.listCredentials.mockResolvedValue([]);
    apiMock.listProviders.mockResolvedValue(["google"]);
    apiMock.listSystemProviders.mockResolvedValue([]);
    useAuthMock.mockImplementation(() => ({
      isLoggedIn: isLoggedIn.value,
    }));
  });

  it("waits for authentication before loading protected provider metadata", async () => {
    const view = render(
      <CredentialsProvider>
        <span>app content</span>
      </CredentialsProvider>,
      { wrapper: TestWrapper },
    );

    expect(screen.getByText("app content")).toBeDefined();
    await act(async () => undefined);
    expect(apiMock.listProviders).not.toHaveBeenCalled();
    expect(apiMock.listSystemProviders).not.toHaveBeenCalled();
    expect(errorHandlerMock).not.toHaveBeenCalled();

    isLoggedIn.value = true;
    view.rerender(
      <CredentialsProvider>
        <span>app content</span>
      </CredentialsProvider>,
    );

    await waitFor(() => {
      expect(apiMock.listProviders).toHaveBeenCalledOnce();
      expect(apiMock.listSystemProviders).toHaveBeenCalledOnce();
      expect(apiMock.listCredentials).toHaveBeenCalledOnce();
    });
  });

  it("discards provider metadata that resolves after logout", async () => {
    const providersRequest = deferred<string[]>();
    const systemProvidersRequest = deferred<string[]>();
    apiMock.listProviders.mockReturnValueOnce(providersRequest.promise);
    apiMock.listSystemProviders.mockReturnValueOnce(
      systemProvidersRequest.promise,
    );
    isLoggedIn.value = true;

    const view = render(
      <CredentialsProvider>
        <ProviderState />
      </CredentialsProvider>,
      { wrapper: TestWrapper },
    );

    await waitFor(() => {
      expect(apiMock.listProviders).toHaveBeenCalledOnce();
      expect(apiMock.listSystemProviders).toHaveBeenCalledOnce();
    });

    isLoggedIn.value = false;
    view.rerender(
      <CredentialsProvider>
        <ProviderState />
      </CredentialsProvider>,
    );

    await act(async () => {
      providersRequest.resolve(["google"]);
      systemProvidersRequest.resolve(["google"]);
      await Promise.all([
        providersRequest.promise,
        systemProvidersRequest.promise,
      ]);
    });

    expect(screen.getByTestId("provider-state").textContent).toBe("");
    expect(apiMock.listCredentials).not.toHaveBeenCalled();
  });
});

describe("CredentialsProvider device-auth upserts", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    queryClient.clear();
    isLoggedIn.value = true;
    apiMock.listCredentials.mockResolvedValue([]);
    apiMock.listProviders.mockResolvedValue(["stripe_link"]);
    apiMock.listSystemProviders.mockResolvedValue([]);
    useAuthMock.mockImplementation(() => ({ isLoggedIn: isLoggedIn.value }));
  });

  const deviceCred = {
    id: "cred-device",
    provider: "stripe_link",
    type: "oauth2",
    title: "Device Auth Credential",
  };

  function savedIds() {
    return (screen.getByTestId("saved-ids").textContent ?? "")
      .split(",")
      .filter(Boolean);
  }

  async function mountProbe() {
    render(
      <CredentialsProvider>
        <UpsertProbe provider="stripe_link" />
      </CredentialsProvider>,
      { wrapper: TestWrapper },
    );
    await waitFor(() => expect(apiMock.listCredentials).toHaveBeenCalled());
  }

  /** Click reload and wait for that specific request to have been issued. */
  async function reloadAndSettle() {
    const before = apiMock.listCredentials.mock.calls.length;
    await act(async () => {
      screen.getByTestId("do-reload").click();
    });
    await waitFor(() =>
      expect(apiMock.listCredentials.mock.calls.length).toBeGreaterThan(before),
    );
    // Let the response publish.
    await act(async () => undefined);
  }

  it("survives a reload whose response predates the credential", async () => {
    // The device-auth race: the credential is minted server-side by the poll,
    // and a listCredentials() call that started before it existed resolves
    // afterwards. Publishing that stale response unmodified drops the
    // credential, and useCredentialsInput then tells the user their brand-new
    // connection was removed.
    await mountProbe();

    await act(async () => {
      screen.getByTestId("do-upsert").click();
    });
    expect(savedIds()).toContain("cred-device");

    apiMock.listCredentials.mockResolvedValue([] as never);
    await reloadAndSettle();

    expect(savedIds()).toContain("cred-device");
  });

  it("retires the pending entry once the server returns it", async () => {
    await mountProbe();
    await act(async () => {
      screen.getByTestId("do-upsert").click();
    });

    // Server catches up: present exactly once, not duplicated by the pending copy.
    apiMock.listCredentials.mockResolvedValue([deviceCred] as never);
    await reloadAndSettle();
    expect(savedIds().filter((i) => i === "cred-device")).toHaveLength(1);

    // The proof that it was actually retired rather than merely deduped: a
    // later empty reload must now drop it, because nothing is pending.
    apiMock.listCredentials.mockResolvedValue([] as never);
    await reloadAndSettle();
    expect(savedIds()).not.toContain("cred-device");
  });

  it("does not carry a pending credential across a logout", async () => {
    // pendingUpsertsRef outlives a session unless cleared. Left in place it
    // would be merged into the next user's list on the same mounted provider,
    // exposing the previous account's credential metadata.
    await mountProbe();
    await act(async () => {
      screen.getByTestId("do-upsert").click();
    });
    expect(savedIds()).toContain("cred-device");

    isLoggedIn.value = false;
    await act(async () => {
      screen.getByTestId("do-reload").click();
    });

    isLoggedIn.value = true;
    apiMock.listCredentials.mockResolvedValue([] as never);
    await reloadAndSettle();

    expect(savedIds()).not.toContain("cred-device");
  });

  it("does not resurrect a credential deleted while still pending", async () => {
    await mountProbe();
    await act(async () => {
      screen.getByTestId("do-upsert").click();
    });

    apiMock.deleteCredentials.mockResolvedValue({ deleted: true });
    await act(async () => {
      screen.getByTestId("do-delete").click();
    });

    apiMock.listCredentials.mockResolvedValue([] as never);
    await reloadAndSettle();

    expect(savedIds()).not.toContain("cred-device");
  });
});
