import { getGetV1ListCredentialsQueryKey } from "@/app/api/__generated__/endpoints/integrations/integrations";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import { useContext, type ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const { apiMock, onFailToastMock } = vi.hoisted(() => {
  const errorHandler = vi.fn();
  return {
    apiMock: {
      listCredentials: vi.fn(),
      listProviders: vi.fn(),
      listSystemProviders: vi.fn(),
    },
    onFailToastMock: vi.fn(() => errorHandler),
  };
});

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ isLoggedIn: true, isUserLoading: false }),
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

// A fresh object every call, as each list response is.
function githubCredential(title = "GitHub") {
  return { id: "cred-1", provider: "github", type: "api_key", title };
}

const providerValues: unknown[] = [];
function ProvidersProbe() {
  const providers = useContext(CredentialsProvidersContext);
  providerValues.push(providers);
  const saved = providers?.github?.savedCredentials ?? [];
  return (
    <span data-testid="titles">{saved.map((c) => c.title).join(",")}</span>
  );
}

const actionValues: unknown[] = [];
function ActionsProbe() {
  actionValues.push(useContext(CredentialsActionsContext));
  return null;
}

const probes = (
  <>
    <ProvidersProbe />
    <ActionsProbe />
  </>
);

async function mountLoaded() {
  const view = render(<CredentialsProvider>{probes}</CredentialsProvider>, {
    wrapper: TestWrapper,
  });
  await waitFor(() =>
    expect(screen.getByTestId("titles").textContent).toBe("GitHub"),
  );
  return view;
}

async function invalidateCredentialsList() {
  const calls = apiMock.listCredentials.mock.calls.length;
  await act(async () => {
    await queryClient.invalidateQueries({
      queryKey: getGetV1ListCredentialsQueryKey(),
    });
  });
  await waitFor(() =>
    expect(apiMock.listCredentials.mock.calls.length).toBeGreaterThan(calls),
  );
  await act(async () => undefined);
}

describe("CredentialsProvider value identity", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    queryClient.clear();
    providerValues.length = 0;
    actionValues.length = 0;
    apiMock.listProviders.mockResolvedValue(["github", "google"]);
    apiMock.listSystemProviders.mockResolvedValue([]);
    apiMock.listCredentials.mockImplementation(() =>
      Promise.resolve([githubCredential()]),
    );
    // The provider only reacts to an existing query's cache events.
    queryClient.setQueryData(getGetV1ListCredentialsQueryKey(), []);
  });

  it("keeps the providers map when a refetch returns the same credentials", async () => {
    await mountLoaded();
    const loaded = providerValues[providerValues.length - 1];
    const renders = providerValues.length;

    await invalidateCredentialsList();

    expect(providerValues.length).toBe(renders);
    expect(providerValues[providerValues.length - 1]).toBe(loaded);
  });

  it("publishes a refetch that changed a credential, keeping the untouched entries", async () => {
    await mountLoaded();
    const loaded = providerValues[providerValues.length - 1] as Record<
      string,
      unknown
    >;

    apiMock.listCredentials.mockImplementation(() =>
      Promise.resolve([githubCredential("Renamed")]),
    );
    await invalidateCredentialsList();

    expect(screen.getByTestId("titles").textContent).toBe("Renamed");
    const updated = providerValues[providerValues.length - 1] as Record<
      string,
      unknown
    >;
    expect(updated).not.toBe(loaded);
    expect(updated.github).not.toBe(loaded.github);
    expect(updated.google).toBe(loaded.google);
  });

  it("keeps the actions value across a provider re-render", async () => {
    const view = await mountLoaded();
    const actions = actionValues[actionValues.length - 1];
    const renders = actionValues.length;

    view.rerender(<CredentialsProvider>{probes}</CredentialsProvider>);

    expect(actionValues.length).toBe(renders);
    expect(actionValues[actionValues.length - 1]).toBe(actions);
  });
});
