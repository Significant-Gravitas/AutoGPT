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
