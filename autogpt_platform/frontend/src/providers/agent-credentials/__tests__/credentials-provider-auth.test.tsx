import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
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

import CredentialsProvider from "../credentials-provider";

const queryClient = new QueryClient();

function TestWrapper({ children }: { children: ReactNode }) {
  return (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

describe("CredentialsProvider authentication", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    queryClient.clear();
    isLoggedIn.value = false;
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
});
