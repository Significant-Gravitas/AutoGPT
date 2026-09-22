import { getGetV1ListCredentialsQueryKey } from "@/app/api/__generated__/endpoints/integrations/integrations";
import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import { server } from "@/mocks/mock-server";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { type ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useConnectedIntegrations } from "../useConnectedIntegrations";

const GOOGLE_CREDENTIAL: CredentialsMetaResponse = {
  id: "google-cred",
  provider: "google",
  type: "oauth2",
  title: "google account",
  scopes: null,
  username: null,
};

const GOOGLE_MENTION = {
  provider: "google",
  name: "Google",
  token: "@Google",
};

const listCredentials = vi.fn();

function credentialsHandler() {
  return http.get("*/api/integrations/credentials", () => {
    listCredentials();
    return HttpResponse.json([GOOGLE_CREDENTIAL]);
  });
}

function createClient() {
  return new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
}

function seedCredentials(client: QueryClient) {
  client.setQueryData(getGetV1ListCredentialsQueryKey(), {
    status: 200,
    data: [GOOGLE_CREDENTIAL],
    headers: new Headers(),
  });
}

function wrapperFor(client: QueryClient) {
  return function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    );
  };
}

afterEach(() => {
  vi.clearAllMocks();
  server.resetHandlers();
});

describe("useConnectedIntegrations", () => {
  it("offers no integrations while disabled even when the credentials query is already cached", () => {
    server.use(credentialsHandler());
    const client = createClient();
    seedCredentials(client);

    const { result } = renderHook(() => useConnectedIntegrations(false), {
      wrapper: wrapperFor(client),
    });

    expect(result.current).toEqual([]);
    expect(listCredentials).not.toHaveBeenCalled();
  });

  it("lists connected providers once enabled and drops them again when disabled", async () => {
    server.use(credentialsHandler());
    const client = createClient();

    const { result, rerender } = renderHook(
      ({ enabled }: { enabled: boolean }) => useConnectedIntegrations(enabled),
      { wrapper: wrapperFor(client), initialProps: { enabled: true } },
    );

    await waitFor(() => expect(result.current).toEqual([GOOGLE_MENTION]));
    expect(listCredentials).toHaveBeenCalledTimes(1);

    rerender({ enabled: false });

    expect(result.current).toEqual([]);
    expect(
      client.getQueryData(getGetV1ListCredentialsQueryKey()),
    ).toBeDefined();
  });
});
