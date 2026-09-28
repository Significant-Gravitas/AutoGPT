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
  it("offers the owner's accounts in Otto", async () => {
    server.use(credentialsHandler());
    const { result } = renderHook(() => useConnectedIntegrations(), {
      wrapper: wrapperFor(createClient()),
    });
    await waitFor(() =>
      expect(result.current.map((account) => account.credentialId)).toEqual([
        "google-cred",
      ]),
    );
  });

  it("only offers granted credentials for an expert even with owner credentials cached", async () => {
    server.use(
      credentialsHandler(),
      http.get("*/api/experts/:expertId/credentials", ({ params }) =>
        HttpResponse.json(
          params.expertId === "expert-a"
            ? [
                {
                  credential_id: "work",
                  provider: "google",
                  title: "Work Gmail",
                  type: "oauth2",
                },
              ]
            : [],
        ),
      ),
    );
    const client = createClient();
    seedCredentials(client);
    const { result, rerender } = renderHook(
      ({ expertId }) => useConnectedIntegrations(expertId),
      { wrapper: wrapperFor(client), initialProps: { expertId: "expert-a" } },
    );
    await waitFor(() =>
      expect(result.current.map((account) => account.credentialId)).toEqual([
        "work",
      ]),
    );
    expect(listCredentials).not.toHaveBeenCalled();
    rerender({ expertId: "expert-b" });
    expect(result.current).toEqual([]);
    await waitFor(() => expect(client.isFetching()).toBe(0));
    expect(result.current).toEqual([]);
  });

  it("does not fall back to owner accounts when expert grants fail", async () => {
    server.use(
      http.get("*/api/experts/:expertId/credentials", () =>
        HttpResponse.json({ detail: "unavailable" }, { status: 500 }),
      ),
    );
    const client = createClient();
    seedCredentials(client);
    const { result } = renderHook(() => useConnectedIntegrations("expert-a"), {
      wrapper: wrapperFor(client),
    });
    await waitFor(() => expect(client.isFetching()).toBe(0));
    expect(result.current).toEqual([]);
  });
});
