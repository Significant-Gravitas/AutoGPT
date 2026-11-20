import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { createElement, type ReactNode } from "react";
import { describe, expect, it } from "vitest";
import { GOOGLE_ACCESS_SCOPES } from "@/lib/google-access";
import { server } from "@/mocks/mock-server";
import { useSignInConnection } from "../useSignInConnection";

// One query cache for the tab, as the app has.
function tabQueryCache() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return function Wrapper({ children }: { children: ReactNode }) {
    return createElement(QueryClientProvider, { client }, children);
  };
}

function googleConnections(scopes: readonly string[] | null) {
  server.use(
    http.get("*/api/integrations/google/credentials", () =>
      HttpResponse.json(
        scopes
          ? [
              {
                id: "google-1",
                provider: "google",
                type: "oauth2",
                title: null,
                username: "pat@example.com",
                scopes: [...scopes],
              },
            ]
          : [],
      ),
    ),
  );
}

describe("useSignInConnection", () => {
  it("answers for each account, never with another's connections", async () => {
    googleConnections(GOOGLE_ACCESS_SCOPES);
    const { result, rerender } = renderHook(
      ({ userID }) => useSignInConnection(userID, "google"),
      { initialProps: { userID: "user-1" }, wrapper: tabQueryCache() },
    );
    await waitFor(() => expect(result.current.isConnected).toBe(true));

    // Another account signs in, in the same tab, with nothing connected.
    googleConnections(null);
    rerender({ userID: "user-2" });

    await waitFor(() => expect(result.current.isReady).toBe(true));
    expect(result.current.isConnected).toBe(false);
  });
});
