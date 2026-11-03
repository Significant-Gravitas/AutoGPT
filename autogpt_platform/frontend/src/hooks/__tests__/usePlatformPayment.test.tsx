import { getGetPlatformConfigMockHandler } from "@/app/api/__generated__/endpoints/platform-config/platform-config.msw";
import { server } from "@/mocks/mock-server";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { ReactNode } from "react";
import { describe, expect, it } from "vitest";
import { usePlatformPayment } from "../usePlatformPayment";

// The app's client keeps React Query's default retries, so the hook must opt out itself.
function renderPayment() {
  const client = new QueryClient();
  return renderHook(usePlatformPayment, {
    wrapper: ({ children }: { children: ReactNode }) => (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    ),
  });
}

describe("usePlatformPayment", () => {
  it.each([true, false])(
    "serves the backend's payment setting (%s)",
    async (enabled) => {
      server.use(getGetPlatformConfigMockHandler({ payment_enabled: enabled }));
      const { result } = renderPayment();
      expect(result.current.isResolved).toBe(false);
      await waitFor(() => expect(result.current.isResolved).toBe(true));
      expect(result.current.isPaymentEnabled).toBe(enabled);
    },
  );

  it("resolves at once with payment off when the backend cannot answer", async () => {
    let requests = 0;
    server.use(
      http.get("/api/proxy/api/platform-config", () => {
        requests += 1;
        return HttpResponse.json({ detail: "unavailable" }, { status: 503 });
      }),
    );
    const { result } = renderPayment();
    await waitFor(() => expect(result.current.isResolved).toBe(true));
    expect(result.current.isPaymentEnabled).toBe(false);
    expect(requests).toBe(1);
  });
});
