import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it } from "vitest";

import { server } from "@/mocks/mock-server";

import { useUsageIndicator } from "../useUsageIndicator";

function makeWrapper() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    );
  }
  return Wrapper;
}

function usageHandler(body: { daily: { percent_used: number } | null }) {
  return http.get("*/api/chat/usage", () => HttpResponse.json(body));
}

afterEach(() => {
  server.resetHandlers();
});

describe("useUsageIndicator", () => {
  it("rounds and clamps the daily percent used", async () => {
    server.use(usageHandler({ daily: { percent_used: 42.7 } }));

    const { result } = renderHook(() => useUsageIndicator(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });
    expect(result.current.percent).toBe(43);
  });

  it("clamps percentages above 100 to 100", async () => {
    server.use(usageHandler({ daily: { percent_used: 150 } }));

    const { result } = renderHook(() => useUsageIndicator(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => {
      expect(result.current.percent).toBe(100);
    });
  });

  it("clamps negative percentages to 0", async () => {
    server.use(usageHandler({ daily: { percent_used: -20 } }));

    const { result } = renderHook(() => useUsageIndicator(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => {
      expect(result.current.percent).toBe(0);
    });
  });

  it("returns a null percent when there is no daily usage", async () => {
    server.use(usageHandler({ daily: null }));

    const { result } = renderHook(() => useUsageIndicator(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });
    expect(result.current.percent).toBeNull();
  });
});
