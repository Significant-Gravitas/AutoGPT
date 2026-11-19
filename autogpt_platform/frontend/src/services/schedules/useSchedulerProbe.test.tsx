import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { http, HttpResponse } from "msw";
import type { ReactNode } from "react";
import { describe, expect, test } from "vitest";
import { getGetV1ListExecutionSchedulesForAUserQueryKey } from "@/app/api/__generated__/endpoints/schedules/schedules";
import { server } from "@/mocks/mock-server";
import { renderHook, waitFor } from "@/tests/integrations/test-utils";
import { useSchedulerProbe } from "./useSchedulerProbe";

describe("useSchedulerProbe", () => {
  test("recovers from a cached success once the probe answers again", async () => {
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    // A success cached before the outage: its timestamp predates the unknown
    // period, so it must not count — until the probe answers again.
    queryClient.setQueryData(
      getGetV1ListExecutionSchedulesForAUserQueryKey(),
      [],
      { updatedAt: Date.now() - 60_000 },
    );
    server.use(
      http.get("/api/proxy/api/schedules", () => HttpResponse.json([])),
    );

    const wrapper = function ProbeWrapper({
      children,
    }: {
      children: ReactNode;
    }) {
      return (
        <QueryClientProvider client={queryClient}>
          {children}
        </QueryClientProvider>
      );
    };
    const { result } = renderHook(() => useSchedulerProbe(true), {
      wrapper,
    });

    // The mount re-read answers after the schedules went unknown, so the
    // cached pre-outage success is replaced by a fresh one.
    await waitFor(() => expect(result.current.recovered).toBe(true), {
      timeout: 5_000,
    });
  });
});
