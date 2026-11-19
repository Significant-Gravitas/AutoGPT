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
    // Hold the fresh probe answer open, so the test pins that the cached
    // success alone cannot trigger recovery while the new answer is still
    // pending.
    let releaseProbe = () => {};
    let probeRequested = false;
    const probeAnswered = new Promise<void>((resolve) => {
      releaseProbe = () => resolve();
    });
    server.use(
      http.get("/api/proxy/api/schedules", async () => {
        probeRequested = true;
        await probeAnswered;
        return HttpResponse.json([]);
      }),
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
    try {
      const { result, rerender } = renderHook(() => useSchedulerProbe(true), {
        wrapper,
      });

      // The probe has started but not answered. The hook only re-evaluates
      // its recovery predicate on a render, so re-render while the cached
      // pre-outage success is still the only answer: it must not count.
      await waitFor(() => expect(probeRequested).toBe(true), {
        timeout: 5_000,
      });
      rerender();
      expect(result.current.recovered).toBe(false);

      // The mount re-read answers after the schedules went unknown, so the
      // cached pre-outage success is replaced by a fresh one.
      releaseProbe();
      await waitFor(() => expect(result.current.recovered).toBe(true), {
        timeout: 5_000,
      });
    } finally {
      // Never leave the held response pending: it would outlive the test.
      releaseProbe();
    }
  });
});
