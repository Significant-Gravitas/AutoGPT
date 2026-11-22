import { getDeleteV1DeleteExecutionScheduleMockHandler } from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import { server } from "@/mocks/mock-server";
import * as invalidateSchedules from "@/services/schedules/invalidate-schedules";
import { act, renderHook, waitFor } from "@/tests/integrations/test-utils";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { afterEach, describe, expect, test, vi } from "vitest";
import { useScheduleDetailHeader } from "./useScheduleDetailHeader";

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

const toastMock = vi.fn();
vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/components/molecules/Toast/use-toast")
    >();
  return {
    ...actual,
    useToast: () => ({ toast: toastMock }),
  };
});

afterEach(() => {
  toastMock.mockClear();
  server.resetHandlers();
});

describe("useScheduleDetailHeader", () => {
  test("deleteSchedule calls the unified invalidator with the graphId so unified page + briefing pill refresh", async () => {
    server.use(getDeleteV1DeleteExecutionScheduleMockHandler());
    const invalidateSpy = vi.spyOn(
      invalidateSchedules,
      "invalidateAllScheduleQueries",
    );

    const { result } = renderHook(
      () => useScheduleDetailHeader("graph-xyz", "sched-1"),
      { wrapper },
    );

    act(() => {
      result.current.deleteSchedule();
    });

    await waitFor(() => {
      expect(toastMock).toHaveBeenCalledWith(
        expect.objectContaining({ title: "Schedule deleted" }),
      );
    });

    expect(invalidateSpy).toHaveBeenCalledWith(expect.anything(), "graph-xyz");
    invalidateSpy.mockRestore();
  });

  test("deleteSchedule is a no-op when scheduleId is undefined", () => {
    const { result } = renderHook(
      () => useScheduleDetailHeader("graph-xyz", undefined),
      { wrapper },
    );
    // Should not throw and isDeleting should never flip.
    act(() => {
      result.current.deleteSchedule();
    });
    expect(result.current.isDeleting).toBe(false);
  });
});
