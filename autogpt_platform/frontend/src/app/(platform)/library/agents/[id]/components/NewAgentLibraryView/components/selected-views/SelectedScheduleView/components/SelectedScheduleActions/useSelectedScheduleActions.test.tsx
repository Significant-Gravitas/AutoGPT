import { getDeleteV1DeleteExecutionScheduleMockHandler } from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { server } from "@/mocks/mock-server";
import * as invalidateSchedules from "@/services/schedules/invalidate-schedules";
import { act, renderHook, waitFor } from "@/tests/integrations/test-utils";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { afterEach, describe, expect, test, vi } from "vitest";
import { useSelectedScheduleActions } from "./useSelectedScheduleActions";

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

const agent = {
  id: "lib-1",
  graph_id: "graph-xyz",
  graph_version: 1,
} as unknown as LibraryAgent;

afterEach(() => {
  toastMock.mockClear();
  server.resetHandlers();
});

describe("useSelectedScheduleActions", () => {
  test("handleDelete invalidates ALL schedule queries for the graph on success", async () => {
    server.use(getDeleteV1DeleteExecutionScheduleMockHandler());
    const invalidateSpy = vi.spyOn(
      invalidateSchedules,
      "invalidateAllScheduleQueries",
    );

    const { result } = renderHook(
      () =>
        useSelectedScheduleActions({
          agent,
          scheduleId: "sched-1",
        }),
      { wrapper },
    );

    act(() => {
      result.current.handleDelete();
    });

    await waitFor(() => {
      expect(toastMock).toHaveBeenCalledWith(
        expect.objectContaining({ title: "Schedule deleted" }),
      );
    });

    expect(invalidateSpy).toHaveBeenCalledWith(expect.anything(), "graph-xyz");
    invalidateSpy.mockRestore();
  });

  test("openInBuilderHref points to /build with graph id + version", () => {
    const { result } = renderHook(
      () =>
        useSelectedScheduleActions({
          agent,
          scheduleId: "sched-1",
        }),
      { wrapper },
    );
    expect(result.current.openInBuilderHref).toBe(
      "/build?flowID=graph-xyz&flowVersion=1",
    );
  });

  test("handleRunNow without a loaded schedule surfaces a destructive toast", async () => {
    const { result } = renderHook(
      () =>
        useSelectedScheduleActions({
          agent,
          scheduleId: "sched-1",
        }),
      { wrapper },
    );

    await act(async () => {
      await result.current.handleRunNow();
    });

    expect(toastMock).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Schedule not loaded",
        variant: "destructive",
      }),
    );
  });
});
