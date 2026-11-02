import { getPostV1CreateExecutionScheduleMockHandler } from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { server } from "@/mocks/mock-server";
import * as invalidateSchedules from "@/services/schedules/invalidate-schedules";
import { act, renderHook, waitFor } from "@/tests/integrations/test-utils";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { afterEach, describe, expect, test, vi } from "vitest";
import { useScheduleAgentModal } from "./useScheduleAgentModal";

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
  name: "My agent",
  recommended_schedule_cron: null,
} as unknown as LibraryAgent;

afterEach(() => {
  toastMock.mockClear();
  server.resetHandlers();
});

describe("useScheduleAgentModal", () => {
  test("create schedule invalidates ALL schedule queries on success so unified page + briefing pill refresh", async () => {
    server.use(getPostV1CreateExecutionScheduleMockHandler());
    const invalidateSpy = vi.spyOn(
      invalidateSchedules,
      "invalidateAllScheduleQueries",
    );

    const { result } = renderHook(() => useScheduleAgentModal(agent, {}, {}), {
      wrapper,
    });

    await act(async () => {
      await result.current.handleSchedule("My schedule", "0 9 * * *");
    });

    await waitFor(() => {
      expect(toastMock).toHaveBeenCalledWith(
        expect.objectContaining({ title: "Schedule created" }),
      );
    });
    expect(invalidateSpy).toHaveBeenCalledWith(expect.anything(), "graph-xyz");
    invalidateSpy.mockRestore();
  });

  test("empty schedule name surfaces destructive toast and rejects without firing mutation", async () => {
    const { result } = renderHook(() => useScheduleAgentModal(agent, {}, {}), {
      wrapper,
    });

    await expect(
      result.current.handleSchedule("   ", "0 9 * * *"),
    ).rejects.toThrow("Schedule name required");
    expect(toastMock).toHaveBeenCalledWith(
      expect.objectContaining({ variant: "destructive" }),
    );
  });
});
