import { getPostV1CreateExecutionScheduleMockHandler } from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { server } from "@/mocks/mock-server";
import { http, HttpResponse } from "msw";
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

  test("a create the scheduler didn't confirm is re-read, and its retry can't add a duplicate", async () => {
    const sentKeys: unknown[] = [];
    let calls = 0;
    server.use(
      http.post("*/api/graphs/:graphId/schedules", async ({ request }) => {
        const body = (await request.json()) as { schedule_id?: unknown };
        sentKeys.push(body.schedule_id);
        calls += 1;
        if (calls === 1) {
          return HttpResponse.json(
            {
              detail: {
                code: "scheduler_unconfirmed",
                message: "The scheduler didn't confirm the change in time.",
              },
            },
            { status: 503 },
          );
        }
        return HttpResponse.json({
          id: String(body.schedule_id),
          name: "My schedule",
          user_id: "user-1",
          graph_id: "graph-xyz",
          graph_version: 1,
          cron: "0 9 * * *",
          input_data: {},
          next_run_time: new Date().toISOString(),
        });
      }),
    );
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    const invalidate = vi.spyOn(client, "invalidateQueries");
    const { result } = renderHook(() => useScheduleAgentModal(agent, {}, {}), {
      wrapper: ({ children }: { children: ReactNode }) => (
        <QueryClientProvider client={client}>{children}</QueryClientProvider>
      ),
    });

    await act(async () => {
      await expect(
        result.current.handleSchedule("My schedule", "0 9 * * *"),
      ).rejects.toBeDefined();
    });
    // The create may have been applied, so the schedule lists are re-read.
    expect(invalidate).toHaveBeenCalled();

    await act(async () => {
      await result.current.handleSchedule("My schedule", "0 9 * * *");
    });
    await act(async () => {
      await result.current.handleSchedule("Another schedule", "0 9 * * *");
    });

    expect(sentKeys).toHaveLength(3);
    expect(sentKeys[0]).toMatch(/^[0-9a-f-]{36}$/);
    // The retry is the same submission, so the scheduler returns the schedule
    // it made instead of creating a second one...
    expect(sentKeys[1]).toBe(sentKeys[0]);
    // ...while a new submission after a success gets its own key.
    expect(sentKeys[2]).not.toBe(sentKeys[0]);
  });
});
