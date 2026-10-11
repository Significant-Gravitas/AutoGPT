import { getPatchV1UpdateExecutionScheduleMockHandler } from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { server } from "@/mocks/mock-server";
import * as invalidateSchedules from "@/services/schedules/invalidate-schedules";
import { act, renderHook, waitFor } from "@/tests/integrations/test-utils";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { http, HttpResponse } from "msw";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { useEditScheduleModal } from "./useEditScheduleModal";

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
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

const schedule = {
  id: "sched-1",
  name: "Daily",
  cron: "0 9 * * *",
} as unknown as GraphExecutionJobInfo;

beforeEach(() => {
  toastMock.mockClear();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("useEditScheduleModal", () => {
  test("saves the new name and time through the schedules PATCH endpoint", async () => {
    let received: { scheduleId: unknown; body: unknown } | undefined;
    server.use(
      getPatchV1UpdateExecutionScheduleMockHandler(
        async ({ request, params }) => {
          received = {
            scheduleId: params.scheduleId,
            body: await request.json(),
          };
          return { ...schedule, cron: "30 18 * * *" };
        },
      ),
    );
    const invalidateSpy = vi.spyOn(
      invalidateSchedules,
      "invalidateAllScheduleQueries",
    );

    const { result } = renderHook(
      () => useEditScheduleModal("graph-xyz", schedule),
      { wrapper },
    );
    act(() => {
      result.current.setName("Evening");
      result.current.setTime("18:30");
    });
    act(() => {
      result.current.handleSave();
    });

    await waitFor(() => {
      expect(invalidateSpy).toHaveBeenCalledWith(
        expect.anything(),
        "graph-xyz",
      );
    });
    expect(received).toEqual({
      scheduleId: "sched-1",
      body: { name: "Evening", cron: "30 18 * * *" },
    });
  });

  test("a rejected save shows the server's reason and does not invalidate", async () => {
    server.use(
      http.patch("/api/proxy/api/schedules/:scheduleId", () =>
        HttpResponse.json({ detail: "Invalid cron" }, { status: 400 }),
      ),
    );
    const invalidateSpy = vi.spyOn(
      invalidateSchedules,
      "invalidateAllScheduleQueries",
    );

    const { result } = renderHook(
      () => useEditScheduleModal("graph-xyz", schedule),
      { wrapper },
    );
    act(() => {
      result.current.handleSave();
    });

    await waitFor(() => {
      expect(toastMock).toHaveBeenCalledWith(
        expect.objectContaining({
          description: expect.stringContaining("Invalid cron"),
          variant: "destructive",
        }),
      );
    });
    expect(invalidateSpy).not.toHaveBeenCalled();
  });
});
