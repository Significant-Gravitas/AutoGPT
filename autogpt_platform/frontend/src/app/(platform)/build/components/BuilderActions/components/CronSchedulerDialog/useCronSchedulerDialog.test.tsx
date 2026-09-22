import { getPostV1CreateExecutionScheduleMockHandler } from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import { server } from "@/mocks/mock-server";
import * as invalidateSchedules from "@/services/schedules/invalidate-schedules";
import { act, renderHook, waitFor } from "@/tests/integrations/test-utils";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import type { ReactNode } from "react";
import { afterEach, describe, expect, test, vi } from "vitest";
import { useCronSchedulerDialog } from "./useCronSchedulerDialog";

function makeWrapper(searchParams: string) {
  return function Wrapper({ children }: { children: ReactNode }) {
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    return (
      <QueryClientProvider client={client}>
        <NuqsTestingAdapter searchParams={searchParams}>
          {children}
        </NuqsTestingAdapter>
      </QueryClientProvider>
    );
  };
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

describe("useCronSchedulerDialog", () => {
  test("successful create invalidates ALL schedule queries (regression: previously invalidated NONE)", async () => {
    server.use(getPostV1CreateExecutionScheduleMockHandler());
    const invalidateSpy = vi.spyOn(
      invalidateSchedules,
      "invalidateAllScheduleQueries",
    );

    const setOpen = vi.fn();
    const { result } = renderHook(
      () =>
        useCronSchedulerDialog({
          open: true,
          setOpen,
          inputs: {},
          credentials: {},
        }),
      { wrapper: makeWrapper("?flowID=graph-xyz&flowVersion=2") },
    );

    act(() => {
      result.current.setCronExpression("0 9 * * *");
      result.current.setScheduleName("Morning run");
    });

    await act(async () => {
      await result.current.handleCreateSchedule();
    });

    await waitFor(() => {
      expect(invalidateSpy).toHaveBeenCalledWith(
        expect.anything(),
        "graph-xyz",
      );
    });
    invalidateSpy.mockRestore();
  });

  test("empty cron expression shows destructive toast and skips the mutation", async () => {
    const { result } = renderHook(
      () =>
        useCronSchedulerDialog({
          open: true,
          setOpen: vi.fn(),
          inputs: {},
          credentials: {},
        }),
      { wrapper: makeWrapper("?flowID=g&flowVersion=1") },
    );

    await act(async () => {
      await result.current.handleCreateSchedule();
    });

    expect(toastMock).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Invalid schedule",
        variant: "destructive",
      }),
    );
  });
});
