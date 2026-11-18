import {
  getPostV1CreateExecutionScheduleMockHandler,
  getPostV1CreateExecutionScheduleResponseMock,
} from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import { server } from "@/mocks/mock-server";
import * as invalidateSchedules from "@/services/schedules/invalidate-schedules";
import { act, renderHook, waitFor } from "@/tests/integrations/test-utils";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { http, HttpResponse } from "msw";
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

function renderFilledDialog() {
  const view = renderHook(
    () =>
      useCronSchedulerDialog({
        open: true,
        setOpen: vi.fn(),
        inputs: { topic: "news" },
        credentials: {},
      }),
    { wrapper: makeWrapper("?flowID=graph-xyz&flowVersion=2") },
  );
  act(() => {
    view.result.current.setCronExpression("0 9 * * *");
    view.result.current.setScheduleName("Morning run");
  });
  return view;
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
  window.sessionStorage.clear();
});

describe("useCronSchedulerDialog", () => {
  test("successful create invalidates ALL schedule queries (regression: previously invalidated NONE)", async () => {
    const requests: unknown[] = [];
    server.use(
      getPostV1CreateExecutionScheduleMockHandler(async ({ request }) => {
        requests.push(await request.json());
        return getPostV1CreateExecutionScheduleResponseMock();
      }),
    );
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
      result.current.setScheduleName("  Morning run  ");
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
    expect(requests).toEqual([
      expect.objectContaining({ name: "Morning run" }),
    ]);
    invalidateSpy.mockRestore();
  });

  test("a create whose answer was lost is sent again with its key, and a new create after success gets a new one (#16901)", async () => {
    const keys: unknown[] = [];
    server.use(
      http.post(
        "/api/proxy/api/graphs/:graphId/schedules",
        async ({ request }) => {
          const body = (await request.json()) as { schedule_id?: unknown };
          keys.push(body.schedule_id);
          return keys.length === 1
            ? HttpResponse.json({ detail: "HTTP 503" }, { status: 503 })
            : HttpResponse.json(getPostV1CreateExecutionScheduleResponseMock());
        },
      ),
    );
    const { result } = renderFilledDialog();

    await act(async () => {
      await result.current.handleCreateSchedule().catch(() => undefined);
    });
    await act(async () => {
      await result.current.handleCreateSchedule();
    });
    await act(async () => {
      await result.current.handleCreateSchedule();
    });

    const [key, retried, next] = keys;
    expect(key).toMatch(
      /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/,
    );
    expect(retried).toBe(key);
    expect(next).not.toBe(key);
  });

  test.each([
    {
      status: 409,
      detail: "Conflict",
      shown:
        "A schedule from an earlier try may already exist. Check your schedules before trying again.",
      linksToSchedules: true,
    },
    {
      status: 503,
      detail: "Service Unavailable",
      shown:
        "We couldn't confirm the schedule was saved, so it may already exist. Check your schedules before trying again.",
      linksToSchedules: true,
    },
    {
      status: 400,
      detail: "Cron expression '0 0 31 2 *' never fires",
      shown: "Cron expression '0 0 31 2 *' never fires",
      linksToSchedules: false,
    },
  ])(
    "a $status answer's toast says $shown",
    async ({ status, detail, shown, linksToSchedules }) => {
      server.use(
        http.post("/api/proxy/api/graphs/:graphId/schedules", () =>
          HttpResponse.json({ detail }, { status }),
        ),
      );
      const { result } = renderFilledDialog();

      await act(async () => {
        await result.current.handleCreateSchedule().catch(() => undefined);
      });

      expect(toastMock).toHaveBeenCalledWith(
        expect.objectContaining({ variant: "destructive", description: shown }),
      );
      expect(toastMock.mock.lastCall?.[0].action !== undefined).toBe(
        linksToSchedules,
      );
    },
  );

  test("empty cron expression shows destructive toast and skips the mutation", async () => {
    const createScheduleMock = vi.fn();
    server.use(getPostV1CreateExecutionScheduleMockHandler(createScheduleMock));

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

    act(() => {
      result.current.setScheduleName("Test Schedule");
    });

    await act(async () => {
      await result.current.handleCreateSchedule();
    });

    expect(toastMock).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Invalid schedule",
        description: "Please enter a valid cron expression",
        variant: "destructive",
      }),
    );
    expect(createScheduleMock).not.toHaveBeenCalled();
  });

  test.each(["", " ", "\t\n", "\u2003"])(
    "blank schedule name %j is rejected without a mutation",
    async (name) => {
      const createScheduleMock = vi.fn();
      server.use(
        getPostV1CreateExecutionScheduleMockHandler(createScheduleMock),
      );

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

      act(() => {
        result.current.setCronExpression("0 9 * * *");
        result.current.setScheduleName(name);
      });

      await act(async () => {
        await result.current.handleCreateSchedule();
      });

      expect(toastMock).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Invalid schedule",
          description: "Please enter a schedule name",
          variant: "destructive",
        }),
      );
      expect(result.current.scheduleNameError).toBe(
        "Schedule name is required",
      );
      expect(createScheduleMock).not.toHaveBeenCalled();
    },
  );
});

test("editing a rejected name clears the inline error", async () => {
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
  expect(result.current.scheduleNameError).toBe("Schedule name is required");
  act(() => {
    result.current.setScheduleName("Weekly report");
  });
  expect(result.current.scheduleNameError).toBe("");
});

test("changing the default cron does not erase a typed schedule name", () => {
  const { result, rerender } = renderHook(
    ({ cron }) =>
      useCronSchedulerDialog({
        open: true,
        setOpen: vi.fn(),
        inputs: {},
        credentials: {},
        defaultCronExpression: cron,
      }),
    {
      initialProps: { cron: "0 9 * * *" },
      wrapper: makeWrapper("?flowID=g&flowVersion=1"),
    },
  );
  act(() => {
    result.current.setScheduleName("Weekly report");
  });
  rerender({ cron: "0 10 * * *" });
  expect(result.current.scheduleName).toBe("Weekly report");
});
