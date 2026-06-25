import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import * as invalidateSchedules from "@/services/schedules/invalidate-schedules";
import { act, renderHook, waitFor } from "@/tests/integrations/test-utils";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { useEditScheduleModal } from "./useEditScheduleModal";

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

const fetchMock = vi.fn();
const originalFetch = globalThis.fetch;

const schedule = {
  id: "sched-1",
  name: "Daily",
  cron: "0 9 * * *",
} as unknown as GraphExecutionJobInfo;

beforeEach(() => {
  fetchMock.mockReset();
  toastMock.mockClear();
  globalThis.fetch = fetchMock as unknown as typeof globalThis.fetch;
});

afterEach(() => {
  vi.restoreAllMocks();
  globalThis.fetch = originalFetch;
});

describe("useEditScheduleModal", () => {
  test("successful PATCH invalidates ALL schedule queries (regression: previously only invalidated per-graph)", async () => {
    fetchMock.mockResolvedValueOnce({
      ok: true,
      json: async () => ({}),
    });
    const invalidateSpy = vi.spyOn(
      invalidateSchedules,
      "invalidateAllScheduleQueries",
    );

    const { result } = renderHook(
      () => useEditScheduleModal("graph-xyz", schedule),
      { wrapper },
    );

    await act(async () => {
      await result.current.mutateAsync();
    });

    await waitFor(() => {
      expect(invalidateSpy).toHaveBeenCalledWith(
        expect.anything(),
        "graph-xyz",
      );
    });
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/schedules/sched-1",
      expect.objectContaining({ method: "PATCH" }),
    );
  });

  test("failed PATCH surfaces destructive toast and does NOT invalidate", async () => {
    fetchMock.mockResolvedValueOnce({
      ok: false,
      json: async () => ({ message: "Bad cron" }),
    });
    const invalidateSpy = vi.spyOn(
      invalidateSchedules,
      "invalidateAllScheduleQueries",
    );

    const { result } = renderHook(
      () => useEditScheduleModal("graph-xyz", schedule),
      { wrapper },
    );

    await expect(
      act(async () => {
        await result.current.mutateAsync();
      }),
    ).rejects.toThrow();

    await waitFor(() => {
      expect(toastMock).toHaveBeenCalledWith(
        expect.objectContaining({
          variant: "destructive",
        }),
      );
    });
    expect(invalidateSpy).not.toHaveBeenCalled();
  });
});
