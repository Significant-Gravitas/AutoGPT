import { server } from "@/mocks/mock-server";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook } from "@testing-library/react";
import { delay, http, HttpResponse } from "msw";
import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";
import { useAtResetJobs } from "../useAtResetJobs";

const JOB = {
  kind: "send" as const,
  run_at: new Date(2026, 9, 12, 1, 7),
  message: "Draft the launch post",
};

function withQueryClient() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    );
  };
}

describe("useAtResetJobs", () => {
  it("keeps a write that a read already on its way would otherwise undo", async () => {
    server.use(
      http.get("*/api/chat/sessions/:sessionId/at-reset", async () => {
        await delay(150);
        return HttpResponse.json({ jobs: [] });
      }),
    );
    const { result } = renderHook(() => useAtResetJobs("session-1"), {
      wrapper: withQueryClient(),
    });

    await act(async () => {
      await result.current.writeJobs(() => [JOB]);
    });
    await act(() => delay(300));

    expect(result.current.jobs?.map((job) => job.message)).toEqual([
      "Draft the launch post",
    ]);
  });

  it("looks again as a set turn comes due, then often until it has gone", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    vi.setSystemTime(JOB.run_at.getTime() - 20_000);
    let reads = 0;
    server.use(
      http.get("*/api/chat/sessions/:sessionId/at-reset", () => {
        reads += 1;
        return HttpResponse.json({ jobs: [JOB] });
      }),
    );
    try {
      renderHook(() => useAtResetJobs("session-1"), {
        wrapper: withQueryClient(),
      });
      await vi.waitFor(() => expect(reads).toBe(1));

      await act(() => vi.advanceTimersByTimeAsync(21_000));
      expect(reads).toBe(2);

      await act(() => vi.advanceTimersByTimeAsync(10_000));
      expect(reads).toBe(3);
    } finally {
      vi.useRealTimers();
    }
  });

  it("looks at a turn hours away only now and then", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    vi.setSystemTime(JOB.run_at.getTime() - 3 * 60 * 60_000);
    let reads = 0;
    server.use(
      http.get("*/api/chat/sessions/:sessionId/at-reset", () => {
        reads += 1;
        return HttpResponse.json({ jobs: [JOB] });
      }),
    );
    try {
      renderHook(() => useAtResetJobs("session-1"), {
        wrapper: withQueryClient(),
      });
      await vi.waitFor(() => expect(reads).toBe(1));

      await act(() => vi.advanceTimersByTimeAsync(10 * 60_000));
      expect(reads).toBe(1);

      await act(() => vi.advanceTimersByTimeAsync(5 * 60_000));
      expect(reads).toBe(2);
    } finally {
      vi.useRealTimers();
    }
  });
});
