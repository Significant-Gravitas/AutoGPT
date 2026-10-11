import { server } from "@/mocks/mock-server";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook } from "@testing-library/react";
import { delay, http, HttpResponse } from "msw";
import type { ReactNode } from "react";
import { describe, expect, it } from "vitest";
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
});
