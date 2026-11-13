import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SESSION_GROUPS_QUERY_KEY } from "../sessionGroups";
import { useSessionTitlePoll } from "../useSessionTitlePoll";

const GROUPS_KEY = [...SESSION_GROUPS_QUERY_KEY, { pinnedSeparately: false }];

function groupedRead(title: string | null) {
  return {
    status: 200,
    headers: new Headers(),
    data: {
      groups: [
        {
          expert_id: null,
          total: 1,
          sessions: [
            {
              id: "s1",
              title,
              is_processing: false,
              created_at: "2026-11-12T10:00:00Z",
              updated_at: "2026-11-12T10:00:00Z",
            },
          ],
        },
      ],
      pinned: [],
    },
  };
}

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

describe("useSessionTitlePoll", () => {
  it("stops once the sidebar's grouped read shows the new chat's title", () => {
    const client = new QueryClient();
    client.setQueryData(GROUPS_KEY, groupedRead(null));
    const invalidate = vi.spyOn(client, "invalidateQueries");
    function wrapper({ children }: { children: ReactNode }) {
      return (
        <QueryClientProvider client={client}>{children}</QueryClientProvider>
      );
    }
    const { rerender } = renderHook(useSessionTitlePoll, {
      wrapper,
      initialProps: {
        sessionId: "s1",
        status: "streaming",
        isReconnecting: false,
      },
    });

    rerender({ sessionId: "s1", status: "ready", isReconnecting: false });
    act(() => vi.advanceTimersByTime(2_000));
    const callsWhileUntitled = invalidate.mock.calls.length;
    expect(callsWhileUntitled).toBe(2);

    client.setQueryData(GROUPS_KEY, groupedRead("Plan the launch"));
    act(() => vi.advanceTimersByTime(12_000));

    expect(invalidate).toHaveBeenCalledTimes(callsWhileUntitled);
  });
});
