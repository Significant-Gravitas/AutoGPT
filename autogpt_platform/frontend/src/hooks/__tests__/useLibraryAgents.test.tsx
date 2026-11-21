import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { delay, http, HttpResponse } from "msw";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { getGetV2ListLibraryAgentsResponseMock } from "@/app/api/__generated__/endpoints/library/library.msw";
import { server } from "@/mocks/mock-server";
import { useLibraryAgents } from "../useLibraryAgents/useLibraryAgents";

function makeWrapper() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    );
  }
  return Wrapper;
}

const template = getGetV2ListLibraryAgentsResponseMock().agents[0];

// Claims `totalItems` agents and serves `perPage` of them per page (default:
// the requested page_size), recording every page asked for. `latencyMs` lets
// React render the in-flight state between pages, as a real network does.
function servePages({
  totalItems,
  perPage,
  failPage,
  latencyMs = 10,
}: {
  totalItems: number;
  perPage?: number;
  failPage?: number;
  latencyMs?: number;
}) {
  const requested: number[] = [];
  server.use(
    http.get("*/api/library/agents", async ({ request }) => {
      if (latencyMs > 0) await delay(latencyMs);
      const params = new URL(request.url).searchParams;
      const page = Number(params.get("page") ?? 1);
      const pageSize = Number(params.get("page_size") ?? 15);
      requested.push(page);
      if (page === failPage) {
        return HttpResponse.json({ detail: "boom" }, { status: 500 });
      }
      const start = (page - 1) * pageSize;
      const count = Math.min(perPage ?? pageSize, totalItems - start);
      return HttpResponse.json({
        agents: Array.from({ length: Math.max(0, count) }, (_, i) => ({
          ...template,
          id: `la-${page}-${i}`,
          graph_id: `g-${page}-${i}`,
        })),
        pagination: {
          total_items: totalItems,
          total_pages: Math.ceil(totalItems / pageSize),
          current_page: page,
          page_size: pageSize,
        },
      });
    }),
  );
  return requested;
}

function settle() {
  return new Promise((resolve) => setTimeout(resolve, 100));
}

describe("useLibraryAgents", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("loads every page, not only the first 100 agents", async () => {
    // Instant replies: the next page must still be requested when React
    // never renders the in-flight state in between.
    const requested = servePages({ totalItems: 250, latencyMs: 0 });

    const { result } = renderHook(() => useLibraryAgents(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => expect(result.current.agents).toHaveLength(250));
    expect(result.current.agentInfoMap.size).toBe(250);
    expect(requested).toEqual([1, 2, 3]);
  });

  it("stops paging after a page fails instead of re-requesting it", async () => {
    const requested = servePages({ totalItems: 350, failPage: 2 });

    const { result } = renderHook(() => useLibraryAgents(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => expect(requested).toContain(2));
    await settle();
    expect(requested).toEqual([1, 2]);
    expect(result.current.agents).toHaveLength(100);
  });

  it("stops at the page cap when the server keeps reporting more", async () => {
    const requested = servePages({ totalItems: 1_000_000, perPage: 1 });

    const { result } = renderHook(() => useLibraryAgents(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => expect(requested).toHaveLength(20), { timeout: 5000 });
    await settle();
    expect(requested).toHaveLength(20);
    expect(result.current.agents).toHaveLength(20);
  });

  it("lets an active multi-page reread finish when the probe answers again", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    // The scheduler is out, then back; the lists' schedules are unknown
    // until a reread answers with known ones. Two pages of 100, so a
    // reread is a two-step read the probe can answer twice during.
    const totalItems = 150;
    let schedulerUp = false;
    let recoveredAgents = false;
    let probeReads = 0;
    const requested: number[] = [];
    const page2Holds: Array<() => void> = [];
    let page2Requests = 0;

    server.use(
      http.get("*/api/library/agents", async ({ request }) => {
        const params = new URL(request.url).searchParams;
        const page = Number(params.get("page") ?? 1);
        const pageSize = Number(params.get("page_size") ?? 100);
        requested.push(page);
        if (page === 2) {
          page2Requests += 1;
          // Hold every reread of the second page, so the reread spans the
          // next probe answer; the initial load answers at once.
          if (page2Requests > 1) {
            await new Promise<void>((resolve) => {
              page2Holds.push(resolve);
            });
          }
        }
        const start = (page - 1) * pageSize;
        const count = Math.min(pageSize, totalItems - start);
        return HttpResponse.json({
          agents: Array.from({ length: Math.max(0, count) }, (_, i) => ({
            ...template,
            id: `la-${page}-${i}`,
            graph_id: `g-${page}-${i}`,
            schedule_info_unavailable: !recoveredAgents,
          })),
          pagination: {
            total_items: totalItems,
            total_pages: 2,
            current_page: page,
            page_size: pageSize,
          },
        });
      }),
      http.get("/api/proxy/api/schedules", () => {
        probeReads += 1;
        if (schedulerUp) return HttpResponse.json([]);
        return HttpResponse.json(
          { detail: "Scheduler unavailable" },
          { status: 503, headers: { "Retry-After": "15" } },
        );
      }),
    );

    const { result } = renderHook(() => useLibraryAgents(), {
      wrapper: makeWrapper(),
    });

    try {
      // Two pages load, and the probe reads the outage.
      await waitFor(() => expect(requested).toEqual([1, 2]));
      await waitFor(() => expect(probeReads).toBe(1));
      expect(result.current.agents).toHaveLength(totalItems);
      expect(
        result.current.agents.some((a) => a.schedule_info_unavailable),
      ).toBe(true);

      // The scheduler answers: the lists re-read once. The second page of
      // the reread is held, so the reread is still in flight when the
      // probe answers again.
      schedulerUp = true;
      recoveredAgents = true;
      await act(async () => {
        vi.advanceTimersByTime(15_000);
      });
      await waitFor(() => expect(probeReads).toBe(2));
      await waitFor(() => expect(requested).toEqual([1, 2, 1, 2]));

      // The probe answers again while the reread is still in flight. A
      // cancelling refetch restarts the reread and asks for the first page
      // again; a reusing one leaves the in-flight reread alone.
      await act(async () => {
        vi.advanceTimersByTime(15_000);
      });
      await waitFor(() => expect(probeReads).toBe(3));
      await act(async () => {
        for (let i = 0; i < 10; i++) await Promise.resolve();
      });
      expect(requested).toEqual([1, 2, 1, 2]);

      // The held second page answers: the one reread finishes, and the
      // schedules are known.
      page2Holds.forEach((release) => release());
      await waitFor(() =>
        expect(
          result.current.agents.every((a) => !a.schedule_info_unavailable),
        ).toBe(true),
      );
      expect(requested).toEqual([1, 2, 1, 2]);
    } finally {
      page2Holds.forEach((release) => release());
      vi.useRealTimers();
    }
  });
});
