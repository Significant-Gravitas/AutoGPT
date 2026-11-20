import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { delay, http, HttpResponse } from "msw";
import type { ReactNode } from "react";
import { describe, expect, it } from "vitest";

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
});
