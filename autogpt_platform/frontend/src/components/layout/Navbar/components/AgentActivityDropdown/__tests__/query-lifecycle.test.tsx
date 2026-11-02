import { getGetV1ListAllExecutionsMockHandler200 } from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import BackendAPI from "@/lib/autogpt-server-api/client";
import { getGetV2ListLibraryAgentsInfiniteQueryOptions } from "@/app/api/__generated__/endpoints/library/library";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { StrictMode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { http, HttpResponse } from "msw";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AccountMenuActivityRow } from "../../AccountMenu/components/AccountMenuActivityRow";
import {
  deferred,
  earlyAgent,
  execution,
  firstPageAgents,
  lateAgent,
  libraryKey,
  libraryURL,
  page,
  renderActivity,
  settleRequests,
} from "./fixtures";

afterEach(() => vi.restoreAllMocks());

describe("Shared activity query lifecycle", () => {
  it("shares a single page request across mounted consumers", async () => {
    const next = deferred();
    const requests: number[] = [];
    server.use(
      http.get(libraryURL, async ({ request }) => {
        const current = Number(new URL(request.url).searchParams.get("page"));
        requests.push(current);
        if (current === 2) await next.promise;
        return HttpResponse.json(
          current === 1 ? page(firstPageAgents) : page([lateAgent], 2),
        );
      }),
    );
    const { queryClient } = renderActivity();
    await waitFor(() => expect(requests).toEqual([1, 2]));
    const second = render(
      <QueryClientProvider client={queryClient}>
        <AccountMenuActivityRow />
      </QueryClientProvider>,
    );
    await settleRequests();
    expect(requests).toEqual([1, 2]);
    second.unmount();
    next.resolve();
    await screen.findByRole("link", { name: /Late-page workflow/ });
    expect(requests).toEqual([1, 2]);
  });

  it("waits for a background refresh before loading later pages", async () => {
    const refresh = deferred();
    const requests: number[] = [];
    let aborted = false;
    server.use(
      http.get(libraryURL, async ({ request }) => {
        const current = Number(new URL(request.url).searchParams.get("page"));
        requests.push(current);
        request.signal.addEventListener("abort", () => {
          aborted = true;
        });
        if (current === 1) await refresh.promise;
        return HttpResponse.json(
          current === 1 ? page(firstPageAgents) : page([lateAgent], 2),
        );
      }),
    );
    const { queryClient } = renderActivity();
    await waitFor(() => expect(requests).toEqual([1]));
    queryClient.setQueryData(libraryKey, {
      pages: [
        { status: 200, data: page(firstPageAgents), headers: new Headers() },
      ],
      pageParams: [1],
    });
    await settleRequests();
    expect(requests).toEqual([1]);
    expect(aborted).toBe(false);
    refresh.resolve();
    await screen.findByRole("link", { name: /Late-page workflow/ });
    expect(requests).toEqual([1, 2]);
    expect(aborted).toBe(false);
  });

  it("retries a failed background refresh before advancing to the next page", async () => {
    const next = deferred();
    let failRefresh = false;
    let firstPage = firstPageAgents;
    const requests: number[] = [];
    server.use(
      http.get(libraryURL, async ({ request }) => {
        const current = Number(new URL(request.url).searchParams.get("page"));
        requests.push(current);
        if (current === 1) {
          return failRefresh
            ? HttpResponse.json({ detail: "Unavailable" }, { status: 503 })
            : HttpResponse.json(page(firstPage));
        }
        await next.promise;
        return HttpResponse.json(page([lateAgent], 2));
      }),
    );
    const { queryClient } = renderActivity([
      execution({ graph_id: earlyAgent.graph_id }),
    ]);
    await waitFor(() => expect(requests).toEqual([1, 2]));
    failRefresh = true;
    await queryClient.invalidateQueries({ queryKey: libraryKey });
    next.resolve();
    await screen.findByRole("button", { name: "Try Again" });
    const previousAttempts = requests.length;
    failRefresh = false;
    firstPage = [
      { ...earlyAgent, name: "Refreshed workflow" },
      ...firstPageAgents.slice(1),
    ];
    fireEvent.click(screen.getByRole("button", { name: "Try Again" }));
    await screen.findByRole("link", { name: /^Refreshed workflow/ });
    expect(requests[previousAttempts]).toBe(1);
  });

  it("finishes pagination when Strict Mode replays a cached consumer mount", async () => {
    const requests: number[] = [];
    server.use(
      http.get(libraryURL, ({ request }) => {
        requests.push(Number(new URL(request.url).searchParams.get("page")));
        return HttpResponse.json(page([lateAgent], 2));
      }),
    );
    vi.spyOn(BackendAPI.prototype, "connectWebSocket").mockResolvedValue();
    server.use(getGetV1ListAllExecutionsMockHandler200([execution()]));
    const cachedClient = new QueryClient({
      defaultOptions: { queries: { retry: false, gcTime: 0 } },
    });
    cachedClient.setQueryData(libraryKey, {
      pages: [
        { status: 200, data: page(firstPageAgents), headers: new Headers() },
      ],
      pageParams: [1],
    });
    render(
      <StrictMode>
        <QueryClientProvider client={cachedClient}>
          <AccountMenuActivityRow />
        </QueryClientProvider>
      </StrictMode>,
    );
    fireEvent.click(screen.getByTestId("account-menu-activity-trigger"));
    await screen.findByRole("link", { name: /Late-page workflow/ });
    await waitFor(() =>
      expect(cachedClient.isFetching({ queryKey: libraryKey })).toBe(0),
    );
    await settleRequests();
    expect(requests.every((current) => current === 2)).toBe(true);
    expect(requests.length).toBeLessThanOrEqual(2);
  });

  it("stops again when an explicit retry fails, retaining the original query key", async () => {
    const requests: number[] = [];
    server.use(
      http.get(libraryURL, ({ request }) => {
        const params = new URL(request.url).searchParams;
        expect(params.get("is_hidden")).toBe("false");
        expect(params.get("page_size")).toBe("100");
        const current = Number(params.get("page"));
        requests.push(current);
        return current === 1
          ? HttpResponse.json(page(firstPageAgents))
          : HttpResponse.json({ detail: "Forbidden" }, { status: 403 });
      }),
    );
    const { queryClient } = renderActivity();
    await screen.findByRole("button", { name: "Try Again" });
    expect(requests).toEqual([1, 2, 2, 2]);
    fireEvent.click(screen.getByRole("button", { name: "Try Again" }));
    await waitFor(() => expect(requests).toHaveLength(7));
    await waitFor(() =>
      expect(queryClient.isFetching({ queryKey: libraryKey })).toBe(0),
    );
    await settleRequests();
    expect(requests).toEqual([1, 2, 2, 2, 2, 2, 2]);
    expect(
      getGetV2ListLibraryAgentsInfiniteQueryOptions({
        page: 1,
        page_size: 100,
        is_hidden: false,
      }).queryKey,
    ).toEqual(libraryKey);
  });
});
