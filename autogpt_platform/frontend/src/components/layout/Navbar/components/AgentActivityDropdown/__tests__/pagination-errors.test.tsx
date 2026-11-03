import { server } from "@/mocks/mock-server";
import { fireEvent, screen, waitFor } from "@/tests/integrations/test-utils";
import { delay, http, HttpResponse } from "msw";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  deferred,
  firstPageAgents,
  lateAgent,
  libraryKey,
  libraryURL,
  page,
  renderActivity,
  settleRequests,
} from "./fixtures";

afterEach(() => vi.restoreAllMocks());

describe("Activity pagination recovery", () => {
  it.each([1, 2])(
    "explicitly retries a failed page %s and enriches existing activity",
    async (failedPage) => {
      let unavailable = true;
      const requests: number[] = [];
      server.use(
        http.get(libraryURL, ({ request }) => {
          const current = Number(new URL(request.url).searchParams.get("page"));
          requests.push(current);
          if (unavailable && current === failedPage)
            return HttpResponse.json(
              { detail: "Unavailable" },
              { status: 503 },
            );
          return HttpResponse.json(
            current === 1 ? page(firstPageAgents) : page([lateAgent], 2),
          );
        }),
      );
      const { queryClient } = renderActivity();
      await screen.findByRole("button", { name: "Try Again" });
      await screen.findByText("Workflow details could not be loaded.");
      await screen.findByText("Workflow abcdef01");
      const attempts = requests.length;
      await settleRequests();
      expect(requests.length).toBe(attempts);
      unavailable = false;
      const retryButton = screen.getByRole("button", { name: "Try Again" });
      fireEvent.click(retryButton);
      fireEvent.click(retryButton);
      await screen.findByRole("link", { name: /Late-page workflow/ });
      expect(requests.filter((value) => value === failedPage)).toHaveLength(4);
      expect(queryClient.getQueryState(libraryKey)?.status).toBe("success");
      expect(screen.queryByRole("button", { name: "Try Again" })).toBeNull();
    },
  );

  it.each([
    "repeated page",
    "duplicate records",
    "empty page",
    "growing total",
    "invalid pagination",
  ])("stops on %s without endless fetching", async (scenario) => {
    const requests: number[] = [];
    server.use(
      http.get(libraryURL, async ({ request }) => {
        const current = Number(new URL(request.url).searchParams.get("page"));
        requests.push(current);
        await delay(5);
        if (current === 1)
          return HttpResponse.json(
            page(firstPageAgents, 1, scenario === "growing total" ? 101 : 301),
          );
        const response = page(
          scenario === "empty page"
            ? []
            : scenario === "duplicate records"
              ? firstPageAgents
              : [lateAgent],
          scenario === "repeated page" ? 1 : current,
          scenario === "growing total" ? current * 100 + 1 : 301,
        );
        if (scenario === "invalid pagination")
          response.pagination.page_size = 0;
        return HttpResponse.json(response);
      }),
    );
    const { queryClient } = renderActivity();
    await waitFor(() => expect(requests.length).toBeGreaterThanOrEqual(2));
    await waitFor(() =>
      expect(queryClient.isFetching({ queryKey: libraryKey })).toBe(0),
    );
    await settleRequests();
    expect(requests).toEqual(
      scenario === "duplicate records" ? [1, 2, 3, 4] : [1, 2],
    );
  });

  it.each(["cancel", "unmount"])(
    "does not restart pagination after %s",
    async (action) => {
      const next = deferred();
      const requests: number[] = [];
      let aborted = false;
      server.use(
        http.get(libraryURL, async ({ request }) => {
          const current = Number(new URL(request.url).searchParams.get("page"));
          requests.push(current);
          if (current === 1)
            return HttpResponse.json(page(firstPageAgents, 1, 201));
          request.signal.addEventListener("abort", () => {
            aborted = true;
          });
          await next.promise;
          return HttpResponse.json(page([lateAgent], current, 201));
        }),
      );
      const { queryClient, unmount } = renderActivity();
      await waitFor(() => expect(requests).toEqual([1, 2]));
      if (action === "unmount") unmount();
      else await queryClient.cancelQueries({ queryKey: libraryKey });
      await waitFor(() => expect(aborted).toBe(true));
      next.resolve();
      await settleRequests();
      expect(requests).toEqual([1, 2]);
      expect(queryClient.isFetching({ queryKey: libraryKey })).toBe(0);
    },
  );
});
