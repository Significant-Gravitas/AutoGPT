import { server } from "@/mocks/mock-server";
import { screen, waitFor } from "@/tests/integrations/test-utils";
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

describe("Activity workflow pagination", () => {
  it("enriches existing activity and its link when a later library page arrives", async () => {
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
    await screen.findByText(/^(Unknown Workflow|Workflow abcdef01)$/);
    expect(
      screen.queryByRole("link", { name: /Late-page workflow/ }),
    ).toBeNull();
    await waitFor(() => expect(requests).toEqual([1, 2]));
    next.resolve();
    const link = await screen.findByRole("link", {
      name: /Late-page workflow/,
    });
    expect(link.getAttribute("href")).toBe(
      `/library/agents/${lateAgent.id}?activeTab=runs&activeItem=execution-1`,
    );
    await waitFor(() =>
      expect(queryClient.isFetching({ queryKey: libraryKey })).toBe(0),
    );
    await settleRequests();
    expect(requests).toEqual([1, 2]);
  });

  it("stops automatically fetching a failed page after the query retry budget", async () => {
    const requests: number[] = [];
    server.use(
      http.get(libraryURL, async ({ request }) => {
        const current = Number(new URL(request.url).searchParams.get("page"));
        requests.push(current);
        if (current === 1) return HttpResponse.json(page(firstPageAgents));
        await delay(10);
        return HttpResponse.json({ detail: "Unavailable" }, { status: 503 });
      }),
    );
    const { queryClient } = renderActivity();
    await waitFor(() =>
      expect(queryClient.getQueryState(libraryKey)?.error).toBeTruthy(),
    );
    const attempts = requests.length;
    await settleRequests();
    expect(requests.length).toBe(attempts);
    expect(attempts).toBeLessThanOrEqual(4);
  });
});
