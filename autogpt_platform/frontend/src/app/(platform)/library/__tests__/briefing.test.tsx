import { afterEach, describe, expect, test, vi } from "vitest";
import { within } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import {
  getGetV2ListLibraryAgentsMockHandler,
  getGetV2ListLibraryAgentsResponseMock,
  getGetV2ListFavoriteLibraryAgentsMockHandler,
  getGetV2ListFavoriteLibraryAgentsResponseMock,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import {
  getGetV2ListLibraryFoldersMockHandler,
  getGetV2ListLibraryFoldersResponseMock,
} from "@/app/api/__generated__/endpoints/folders/folders.msw";
import {
  getGetV1ListAllExecutionsMockHandler,
  getGetV1UserCostSummaryMockHandler,
} from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import LibraryPage from "../page";

afterEach(() => {
  vi.useRealTimers();
  flags.paymentsOn = true;
});

// Payments on is a billed install; off is a self-hosted one without billing.
const flags = vi.hoisted(() => ({ paymentsOn: true }));

vi.mock("@/services/feature-flags/use-get-flag", async () => {
  const actual = await vi.importActual<
    typeof import("@/services/feature-flags/use-get-flag")
  >("@/services/feature-flags/use-get-flag");
  return {
    ...actual,
    useGetFlag: () => false,
  };
});

vi.mock("@/hooks/usePlatformPayment", () => ({
  usePlatformPayment: () => ({
    isPaymentEnabled: flags.paymentsOn,
    isResolved: true,
  }),
}));

function setupHandlers({
  totalCents,
  agents = [
    { ...getGetV2ListLibraryAgentsResponseMock().agents[0], graph_id: "g-1" },
  ],
}: {
  totalCents: number;
  agents?: LibraryAgent[];
}) {
  server.use(
    // Pages like the backend: honours `page` and `page_size`.
    getGetV2ListLibraryAgentsMockHandler(({ request }) => {
      const params = new URL(request.url).searchParams;
      const page = Number(params.get("page") ?? 1);
      const pageSize = Number(params.get("page_size") ?? 15);
      const start = (page - 1) * pageSize;
      return {
        agents: agents.slice(start, start + pageSize),
        pagination: {
          total_items: agents.length,
          total_pages: Math.ceil(agents.length / pageSize),
          current_page: page,
          page_size: pageSize,
        },
      };
    }),
    getGetV2ListFavoriteLibraryAgentsMockHandler({
      ...getGetV2ListFavoriteLibraryAgentsResponseMock(),
      agents: [],
      pagination: {
        total_items: 0,
        total_pages: 1,
        current_page: 1,
        page_size: 10,
      },
    }),
    getGetV2ListLibraryFoldersMockHandler(
      getGetV2ListLibraryFoldersResponseMock({
        folders: [],
        pagination: {
          total_items: 0,
          total_pages: 1,
          current_page: 1,
          page_size: 20,
        },
      }),
    ),
    getGetV1ListAllExecutionsMockHandler([]),
    getGetV1UserCostSummaryMockHandler({
      total_cents: totalCents,
      run_count: totalCents > 0 ? 3 : 0,
      billable_run_count: totalCents > 0 ? 3 : 0,
      failed_cost_cents: 0,
      by_agent:
        totalCents > 0
          ? [{ graph_id: "g-1", cost_cents: totalCents, run_count: 3 }]
          : [],
      top_runs: [],
      daily: [],
    }),
  );
}

describe("LibraryPage — AgentBriefingPanel 'Spent this month' tile", () => {
  test("shows monthly spend from the cost-summary endpoint", async () => {
    setupHandlers({ totalCents: 825 });

    render(<LibraryPage />);

    const tile = (await screen.findByText("Spent this month")).closest(
      "button",
    );
    if (!tile) {
      throw new Error("Spent this month tile should render inside a button");
    }
    expect(within(tile).getByText("$8.25")).toBeDefined();
  });

  test("renders $0.00 when the endpoint reports zero spend", async () => {
    setupHandlers({ totalCents: 0 });

    render(<LibraryPage />);

    const tile = (await screen.findByText("Spent this month")).closest(
      "button",
    );
    if (!tile) {
      throw new Error("Spent this month tile should render inside a button");
    }
    expect(within(tile).getByText("$0.00")).toBeDefined();
  });
});

describe("LibraryPage — AgentBriefingPanel without billing (self-hosted)", () => {
  test("shows no spend, no costs breakdown, and never asks for spend", async () => {
    flags.paymentsOn = false;
    setupHandlers({ totalCents: 825 });
    let costSummaryRequests = 0;
    server.use(
      http.get("/api/proxy/api/executions/cost-summary", () => {
        costSummaryRequests += 1;
        return HttpResponse.json({}, { status: 500 });
      }),
    );

    render(<LibraryPage />);

    const runningTile = (await screen.findByText("Running now")).closest(
      "button",
    );
    expect(runningTile).not.toBeNull();
    expect(screen.queryByText("Spent this month")).toBeNull();
    expect(screen.queryByText(/costs breakdown/i)).toBeNull();
    expect(screen.queryByText("$8.25")).toBeNull();
    expect(costSummaryRequests).toBe(0);
  });

  test("a second click on the active tile goes back to the overview", async () => {
    flags.paymentsOn = false;
    setupHandlers({ totalCents: 0 });

    render(<LibraryPage />);

    const idle = await screen.findByRole("button", { name: /^Idle\b/ });
    fireEvent.click(idle);
    expect(idle.getAttribute("aria-pressed")).toBe("true");
    fireEvent.click(idle);
    expect(idle.getAttribute("aria-pressed")).toBe("false");
    expect(
      screen
        .getByRole("button", { name: /^Running now\b/ })
        .getAttribute("aria-pressed"),
    ).toBe("false");
  });
});

describe("LibraryPage — AgentBriefingPanel status tiles", () => {
  test("counts every library agent, not only the first page of 100", async () => {
    const template = getGetV2ListLibraryAgentsResponseMock().agents[0];
    // All idle (no trigger, no schedule, no executions), so the Idle tile
    // alone has to account for every agent across both pages.
    const agents = Array.from({ length: 150 }, (_, i) => ({
      ...template,
      id: `la-${i}`,
      graph_id: `g-${i}`,
      has_external_trigger: false,
      is_scheduled: false,
    }));
    setupHandlers({ totalCents: 0, agents });

    render(<LibraryPage />);

    const idle = await screen.findByRole("button", { name: /^Idle\b/ });
    await waitFor(() => expect(within(idle).getByText("150")).toBeDefined());
  });
});
