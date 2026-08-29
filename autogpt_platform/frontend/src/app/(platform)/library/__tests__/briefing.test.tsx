import { afterEach, describe, expect, test, vi } from "vitest";
import { within } from "@testing-library/react";
import { render, screen } from "@/tests/integrations/test-utils";
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
import { Flag } from "@/services/feature-flags/use-get-flag";
import LibraryPage from "../page";

afterEach(() => {
  vi.useRealTimers();
});

vi.mock("@/services/feature-flags/use-get-flag", async () => {
  const actual = await vi.importActual<
    typeof import("@/services/feature-flags/use-get-flag")
  >("@/services/feature-flags/use-get-flag");
  return {
    ...actual,
    useGetFlag: (flag: Flag) => flag === "agent-briefing",
  };
});

function setupHandlers({ totalCents }: { totalCents: number }) {
  const agents = [
    { ...getGetV2ListLibraryAgentsResponseMock().agents[0], graph_id: "g-1" },
  ];
  server.use(
    getGetV2ListLibraryAgentsMockHandler({
      ...getGetV2ListLibraryAgentsResponseMock(),
      agents,
      pagination: {
        total_items: 1,
        total_pages: 1,
        current_page: 1,
        page_size: 20,
      },
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
