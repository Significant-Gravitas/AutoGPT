import { describe, expect, test, vi } from "vitest";
import { LibraryAgentSort } from "@/app/api/__generated__/models/libraryAgentSort";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
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
import { getGetV1ListAllExecutionsMockHandler } from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import { Flag } from "@/services/feature-flags/use-get-flag";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { LibraryAgentList } from "../components/LibraryAgentList/LibraryAgentList";
import { FavoriteAnimationProvider } from "../context/FavoriteAnimationContext";

vi.mock("@/services/feature-flags/use-get-flag", async () => {
  const actual = await vi.importActual<
    typeof import("@/services/feature-flags/use-get-flag")
  >("@/services/feature-flags/use-get-flag");
  return {
    ...actual,
    useGetFlag: (flag: Flag) => flag === "agent-briefing",
  };
});

function makeAgent(overrides: Partial<LibraryAgent>): LibraryAgent {
  const base = getGetV2ListLibraryAgentsResponseMock().agents[0];
  return {
    ...base,
    has_external_trigger: false,
    is_scheduled: false,
    recommended_schedule_cron: null,
    next_scheduled_run: null,
    ...overrides,
  };
}

function setupHandlers(agents: LibraryAgent[]) {
  server.use(
    getGetV2ListLibraryAgentsMockHandler({
      ...getGetV2ListLibraryAgentsResponseMock(),
      agents,
      pagination: {
        total_items: agents.length,
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
  );
}

function renderList(statusFilter: "all" | "scheduled" | "idle") {
  return render(
    <FavoriteAnimationProvider onAnimationComplete={() => undefined}>
      <LibraryAgentList
        searchTerm=""
        librarySort={LibraryAgentSort.createdAt}
        setLibrarySort={() => undefined}
        selectedFolderId={null}
        onFolderSelect={() => undefined}
        tabs={[]}
        activeTab="all"
        onTabChange={() => undefined}
        statusFilter={statusFilter}
        onStatusFilterChange={() => undefined}
      />
    </FavoriteAnimationProvider>,
  );
}

describe("LibraryAgentList — Scheduled status filter", () => {
  test("includes agents with is_scheduled=true even when recommended_schedule_cron is null", async () => {
    setupHandlers([
      makeAgent({
        id: "a-actually-scheduled",
        graph_id: "g-actually-scheduled",
        name: "Actually Scheduled Agent",
        is_scheduled: true,
        recommended_schedule_cron: null,
      }),
      makeAgent({
        id: "a-idle",
        graph_id: "g-idle",
        name: "Idle Agent",
      }),
    ]);

    renderList("scheduled");

    expect(await screen.findByText("Actually Scheduled Agent")).toBeDefined();
    await waitFor(() => {
      expect(screen.queryByText("Idle Agent")).toBeNull();
    });
  });

  test("excludes agents that only have recommended_schedule_cron (not actually scheduled by the user) from the scheduled filter", async () => {
    setupHandlers([
      makeAgent({
        id: "a-recommendation-only",
        graph_id: "g-recommendation-only",
        name: "Recommendation Only Agent",
        is_scheduled: false,
        recommended_schedule_cron: "0 9 * * *",
      }),
    ]);

    renderList("scheduled");

    await waitFor(() => {
      expect(screen.queryByText("Recommendation Only Agent")).toBeNull();
    });
  });

  test("includes agents that only have recommended_schedule_cron in the idle filter", async () => {
    setupHandlers([
      makeAgent({
        id: "a-recommendation-only",
        graph_id: "g-recommendation-only",
        name: "Recommendation Only Agent",
        is_scheduled: false,
        recommended_schedule_cron: "0 9 * * *",
      }),
    ]);

    renderList("idle");

    expect(await screen.findByText("Recommendation Only Agent")).toBeDefined();
  });

  test("excludes idle agents (no schedule, no recommendation) from the scheduled filter", async () => {
    setupHandlers([
      makeAgent({
        id: "a-idle",
        graph_id: "g-idle",
        name: "Idle Agent",
      }),
    ]);

    renderList("scheduled");

    // List should render (no loading spinner stuck) — wait briefly for the
    // empty filtered state to settle.
    await waitFor(() => {
      expect(screen.queryByText("Idle Agent")).toBeNull();
    });
  });

  test("excludes agents with is_scheduled=true from the idle filter", async () => {
    setupHandlers([
      makeAgent({
        id: "a-actually-scheduled",
        graph_id: "g-actually-scheduled",
        name: "Actually Scheduled Agent",
        is_scheduled: true,
        recommended_schedule_cron: null,
      }),
      makeAgent({
        id: "a-idle",
        graph_id: "g-idle",
        name: "Idle Agent",
      }),
    ]);

    renderList("idle");

    expect(await screen.findByText("Idle Agent")).toBeDefined();
    await waitFor(() => {
      expect(screen.queryByText("Actually Scheduled Agent")).toBeNull();
    });
  });
});
