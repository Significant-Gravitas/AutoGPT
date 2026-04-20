import { describe, expect, test, vi } from "vitest";
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
import { getGetV1ListAllExecutionsMockHandler } from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import { Flag } from "@/services/feature-flags/use-get-flag";
import LibraryPage from "../page";

vi.mock("@/services/feature-flags/use-get-flag", async () => {
  const actual = await vi.importActual<
    typeof import("@/services/feature-flags/use-get-flag")
  >("@/services/feature-flags/use-get-flag");
  return {
    ...actual,
    useGetFlag: (flag: Flag) => flag === "agent-briefing",
  };
});

function setupHandlers(
  executions: Parameters<typeof getGetV1ListAllExecutionsMockHandler>[0],
) {
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
    getGetV1ListAllExecutionsMockHandler(executions),
  );
}

describe("LibraryPage — AgentBriefingPanel 'Spent this month' tile", () => {
  test("sums execution costs from the current UTC month and formats as currency", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    vi.setSystemTime(new Date("2026-04-15T12:00:00.000Z"));

    setupHandlers([
      {
        id: "this-month-a",
        user_id: "test-user",
        graph_id: "g-1",
        graph_version: 1,
        inputs: {},
        credential_inputs: {},
        nodes_input_masks: {},
        preset_id: null,
        status: "COMPLETED",
        started_at: new Date("2026-04-02T10:00:00.000Z"),
        ended_at: new Date("2026-04-02T10:05:00.000Z"),
        stats: { cost: 250 },
      },
      {
        id: "this-month-b",
        user_id: "test-user",
        graph_id: "g-1",
        graph_version: 1,
        inputs: {},
        credential_inputs: {},
        nodes_input_masks: {},
        preset_id: null,
        status: "COMPLETED",
        started_at: new Date("2026-04-10T10:00:00.000Z"),
        ended_at: new Date("2026-04-10T10:02:00.000Z"),
        stats: { cost: 75 },
      },
      {
        id: "previous-month",
        user_id: "test-user",
        graph_id: "g-1",
        graph_version: 1,
        inputs: {},
        credential_inputs: {},
        nodes_input_masks: {},
        preset_id: null,
        status: "COMPLETED",
        started_at: new Date("2026-03-31T23:59:00.000Z"),
        ended_at: new Date("2026-04-01T00:00:00.000Z"),
        stats: { cost: 9999 },
      },
    ]);

    render(<LibraryPage />);

    const tile = (await screen.findByText("Spent this month")).closest(
      "button",
    );
    expect(tile).toBeDefined();
    // 250 + 75 = 325 cents = $3.25. The late-March UTC execution must NOT
    // contribute (confirms the UTC-vs-local-time boundary fix).
    expect(await screen.findByText("$3.25")).toBeDefined();

    vi.useRealTimers();
  });

  test("renders $0.00 when no executions ran this month", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    vi.setSystemTime(new Date("2026-04-15T12:00:00.000Z"));

    setupHandlers([]);

    render(<LibraryPage />);

    await screen.findByText("Spent this month");
    expect(await screen.findByText("$0.00")).toBeDefined();

    vi.useRealTimers();
  });
});
