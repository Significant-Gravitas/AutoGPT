// Force a non-UTC timezone so the UTC-month boundary logic in
// `startOfCurrentMonth` can be distinguished from a hypothetical local-month
// implementation. Node reads TZ on each Date operation on Linux (CI), so this
// takes effect before any `new Date(...)` below.
process.env.TZ = "America/Los_Angeles";

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
import { getGetV1ListAllExecutionsMockHandler } from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import { Flag } from "@/services/feature-flags/use-get-flag";
import LibraryPage from "../page";

// Defensive teardown: if a test fails before its final `vi.useRealTimers()`,
// later tests would inherit fake timers. Global `cleanup()` is handled in
// `src/tests/integrations/vitest.setup.tsx`.
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
        // April 1 in UTC, but March 31 22:00 in America/Los_Angeles.
        // A buggy local-month implementation would exclude this execution
        // under TZ=America/Los_Angeles; the correct UTC-month logic includes
        // it. Contributes 500 cents to the expected $8.25 total.
        id: "utc-vs-local-boundary",
        user_id: "test-user",
        graph_id: "g-1",
        graph_version: 1,
        inputs: {},
        credential_inputs: {},
        nodes_input_masks: {},
        preset_id: null,
        status: "COMPLETED",
        started_at: new Date("2026-04-01T05:00:00.000Z"),
        ended_at: new Date("2026-04-01T05:02:00.000Z"),
        stats: { cost: 500 },
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
    if (!tile) {
      throw new Error("Spent this month tile should render inside a button");
    }
    // 250 + 75 + 500 = 825 cents = $8.25. The late-March UTC execution must
    // NOT contribute, and the April-1-UTC/March-31-local execution MUST
    // contribute (confirms UTC-month boundary under a non-UTC test zone).
    expect(within(tile).getByText("$8.25")).toBeDefined();

    vi.useRealTimers();
  });

  test("renders $0.00 when no executions ran this month", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    vi.setSystemTime(new Date("2026-04-15T12:00:00.000Z"));

    setupHandlers([]);

    render(<LibraryPage />);

    const tile = (await screen.findByText("Spent this month")).closest(
      "button",
    );
    if (!tile) {
      throw new Error("Spent this month tile should render inside a button");
    }
    expect(within(tile).getByText("$0.00")).toBeDefined();

    vi.useRealTimers();
  });
});
