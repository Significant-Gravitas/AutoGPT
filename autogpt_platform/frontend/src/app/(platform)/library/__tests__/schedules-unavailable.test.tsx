import { act, within } from "@testing-library/react";
import { afterEach, describe, expect, test, vi } from "vitest";
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
import { getGetV1ListAllExecutionsMockHandler } from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import LibraryPage from "../page";

vi.mock("@/services/feature-flags/use-get-flag", async () => {
  const actual = await vi.importActual<
    typeof import("@/services/feature-flags/use-get-flag")
  >("@/services/feature-flags/use-get-flag");
  return { ...actual, useGetFlag: () => false };
});

// Billing off keeps the briefing to its status tiles.
vi.mock("@/hooks/usePlatformPayment", () => ({
  usePlatformPayment: () => ({ isPaymentEnabled: false, isResolved: true }),
}));

afterEach(() => {
  vi.useRealTimers();
});

const NOTICE =
  "Schedules are temporarily unavailable — your schedules are safe and will run once it's back.";

function makeAgent(overrides: Partial<LibraryAgent>): LibraryAgent {
  return {
    ...getGetV2ListLibraryAgentsResponseMock().agents[0],
    has_external_trigger: false,
    is_scheduled: false,
    next_scheduled_run: null,
    schedule_info_unavailable: false,
    ...overrides,
  };
}

const digest = makeAgent({
  id: "la-digest",
  graph_id: "g-digest",
  name: "Morning digest",
});
const triage = makeAgent({
  id: "la-triage",
  graph_id: "g-triage",
  name: "Inbox triage",
});

// What the library API answers while the scheduler is out: every workflow
// loads, and its schedule is unknown rather than absent.
const duringOutage: LibraryAgent[] = [digest, triage].map((agent) => ({
  ...agent,
  schedule_info_unavailable: true,
}));

const afterRecovery: LibraryAgent[] = [
  {
    ...digest,
    is_scheduled: true,
    next_scheduled_run: new Date(Date.now() + 2 * 60 * 60 * 1000).toISOString(),
  },
  triage,
];

function serveLibrary(agents: () => LibraryAgent[]) {
  server.use(
    getGetV2ListLibraryAgentsMockHandler(() => ({
      agents: agents(),
      pagination: {
        total_items: agents().length,
        total_pages: 1,
        current_page: 1,
        page_size: 100,
      },
    })),
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

function tile(name: "Scheduled" | "Idle") {
  return screen.getByRole("button", { name: new RegExp(`^${name}\\b`) });
}

async function cardFor(name: string) {
  const cards = await screen.findAllByTestId("library-agent-card");
  const card = cards.find((c) => within(c).queryByText(name));
  if (!card) throw new Error(`No card for ${name}`);
  return card;
}

describe("LibraryPage while the scheduler is out", () => {
  test("says schedules are unavailable instead of showing none, and recovers on Try again", async () => {
    let agents = duringOutage;
    serveLibrary(() => agents);

    render(<LibraryPage />);

    // The rest of the library still loads.
    expect(await cardFor("Morning digest")).toBeDefined();
    expect(await cardFor("Inbox triage")).toBeDefined();
    expect(await screen.findByText(NOTICE)).toBeDefined();

    // Neither count is known: a scheduled workflow can't be told from an
    // idle one, so neither tile shows a number.
    await waitFor(() =>
      expect(within(tile("Scheduled")).getByText("—")).toBeDefined(),
    );
    expect(within(tile("Idle")).getByText("—")).toBeDefined();

    // No card calls a workflow idle when its schedule is unknown.
    expect(
      within(await cardFor("Morning digest")).queryByText("Idle"),
    ).toBeNull();

    // The Scheduled tab says why it can't list them, not that there are none.
    fireEvent.click(tile("Scheduled"));
    await waitFor(() => expect(screen.getAllByText(NOTICE)).toHaveLength(2));
    expect(screen.queryByText("No workflows with scheduled tasks")).toBeNull();

    agents = afterRecovery;
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));

    await waitFor(() => expect(screen.queryByText(NOTICE)).toBeNull());
    await waitFor(() =>
      expect(within(tile("Scheduled")).getByText("1")).toBeDefined(),
    );
    expect(
      within(await cardFor("Morning digest")).getByText("Scheduled"),
    ).toBeDefined();
    expect(
      within(await cardFor("Inbox triage")).getByText("Idle"),
    ).toBeDefined();
  });

  test("rechecks on its own once the scheduler's Retry-After has passed", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    let agents = duringOutage;
    let reads = 0;
    serveLibrary(() => {
      reads += 1;
      return agents;
    });

    render(<LibraryPage />);
    expect(await screen.findByText(NOTICE)).toBeDefined();

    agents = afterRecovery;
    const before = reads;
    await act(async () => {
      vi.advanceTimersByTime(15_000);
    });

    await waitFor(() => expect(reads).toBeGreaterThan(before));
    await waitFor(() => expect(screen.queryByText(NOTICE)).toBeNull());
    expect(
      within(await cardFor("Morning digest")).getByText("Scheduled"),
    ).toBeDefined();
    // The briefing reads the whole library on its own, and rechecks too.
    await waitFor(() =>
      expect(within(tile("Scheduled")).getByText("1")).toBeDefined(),
    );
  });

  test("a library whose schedules are known shows no notice and never rechecks", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    let reads = 0;
    serveLibrary(() => {
      reads += 1;
      return afterRecovery;
    });

    render(<LibraryPage />);
    expect(
      within(await cardFor("Morning digest")).getByText("Scheduled"),
    ).toBeDefined();
    await waitFor(() =>
      expect(within(tile("Scheduled")).getByText("1")).toBeDefined(),
    );
    // Let the page's first reads finish before counting.
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 50));
    });
    const settled = reads;

    await act(async () => {
      vi.advanceTimersByTime(60_000);
    });
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 50));
    });

    expect(reads).toBe(settled);
    expect(screen.queryByText(NOTICE)).toBeNull();
  });
});
