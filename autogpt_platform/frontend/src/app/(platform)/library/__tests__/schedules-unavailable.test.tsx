import { act, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient } from "@tanstack/react-query";
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

function serveLibrary(
  agents: () => LibraryAgent[],
  favorites: () => LibraryAgent[] = () => [],
) {
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
    getGetV2ListFavoriteLibraryAgentsMockHandler(() => ({
      ...getGetV2ListFavoriteLibraryAgentsResponseMock(),
      agents: favorites(),
      pagination: {
        total_items: favorites().length,
        total_pages: 1,
        current_page: 1,
        page_size: 10,
      },
    })),
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

function favoritesTab() {
  return screen
    .getAllByRole("tab")
    .find((tab) => tab.textContent?.match(/favorites/i));
}

function testClient() {
  return new QueryClient({ defaultOptions: { queries: { retry: false } } });
}

// Every library read the page started has been answered and handled.
async function libraryReadsSettled(queryClient: QueryClient) {
  await waitFor(() =>
    expect(
      queryClient.isFetching({
        predicate: (query) =>
          String(query.queryKey[0]).startsWith("/api/library/agents"),
      }),
    ).toBe(0),
  );
}

// Moves the clock on, then waits out any read it started. A read the clock
// starts is in flight as soon as the timer fires, so none is missed.
async function advanceAndSettle(queryClient: QueryClient, ms: number) {
  await act(async () => {
    vi.advanceTimersByTime(ms);
  });
  await libraryReadsSettled(queryClient);
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
    const queryClient = testClient();

    render(<LibraryPage />, { queryClient });
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

    // Known again, so the rechecks stop.
    await libraryReadsSettled(queryClient);
    const recovered = reads;
    await act(async () => {
      vi.advanceTimersByTime(30_000);
    });
    await libraryReadsSettled(queryClient);
    expect(reads).toBe(recovered);
  });

  test("the Favorites tab rechecks on its own too", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    let favorites = duringOutage;
    serveLibrary(
      () => afterRecovery,
      () => favorites,
    );

    render(<LibraryPage />);
    await waitFor(() =>
      expect(favoritesTab()?.hasAttribute("data-disabled")).toBe(false),
    );
    await userEvent.click(favoritesTab()!);
    expect(await screen.findByText(NOTICE)).toBeDefined();

    favorites = afterRecovery;
    await act(async () => {
      vi.advanceTimersByTime(15_000);
    });

    await waitFor(() => expect(screen.queryByText(NOTICE)).toBeNull());
    expect(
      within(await cardFor("Morning digest")).getByText("Scheduled"),
    ).toBeDefined();
  });

  test("an outage that outlasts a deploy rechecks once a minute", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    let reads = 0;
    serveLibrary(() => {
      reads += 1;
      return duringOutage;
    });
    const queryClient = testClient();

    render(<LibraryPage />, { queryClient });
    expect(await screen.findByText(NOTICE)).toBeDefined();
    await libraryReadsSettled(queryClient);

    // Two minutes in, still within a deploy's gap, it rechecks after every
    // Retry-After.
    await advanceAndSettle(queryClient, 120_000);
    let before = reads;
    await advanceAndSettle(queryClient, 15_000);
    expect(reads).toBeGreaterThan(before);

    // Well past the gap, each list re-reads every loaded page once a minute:
    // not within 45 s of its last read, and again within 75 s.
    await advanceAndSettle(queryClient, 60_000);
    before = reads;
    await advanceAndSettle(queryClient, 45_000);
    expect(reads).toBe(before);
    await advanceAndSettle(queryClient, 30_000);
    expect(reads).toBeGreaterThan(before);
    expect(screen.getByText(NOTICE)).toBeDefined();
  });

  test("a library whose schedules are known shows no notice and never rechecks", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    let reads = 0;
    serveLibrary(() => {
      reads += 1;
      return afterRecovery;
    });
    const queryClient = testClient();

    render(<LibraryPage />, { queryClient });
    expect(
      within(await cardFor("Morning digest")).getByText("Scheduled"),
    ).toBeDefined();
    await waitFor(() =>
      expect(within(tile("Scheduled")).getByText("1")).toBeDefined(),
    );
    // Let the page's first reads finish before counting.
    await libraryReadsSettled(queryClient);
    const settled = reads;

    await act(async () => {
      vi.advanceTimersByTime(60_000);
    });
    await libraryReadsSettled(queryClient);

    expect(reads).toBe(settled);
    expect(screen.queryByText(NOTICE)).toBeNull();
  });
});
