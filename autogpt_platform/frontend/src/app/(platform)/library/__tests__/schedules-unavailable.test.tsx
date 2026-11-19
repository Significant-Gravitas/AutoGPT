import { act, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient } from "@tanstack/react-query";
import { http, HttpResponse } from "msw";
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
import { getGetV1ListExecutionSchedulesForAUserQueryKey } from "@/app/api/__generated__/endpoints/schedules/schedules";
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
  // Whether the scheduler answers its schedule list, and a hook counting
  // each read of it. Out by default: that is what an outage is.
  schedulesUp: () => boolean = () => false,
  onProbeRead: () => void = () => {},
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
    // The scheduler's own answer: a 503 with Retry-After while it is out,
    // the (empty) schedule list once it is back.
    http.get("/api/proxy/api/schedules", () => {
      onProbeRead();
      if (schedulesUp()) return HttpResponse.json([]);
      return HttpResponse.json(
        { detail: "Scheduler unavailable" },
        { status: 503, headers: { "Retry-After": "15" } },
      );
    }),
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

// Every library read the page started has been answered and handled —
// including the scheduler probe, a separate schedules query the library
// reads do not cover.
async function libraryReadsSettled(queryClient: QueryClient) {
  await waitFor(() =>
    expect(
      queryClient.isFetching({
        predicate: (query) =>
          String(query.queryKey[0]).startsWith("/api/library/agents"),
      }),
    ).toBe(0),
  );
  await waitFor(() =>
    expect(
      queryClient.isFetching({
        queryKey: getGetV1ListExecutionSchedulesForAUserQueryKey(),
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

// Advances the clock in Retry-After steps, letting each probe read finish
// before the next interval tick: a single long jump leaves the first read
// in flight, so later ticks are deduped and never read.
async function advanceByIntervals(queryClient: QueryClient, ms: number) {
  for (let remaining = ms; remaining > 0; remaining -= 15_000) {
    await advanceAndSettle(queryClient, Math.min(15_000, remaining));
  }
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
    let schedulerUp = false;
    let reads = 0;
    let probeReads = 0;
    serveLibrary(
      () => {
        reads += 1;
        return agents;
      },
      undefined,
      () => schedulerUp,
      () => {
        probeReads += 1;
      },
    );
    const queryClient = testClient();

    render(<LibraryPage />, { queryClient });
    expect(await screen.findByText(NOTICE)).toBeDefined();

    // While the scheduler is still out, the probe reads on the interval but
    // the lists do not re-read: one cheap read instead of every loaded page.
    await libraryReadsSettled(queryClient);
    const settledReads = reads;
    const settledProbes = probeReads;
    await act(async () => {
      vi.advanceTimersByTime(15_000);
    });
    await waitFor(() => expect(probeReads).toBeGreaterThan(settledProbes));
    expect(reads).toBe(settledReads);

    // The scheduler is back, so the probe answers and the lists re-read once.
    agents = afterRecovery;
    schedulerUp = true;
    await act(async () => {
      vi.advanceTimersByTime(15_000);
    });

    await waitFor(() => expect(reads).toBeGreaterThan(settledReads));
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
    const recoveredProbes = probeReads;
    await act(async () => {
      vi.advanceTimersByTime(30_000);
    });
    await libraryReadsSettled(queryClient);
    expect(reads).toBe(recovered);
    expect(probeReads).toBe(recoveredProbes);
  });

  test("the Favorites tab rechecks on its own too", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    let favorites = duringOutage;
    let schedulerUp = false;
    serveLibrary(
      () => afterRecovery,
      () => favorites,
      () => schedulerUp,
    );

    render(<LibraryPage />);
    await waitFor(() =>
      expect(favoritesTab()?.hasAttribute("data-disabled")).toBe(false),
    );
    await userEvent.click(favoritesTab()!);
    expect(await screen.findByText(NOTICE)).toBeDefined();

    favorites = afterRecovery;
    schedulerUp = true;
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
    let probeReads = 0;
    serveLibrary(
      () => {
        reads += 1;
        return duringOutage;
      },
      undefined,
      () => false,
      () => {
        probeReads += 1;
      },
    );
    const queryClient = testClient();

    render(<LibraryPage />, { queryClient });
    expect(await screen.findByText(NOTICE)).toBeDefined();
    await libraryReadsSettled(queryClient);

    // Within a deploy's gap the probe reads after every Retry-After, while
    // the lists stay put.
    const settledProbes = probeReads;
    const settledReads = reads;
    await advanceByIntervals(queryClient, 60_000);
    const gapReads = probeReads - settledProbes;
    expect(gapReads).toBeGreaterThan(0);
    expect(reads).toBe(settledReads);

    // Well past the gap, the probe reads once a minute: fewer reads over
    // the same span, and the lists still stay put.
    await advanceByIntervals(queryClient, 120_000);
    const before = probeReads;
    const listBefore = reads;
    await advanceByIntervals(queryClient, 60_000);
    const longOutageReads = probeReads - before;
    expect(longOutageReads).toBeGreaterThan(0);
    expect(longOutageReads).toBeLessThan(gapReads);
    expect(reads).toBe(listBefore);
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
