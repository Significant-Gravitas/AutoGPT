import { act, within } from "@testing-library/react";
import { QueryClient } from "@tanstack/react-query";
import { http, HttpResponse } from "msw";
import { afterEach, expect, test, vi } from "vitest";
import { getGetHomeDashboardQueryKey } from "@/app/api/__generated__/endpoints/home/home";
import { getGetHomeDashboardMockHandler } from "@/app/api/__generated__/endpoints/home/home.msw";
import type { HomeDashboardResponse } from "@/app/api/__generated__/models/homeDashboardResponse";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { HomeRecap } from "../components/HomeRecap/HomeRecap";
import { ada, makeDashboard, NOW } from "./heldItems";

vi.mock("@/services/feature-flags/use-get-flag", async (importActual) => {
  const actual =
    await importActual<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useFlagStatus: () => ({ enabled: true, ready: true }),
    useGetFlag: () => false,
  };
});

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ user: { user_metadata: { preferred_name: "Sam" } } }),
}));

afterEach(() => {
  vi.useRealTimers();
});

const NOTICE =
  "Schedules are temporarily unavailable — your schedules are safe and will run once it's back.";

// Home's dashboard while the scheduler is out: every other tile loads, and the
// upcoming runs are unknown rather than absent.
const duringOutage: HomeDashboardResponse = {
  ...makeDashboard([]),
  active_tasks: [
    {
      id: "active-1",
      title: "Checking recurring subscriptions",
      status: "running",
      expert: ada,
      started_at: new Date(NOW.getTime() - 8 * 60_000),
      link: "/library",
    },
  ],
  team: { total: 1, ready: 0, working: 1, needs_attention: 0 },
  agents: [
    {
      expert: ada,
      status: "working",
      detail: "Checking recurring subscriptions",
    },
  ],
  upcoming_tasks: [],
  schedules_unavailable: true,
};

const afterRecovery: HomeDashboardResponse = {
  ...duringOutage,
  upcoming_tasks: [
    {
      id: "upcoming-1",
      title: "Spanish practice plan",
      kind: "agent",
      expert: ada,
      next_run_time: new Date(NOW.getTime() + 80 * 60_000),
    },
  ],
  schedules_unavailable: false,
};

function serveDashboard(current: () => HomeDashboardResponse) {
  let reads = 0;
  server.use(
    getGetHomeDashboardMockHandler(() => {
      reads += 1;
      return current();
    }),
  );
  return () => reads;
}

test("Now & next says schedules are unavailable while the rest of Home loads, and recovers on Try again", async () => {
  let dashboard = duringOutage;
  serveDashboard(() => dashboard);

  render(<HomeRecap />);

  const tile = await screen.findByRole("region", { name: "Now & next" });
  expect(await within(tile).findByText(NOTICE)).toBeDefined();
  // Unknown, not empty: no "nothing scheduled" and no page error.
  expect(within(tile).queryByText("Nothing is scheduled")).toBeNull();
  expect(
    screen.queryByText("Your Home briefing could not be loaded"),
  ).toBeNull();
  // The rest of Home still works.
  expect(
    within(tile).getByText("Checking recurring subscriptions"),
  ).toBeDefined();
  expect(screen.getByRole("heading", { name: "Your team" })).toBeDefined();
  expect(screen.getByRole("heading", { name: "Recent work" })).toBeDefined();

  dashboard = afterRecovery;
  fireEvent.click(within(tile).getByRole("button", { name: "Try again" }));

  expect(await within(tile).findByText("Spanish practice plan")).toBeDefined();
  await waitFor(() => expect(within(tile).queryByText(NOTICE)).toBeNull());
});

test("Home rechecks on its own once the scheduler's Retry-After has passed", async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  let dashboard = duringOutage;
  const reads = serveDashboard(() => dashboard);

  render(<HomeRecap />);
  const tile = await screen.findByRole("region", { name: "Now & next" });
  expect(await within(tile).findByText(NOTICE)).toBeDefined();

  dashboard = afterRecovery;
  const before = reads();
  // Sooner than Home's usual one-minute refresh.
  await act(async () => {
    vi.advanceTimersByTime(15_000);
  });

  await waitFor(() => expect(reads()).toBeGreaterThan(before));
  expect(await within(tile).findByText("Spanish practice plan")).toBeDefined();
  expect(within(tile).queryByText(NOTICE)).toBeNull();
});

test("an outage that outlasts a deploy rechecks Home only once a minute", async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  const reads = serveDashboard(() => duringOutage);
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });

  render(<HomeRecap />, { queryClient });
  const tile = await screen.findByRole("region", { name: "Now & next" });
  expect(await within(tile).findByText(NOTICE)).toBeDefined();

  // Moves the clock on, then waits out any read it started.
  async function advanceAndSettle(ms: number) {
    await act(async () => {
      vi.advanceTimersByTime(ms);
    });
    await waitFor(() =>
      expect(
        queryClient.isFetching({ queryKey: getGetHomeDashboardQueryKey() }),
      ).toBe(0),
    );
  }

  // Well past a deploy's gap it keeps Home's usual minute: not within 45 s of
  // its last read, and again within 75 s.
  await advanceAndSettle(195_000);
  const before = reads();
  await advanceAndSettle(45_000);
  expect(reads()).toBe(before);
  await advanceAndSettle(30_000);
  expect(reads()).toBeGreaterThan(before);
  expect(within(tile).getByText(NOTICE)).toBeDefined();
});

test("a recheck that fails keeps Home's tiles instead of the page error", async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  serveDashboard(() => duringOutage);
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });

  render(<HomeRecap />, { queryClient });
  const tile = await screen.findByRole("region", { name: "Now & next" });
  expect(await within(tile).findByText(NOTICE)).toBeDefined();

  let failed = 0;
  server.use(
    http.get("/api/proxy/api/home", () => {
      failed += 1;
      return HttpResponse.json({ detail: "boom" }, { status: 500 });
    }),
  );
  await act(async () => {
    vi.advanceTimersByTime(15_000);
  });

  // The failed refresh has been handled: the query holds the error and the
  // last dashboard together.
  await waitFor(() =>
    expect(
      queryClient.getQueryState(getGetHomeDashboardQueryKey()),
    ).toMatchObject({ status: "error", fetchStatus: "idle" }),
  );
  expect(failed).toBeGreaterThan(0);
  expect(
    screen.queryByText("Your Home briefing could not be loaded"),
  ).toBeNull();
  expect(within(tile).getByText(NOTICE)).toBeDefined();
  expect(screen.getByRole("heading", { name: "Your team" })).toBeDefined();
});

test("Now & next's live region is in place before an outage starts", async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  let dashboard = afterRecovery;
  serveDashboard(() => dashboard);

  render(<HomeRecap />);
  const tile = await screen.findByRole("region", { name: "Now & next" });
  expect(await within(tile).findByText("Spanish practice plan")).toBeDefined();
  // Mounted, and empty, so its text is announced when it arrives.
  const region = within(tile).getByTestId("schedules-unavailable-status");
  expect(region.textContent).toBe("");

  dashboard = duringOutage;
  await act(async () => {
    vi.advanceTimersByTime(60_000);
  });

  await waitFor(() => expect(region.textContent).toContain(NOTICE));
});
