import { QueryClient } from "@tanstack/react-query";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  getGetV1GetAutoTopUpMockHandler,
  getGetV1GetCreditHistoryMockHandler,
  getGetV1GetUserCreditsMockHandler,
} from "@/app/api/__generated__/endpoints/credits/credits.msw";
import { getGetV1CheckIfOnboardingIsCompletedMockHandler200 } from "@/app/api/__generated__/endpoints/onboarding/onboarding.msw";
import { getGetTrialsGetTrialStatusMockHandler200 } from "@/app/api/__generated__/endpoints/trials/trials.msw";
import type { CreditTransactionItem } from "@/app/api/__generated__/models/creditTransactionItem";
import { resetClientRequestFailureLog } from "@/app/api/mutators/request-failure-log";
import { getCurrentUser } from "@/lib/auth/actions";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { server } from "@/mocks/mock-server";
import {
  act,
  cleanup,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import { trialResponse } from "@/tests/integrations/trial-fixtures";

import SettingsBillingPage from "../page";

// Sign-in's start-up lookup, which a test holds to keep sign-in loading.
vi.mock("@/lib/auth/actions", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/auth/actions")>()),
  getCurrentUser: vi.fn(),
}));

// These cover the hosted billing UI, which self-hosted builds hide.
vi.mock("@/hooks/useIsBillingEnabled", () => ({
  useIsBillingEnabled: () => true,
  useBillingStatus: () => ({ isBillingEnabled: true, isResolved: true }),
}));

// The address bar. The router replaces it and re-renders the page as Next's
// does, so rendering the page again after a replace is a reload there.
const address = vi.hoisted(() => ({
  search: new URLSearchParams(),
  listeners: new Set<() => void>(),
}));
vi.mock("next/navigation", async (importOriginal) => {
  const { useSyncExternalStore } = await import("react");
  function subscribe(listener: () => void) {
    address.listeners.add(listener);
    return () => {
      address.listeners.delete(listener);
    };
  }
  const router = {
    push: vi.fn(),
    replace: (url: string) => {
      address.search = new URL(url, "https://app.test").searchParams;
      address.listeners.forEach((listener) => listener());
    },
    prefetch: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
    refresh: vi.fn(),
  };
  return {
    ...(await importOriginal<typeof import("next/navigation")>()),
    useSearchParams: () =>
      useSyncExternalStore(subscribe, () => address.search),
    useRouter: () => router,
    usePathname: () => "/settings/billing",
    useParams: () => ({}),
  };
});

// One credit is one cent: $6.00 before a $7.00 top-up.
const BEFORE = 600;
const TOP_UP = 700;

const buyer = { id: "buyer", email: "buyer@example.com", user_metadata: {} };

const grant: CreditTransactionItem = {
  id: "grant-1",
  transaction_key: "grant-1",
  transaction_type: "GRANT",
  transaction_time: new Date("2026-11-20T09:00:00Z"),
  amount: BEFORE,
  user_id: buyer.id,
};

const purchase: CreditTransactionItem = {
  id: "top-up-1",
  transaction_key: "pi_test_top_up",
  transaction_type: "TOP_UP",
  transaction_time: new Date("2026-11-22T15:00:00Z"),
  amount: TOP_UP,
  user_id: buyer.id,
};

// Answers for the calls tests hold: the page's fulfil call and sign-in's
// start-up lookup. Teardown unmounts the page, then sends any a failed test
// left unanswered, so none outlives its test.
const heldAnswers: Array<() => Promise<void>> = [];

beforeEach(() => {
  address.search = new URLSearchParams({
    topup: "success",
    session_id: "cs_test_top_up",
  });
  useAuthStore.setState({
    user: buyer,
    isUserLoading: false,
    hasLoadedUser: true,
  });
});

afterEach(async () => {
  cleanup();
  // A new epoch makes a start-up a failed test left waiting drop its answer
  // instead of signing the next test out.
  useAuthStore.setState({
    sessionEpoch: useAuthStore.getState().sessionEpoch + 1,
  });
  await Promise.all(heldAnswers.splice(0).map((answer) => answer()));
  useAuthStore.getState().cleanup();
  vi.mocked(getCurrentUser).mockReset();
  useAuthStore.setState({ user: null, initializationPromise: null });
  address.search = new URLSearchParams();
  resetClientRequestFailureLog();
});

describe("returning from a top-up Checkout", () => {
  it("shows the credits once the page's own fulfil call adds them, without a reload", async () => {
    const platform = mockTopUpReturn();
    renderBillingPage();

    expect(await findBalance("$6.00")).toBeDefined();
    expect(await findHistoryRow("Credits granted")).toBeDefined();
    await waitFor(() => expect(platform.ledger.fulfilCalls).toBe(1));
    expect(queryHistoryRow("Credits added")).toBeNull();

    await platform.answerFulfil();

    expect(await findBalance("$13.00")).toBeDefined();
    const row = await findHistoryRow("Credits added");
    expect(within(row).getByText("+$7.00")).toBeDefined();
  });

  it("shows the credits on the next visit when the page closed before the fulfil call answered", async () => {
    const platform = mockTopUpReturn();
    const firstVisit = renderBillingPage();
    expect(await findBalance("$6.00")).toBeDefined();
    await waitFor(() => expect(platform.ledger.fulfilCalls).toBe(1));
    firstVisit.unmount();

    await platform.answerFulfil();
    await waitFor(() => expect(firstVisit.queryClient.isMutating()).toBe(0));
    renderBillingPage(firstVisit.queryClient);

    expect(await findBalance("$13.00")).toBeDefined();
  });

  it("keeps Automation Credits in the address, so the refresh the toast suggests opens it", async () => {
    const platform = mockTopUpReturn();
    const firstVisit = renderBillingPage();
    await waitFor(() => expect(address.search.has("topup")).toBe(false));
    await waitFor(() => expect(platform.ledger.fulfilCalls).toBe(1));
    await platform.answerFulfil();
    firstVisit.unmount();

    renderBillingPage();

    expect(selectedTab()).toBe("Automation Credits");
    expect(await findBalance("$13.00")).toBeDefined();
    expect(address.search.toString()).toBe("tab=automation-credits");
  });

  it("keeps the tab someone switches to in the address, so a reload opens it", async () => {
    const platform = mockTopUpReturn();
    mockNoTrial();
    const firstVisit = renderBillingPage();
    await waitFor(() => expect(platform.ledger.fulfilCalls).toBe(1));
    await platform.answerFulfil();

    await userEvent.click(screen.getByRole("tab", { name: "Subscription" }));

    expect(address.search.toString()).toBe("tab=subscription");
    firstVisit.unmount();
    renderBillingPage();
    expect(selectedTab()).toBe("Subscription");
  });

  it("keeps a tab picked while sign-in loads when the return cleans up the address", async () => {
    const platform = mockTopUpReturn();
    mockNoTrial();
    const signIn = holdSignIn();
    useAuthStore.setState({
      user: null,
      isUserLoading: true,
      hasLoadedUser: false,
    });
    renderBillingPage();
    await userEvent.click(screen.getByRole("tab", { name: "Subscription" }));

    await act(async () => {
      signIn.resolve({ user: buyer });
      await useAuthStore.getState().initializationPromise;
    });
    await waitFor(() => expect(platform.ledger.fulfilCalls).toBe(1));
    await platform.answerFulfil();

    expect(selectedTab()).toBe("Subscription");
    expect(address.search.toString()).toBe("tab=subscription");
  });
});

describe("when Stripe's webhook adds the credits after the fulfil call", () => {
  beforeEach(() => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("reads the balance and history once more a little later, then stops", async () => {
    const platform = mockTopUpReturn({ fulfilFails: true });
    const { queryClient } = renderBillingPage();
    expect(await findBalance("$6.00")).toBeDefined();

    await platform.answerFulfil();
    await waitFor(() => expect(platform.ledger.balanceReads).toBe(2));
    await waitFor(() => expect(queryClient.isFetching()).toBe(0));
    // Only advance() moves the clock across the late read, however slow the
    // runner; re-installing fake timers would drop the pending read instead.
    vi.setTimerTickMode("manual");
    expect(queryHistoryRow("Credits added")).toBeNull();
    platform.creditFromWebhook();
    await advance(10_000);
    vi.setTimerTickMode("interval");

    expect(await findBalance("$13.00")).toBeDefined();
    expect(await findHistoryRow("Credits added")).toBeDefined();
    const reads = platform.ledger.balanceReads;
    await advance(60_000);
    expect(platform.ledger.balanceReads).toBe(reads);
  });
});

interface Options {
  fulfilFails?: boolean;
}

// The platform's side of the top-up, for a buyer past onboarding. The page's
// fulfil call is held until the test answers it; a successful one credits the
// purchase, as the platform does once Stripe reports the session paid.
function mockTopUpReturn({ fulfilFails = false }: Options = {}) {
  const ledger = { credited: false, balanceReads: 0, fulfilCalls: 0 };
  const release = Promise.withResolvers<void>();
  const answered = Promise.withResolvers<void>();
  heldAnswers.push(async () => {
    release.resolve();
    if (ledger.fulfilCalls > 0) await answered.promise;
  });
  server.use(
    getGetV1GetUserCreditsMockHandler(() => {
      ledger.balanceReads += 1;
      return { credits: ledger.credited ? BEFORE + TOP_UP : BEFORE };
    }),
    getGetV1GetCreditHistoryMockHandler(() => ({
      transactions: ledger.credited ? [purchase, grant] : [grant],
      next_transaction_time: null,
      next_cursor: null,
    })),
    getGetV1GetAutoTopUpMockHandler({ amount: 0, threshold: 0 }),
    getGetV1CheckIfOnboardingIsCompletedMockHandler200({ is_completed: true }),
    http.patch("*/api/credits", async () => {
      ledger.fulfilCalls += 1;
      await release.promise;
      if (!fulfilFails) ledger.credited = true;
      answered.resolve();
      if (fulfilFails)
        return HttpResponse.json(
          { detail: "Stripe did not answer" },
          { status: 502 },
        );
      return new HttpResponse(null, { status: 200 });
    }),
  );
  return {
    ledger,
    answerFulfil: () =>
      act(async () => {
        release.resolve();
        await answered.promise;
      }),
    creditFromWebhook: () => {
      ledger.credited = true;
    },
  };
}

// Holds sign-in's start-up lookup until the test answers it.
function holdSignIn() {
  const lookup =
    Promise.withResolvers<Awaited<ReturnType<typeof getCurrentUser>>>();
  vi.mocked(getCurrentUser).mockReturnValue(lookup.promise);
  heldAnswers.push(async () => {
    lookup.resolve({ user: null });
    await useAuthStore.getState().initializationPromise;
  });
  return lookup;
}

// Signed in, the Subscription tab reads the trial; this buyer has none.
function mockNoTrial() {
  server.use(
    getGetTrialsGetTrialStatusMockHandler200(
      trialResponse({ active: false, status: "canceled" }),
    ),
  );
}

// The app keeps query data fresh for a minute (lib/react-query/queryClient.ts),
// so only the page itself reads the balance or history again.
function renderBillingPage(
  queryClient = new QueryClient({
    defaultOptions: { queries: { staleTime: 60_000, retry: false } },
  }),
) {
  return {
    queryClient,
    ...render(<SettingsBillingPage />, { queryClient }),
  };
}

async function findBalance(amount: string) {
  const addCredits = await screen.findByRole("button", {
    name: "Add credits",
  });
  const card = addCredits.closest("section");
  if (!card) throw new Error("Add credits is outside the balance card");
  return within(card).findByText(amount);
}

function historyRowName(activity: string) {
  return new RegExp(`^Open details for ${activity},`);
}

async function findHistoryRow(activity: string) {
  const details = await screen.findByRole("button", {
    name: historyRowName(activity),
  });
  const row = details.closest("tr");
  if (!row) throw new Error(`No history row for ${activity}`);
  return row;
}

function queryHistoryRow(activity: string) {
  return screen.queryByRole("button", { name: historyRowName(activity) });
}

function selectedTab() {
  return screen.getByRole("tab", { selected: true }).textContent;
}

function advance(ms: number) {
  return act(() => vi.advanceTimersByTimeAsync(ms));
}
