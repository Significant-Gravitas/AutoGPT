import { QueryClient } from "@tanstack/react-query";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  getGetV1GetAutoTopUpMockHandler,
  getGetV1GetCreditHistoryMockHandler,
  getGetV1GetUserCreditsMockHandler,
} from "@/app/api/__generated__/endpoints/credits/credits.msw";
import type { CreditTransactionItem } from "@/app/api/__generated__/models/creditTransactionItem";
import { resetClientRequestFailureLog } from "@/app/api/mutators/request-failure-log";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { server } from "@/mocks/mock-server";
import {
  act,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";

import SettingsBillingPage from "../page";

// These cover the hosted billing UI, which self-hosted builds hide.
vi.mock("@/hooks/useIsBillingEnabled", () => ({
  useIsBillingEnabled: () => true,
  useBillingStatus: () => ({ isBillingEnabled: true, isResolved: true }),
}));

// The address bar. The router replaces it as Next's does, so rendering the
// page again after a replace is the page reloaded at the new address.
const address = vi.hoisted(() => ({ search: new URLSearchParams() }));
vi.mock("next/navigation", async (importOriginal) => {
  const router = {
    push: vi.fn(),
    replace: (url: string) => {
      address.search = new URL(url, "https://app.test").searchParams;
    },
    prefetch: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
    refresh: vi.fn(),
  };
  return {
    ...(await importOriginal<typeof import("next/navigation")>()),
    useSearchParams: () => address.search,
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

afterEach(() => {
  useAuthStore.setState({ user: null });
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

  it("keeps Automation Credits in the address, so the refresh the toast suggests opens it", async () => {
    const platform = mockTopUpReturn();
    const firstVisit = renderBillingPage();
    await waitFor(() => expect(address.search.has("topup")).toBe(false));
    await waitFor(() => expect(platform.ledger.fulfilCalls).toBe(1));
    await platform.answerFulfil();
    firstVisit.unmount();

    renderBillingPage();

    expect(screen.getByRole("tab", { selected: true }).textContent).toBe(
      "Automation Credits",
    );
    expect(await findBalance("$13.00")).toBeDefined();
    expect(address.search.toString()).toBe("tab=automation-credits");
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
    expect(queryHistoryRow("Credits added")).toBeNull();
    platform.creditFromWebhook();
    await advance(10_000);

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

// The platform's side of the top-up. The page's fulfil call is held until the
// test answers it; a successful one credits the purchase, as the platform does
// once Stripe reports the session paid.
function mockTopUpReturn({ fulfilFails = false }: Options = {}) {
  const ledger = { credited: false, balanceReads: 0, fulfilCalls: 0 };
  const release = Promise.withResolvers<void>();
  const answered = Promise.withResolvers<void>();
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

// The app keeps query data fresh for a minute (lib/react-query/queryClient.ts),
// so only the page itself reads the balance or history again.
function renderBillingPage() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { staleTime: 60_000, retry: false } },
  });
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

function advance(ms: number) {
  return act(() => vi.advanceTimersByTimeAsync(ms));
}
