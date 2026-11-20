import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import {
  act,
  render,
  screen,
  fireEvent,
  waitFor,
} from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import {
  getGetV1GetUserCreditsMockHandler,
  getGetV1GetAutoTopUpMockHandler,
} from "@/app/api/__generated__/endpoints/credits/credits.msw";
import { getGetV1CheckIfOnboardingIsCompletedMockHandler } from "@/app/api/__generated__/endpoints/onboarding/onboarding.msw";
import { Key } from "@/services/storage/local-storage";
import { TopUpPromptProvider } from "@/components/layout/TopUpPrompt/TopUpPromptProvider";
import { LowCreditBanner } from "@/components/layout/TopUpPrompt/LowCreditBanner/LowCreditBanner";
import { useTopUpPrompt } from "@/components/layout/TopUpPrompt/useTopUpPrompt";
import { useOnboarding } from "@/providers/onboarding/onboarding-provider";
import { useState } from "react";

// Billing must be on for the provider to derive `isOutOfCredits`; keep the real
// `Flag` enum and let each test toggle whether the flag resolves true.
let isBillingEnabled = true;
let authenticatedUserId: string | null = "user-a";

// Simulate an authenticated user so that fetchInitialCredits and
// fetchInitialAutoTopUpConfig are enabled (they gate on isLoggedIn).
vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    user: authenticatedUserId ? { id: authenticatedUserId } : null,
    isLoggedIn: Boolean(authenticatedUserId),
  }),
}));

vi.mock("@/services/feature-flags/use-get-flag", async (importActual) => {
  const actual =
    await importActual<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: () => false,
  };
});

vi.mock("@/hooks/usePlatformPayment", () => ({
  usePlatformPayment: () => ({
    isPaymentEnabled: isBillingEnabled,
    isResolved: true,
  }),
}));

function setupCredits(args: {
  credits: number;
  amount: number;
  threshold: number;
}) {
  let creditsRequested = false;
  let autoTopUpRequested = false;
  server.use(
    getGetV1GetUserCreditsMockHandler(() => {
      creditsRequested = true;
      return { credits: args.credits };
    }),
    getGetV1GetAutoTopUpMockHandler(() => {
      autoTopUpRequested = true;
      return { amount: args.amount, threshold: args.threshold };
    }),
  );
  return {
    waitForCreditsFetch: () =>
      waitFor(() => {
        expect(creditsRequested).toBe(true);
        expect(autoTopUpRequested).toBe(true);
      }),
  };
}

function providerTree() {
  return (
    <TopUpPromptProvider>
      <div>ready</div>
      <StatefulChild />
      <LowCreditBanner />
    </TopUpPromptProvider>
  );
}

function StatefulChild() {
  const [value, setValue] = useState(0);
  return (
    <button onClick={() => setValue(value + 1)}>page state {value}</button>
  );
}

function renderProvider() {
  return render(providerTree());
}

beforeEach(() => {
  localStorage.clear();
  authenticatedUserId = "user-a";
  isBillingEnabled = true;
});

afterEach(() => {
  localStorage.clear();
  vi.restoreAllMocks();
});

describe("TopUpPromptProvider daily auto-opener", () => {
  test("auto-opens the top-up dialog once when out of credits", async () => {
    setupCredits({ credits: 0, amount: 0, threshold: 0 });

    renderProvider();

    // The dialog body copy mentions experts, which the banner copy does not,
    // so it unambiguously signals the dialog auto-opened.
    expect(
      await screen.findByText(/keep your workflows and experts/i),
    ).toBeDefined();
  });

  test("does not auto-open when the modal was already shown today", async () => {
    localStorage.setItem(
      Key.TOP_UP_MODAL_LAST_SHOWN,
      new Date().toDateString(),
    );
    setupCredits({ credits: 0, amount: 0, threshold: 0 });

    renderProvider();

    // The banner is a separate concern and still shows; awaiting it proves the
    // credit fetch resolved before we assert the dialog is absent.
    await screen.findByText(/out of automation credits/i);

    expect(screen.queryByText(/keep your workflows and experts/i)).toBeNull();
  });
});

describe("LowCreditBanner dismissal", () => {
  test("hides the banner for the day and records the dismissal date", async () => {
    // Suppress the daily auto-opener so only the banner dismissal is under test.
    localStorage.setItem(
      Key.TOP_UP_MODAL_LAST_SHOWN,
      new Date().toDateString(),
    );
    setupCredits({ credits: 0, amount: 0, threshold: 0 });

    renderProvider();

    await screen.findByText(/out of automation credits/i);

    fireEvent.click(screen.getByRole("button", { name: /dismiss/i }));

    await waitFor(() =>
      expect(screen.queryByText(/out of automation credits/i)).toBeNull(),
    );
    expect(localStorage.getItem(Key.LOW_CREDIT_BANNER_DISMISSED)).toBe(
      new Date().toDateString(),
    );
  });
});

describe("useTopUpPrompt without a provider", () => {
  test("falls back to an inert value instead of throwing", () => {
    function Consumer() {
      const { isOutOfCredits, openTopUp, closeTopUp } = useTopUpPrompt();
      return (
        <button
          onClick={() => {
            openTopUp();
            closeTopUp();
          }}
        >
          {isOutOfCredits ? "out of credits" : "has credits"}
        </button>
      );
    }

    render(<Consumer />);

    const button = screen.getByRole("button");
    expect(button.textContent).toBe("has credits");
    // The inert openTopUp/closeTopUp must be safe no-ops.
    fireEvent.click(button);
    expect(button.textContent).toBe("has credits");
  });
});

describe("TopUpPromptProvider out-of-credits suppression", () => {
  test("does not fetch billing data when logged out", async () => {
    authenticatedUserId = null;
    let creditsRequested = false;
    let autoTopUpRequested = false;
    server.use(
      getGetV1GetUserCreditsMockHandler(() => {
        creditsRequested = true;
        return { credits: 0 };
      }),
      getGetV1GetAutoTopUpMockHandler(() => {
        autoTopUpRequested = true;
        return { amount: 0, threshold: 0 };
      }),
    );

    renderProvider();
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(creditsRequested).toBe(false);
    expect(autoTopUpRequested).toBe(false);
  });

  test("resets and refetches billing state when the authenticated user changes", async () => {
    let credits = 0;
    let creditRequests = 0;
    server.use(
      getGetV1GetUserCreditsMockHandler(() => {
        creditRequests += 1;
        return { credits };
      }),
      getGetV1GetAutoTopUpMockHandler({ amount: 0, threshold: 0 }),
    );

    const view = renderProvider();

    const banner = await screen.findByRole("alert");
    expect(banner.textContent).toMatch(/out of automation credits/i);
    expect(banner.textContent).toMatch(/keep your workflows running/i);
    expect(creditRequests).toBe(1);
    fireEvent.click(screen.getByText("page state 0"));

    credits = 500;
    authenticatedUserId = "user-b";
    view.rerender(providerTree());

    await waitFor(() => expect(creditRequests).toBe(2));
    expect(screen.getByText("page state 1")).toBeDefined();
    await waitFor(() =>
      expect(screen.queryByText(/out of automation credits/i)).toBeNull(),
    );
  });

  test("suppresses dialog and banner when auto-refill is enabled", async () => {
    const { waitForCreditsFetch } = setupCredits({
      credits: 0,
      amount: 1000,
      threshold: 500,
    });

    renderProvider();

    // What actually keeps the banner/dialog out here is the invariant: with
    // auto-refill enabled, `isOutOfCredits` is permanently false, so nothing can
    // appear regardless of fetch timing. `waitForCreditsFetch` only gates on
    // both backend reads (credits + auto-top-up) being intercepted (not on
    // state being applied to the tree), kept as belt-and-suspenders so the
    // assertions don't run on a bare initial mount.
    await screen.findByText("ready");
    await waitForCreditsFetch();

    expect(screen.queryByText(/out of automation credits/i)).toBeNull();
    expect(screen.queryByText(/keep your workflows and experts/i)).toBeNull();
  });

  test("renders nothing when the billing flag is off", async () => {
    isBillingEnabled = false;
    const { waitForCreditsFetch } = setupCredits({
      credits: 0,
      amount: 0,
      threshold: 0,
    });

    renderProvider();

    // With the billing flag off, `isOutOfCredits` is permanently false — the
    // invariant that keeps banner/dialog out, independent of fetch timing.
    // `waitForCreditsFetch` only confirms both backend reads were intercepted
    // (not that state was applied), kept as belt-and-suspenders.
    await screen.findByText("ready");
    await waitForCreditsFetch();

    expect(screen.queryByText(/out of automation credits/i)).toBeNull();
    expect(screen.queryByText(/keep your workflows and experts/i)).toBeNull();
  });

  test("renders nothing when the user still has a positive balance", async () => {
    const { waitForCreditsFetch } = setupCredits({
      credits: 500,
      amount: 0,
      threshold: 0,
    });

    renderProvider();

    await screen.findByText("ready");
    await waitForCreditsFetch();

    expect(screen.queryByText(/out of automation credits/i)).toBeNull();
    expect(screen.queryByText(/keep your workflows and experts/i)).toBeNull();
  });

  test("suppresses the prompt when the credits fetch fails", async () => {
    // Simulates a transient backend error on GET /credits. The client used to
    // silently treat this as a `$0` balance, which fired the out-of-credits
    // prompt on any API failure. Returning `null` from `getUserCredit` on
    // error keeps the provider's `credits !== null` guard short-circuiting.
    let autoTopUpRequested = false;
    server.use(
      http.get("*/api/v1/credits", () =>
        HttpResponse.json({ detail: "boom" }, { status: 500 }),
      ),
      getGetV1GetAutoTopUpMockHandler(() => {
        autoTopUpRequested = true;
        return { amount: 0, threshold: 0 };
      }),
    );

    renderProvider();

    await screen.findByText("ready");
    await waitFor(() => expect(autoTopUpRequested).toBe(true));

    expect(screen.queryByText(/out of automation credits/i)).toBeNull();
    expect(screen.queryByText(/keep your workflows and experts/i)).toBeNull();
  });
});

// A new account has 0 credits until onboarding sets up its plan. Neither
// prompt may show before onboarding is known to be done, and the once-a-day
// auto-open must not be used up on the way to it.
describe("TopUpPromptProvider before onboarding is finished", () => {
  // Released here, not in onTestFinished: a describe's afterEach runs before
  // the shared teardown, so the answer it lets out is still awaited there.
  let releaseOnboardingCheck = () => {};
  afterEach(() => releaseOnboardingCheck());

  function OnboardingAnswer() {
    const { isOnboardingComplete } = useOnboarding();
    return <span>onboarding {String(isOnboardingComplete)}</span>;
  }

  function renderWithAnswer() {
    return render(
      <>
        {providerTree()}
        <OnboardingAnswer />
      </>,
    );
  }

  // Lets the credit responses already received reach the tree; the
  // assertions after it are absences, which no event can announce.
  function settle() {
    return act(() => new Promise((resolve) => setTimeout(resolve, 50)));
  }

  test("shows neither the dialog nor the banner to a user who has not finished onboarding", async () => {
    let onboardingChecked = false;
    server.use(
      getGetV1CheckIfOnboardingIsCompletedMockHandler(() => {
        onboardingChecked = true;
        return { is_completed: false };
      }),
    );
    const { waitForCreditsFetch } = setupCredits({
      credits: 0,
      amount: 0,
      threshold: 0,
    });

    renderWithAnswer();

    await screen.findByText("ready");
    await waitForCreditsFetch();
    await waitFor(() => expect(onboardingChecked).toBe(true));
    await screen.findByText("onboarding false");
    await settle();

    expect(screen.queryByText(/out of automation credits/i)).toBeNull();
    expect(screen.queryByText(/keep your workflows and experts/i)).toBeNull();
    expect(localStorage.getItem(Key.TOP_UP_MODAL_LAST_SHOWN)).toBeNull();
  });

  test("waits for the onboarding check, then opens once onboarding is known to be done", async () => {
    const onboardingAnswer = new Promise<void>((resolve) => {
      releaseOnboardingCheck = resolve;
    });
    let onboardingRequested = false;
    server.use(
      getGetV1CheckIfOnboardingIsCompletedMockHandler(async () => {
        onboardingRequested = true;
        await onboardingAnswer;
        return { is_completed: true };
      }),
    );
    const { waitForCreditsFetch } = setupCredits({
      credits: 0,
      amount: 0,
      threshold: 0,
    });

    renderWithAnswer();

    await screen.findByText("ready");
    await waitForCreditsFetch();
    await waitFor(() => expect(onboardingRequested).toBe(true));
    await settle();
    expect(screen.getByText("onboarding null")).toBeDefined();

    expect(screen.queryByText(/out of automation credits/i)).toBeNull();
    expect(screen.queryByText(/keep your workflows and experts/i)).toBeNull();
    expect(localStorage.getItem(Key.TOP_UP_MODAL_LAST_SHOWN)).toBeNull();

    releaseOnboardingCheck();

    expect(await screen.findByText("onboarding true")).toBeDefined();
    expect(
      await screen.findByText(/keep your workflows and experts/i),
    ).toBeDefined();
  });
});
