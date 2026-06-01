import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import {
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
import { Key } from "@/services/storage/local-storage";
import { TopUpPromptProvider } from "@/components/layout/TopUpPrompt/TopUpPromptProvider";
import { LowCreditBanner } from "@/components/layout/TopUpPrompt/LowCreditBanner/LowCreditBanner";
import { useTopUpPrompt } from "@/components/layout/TopUpPrompt/useTopUpPrompt";

// Billing must be on for the provider to derive `isOutOfCredits`; keep the real
// `Flag` enum and let each test toggle whether the flag resolves true.
let isBillingEnabled = true;

vi.mock("@/services/feature-flags/use-get-flag", async (importActual) => {
  const actual =
    await importActual<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) =>
      flag === actual.Flag.ENABLE_PLATFORM_PAYMENT ? isBillingEnabled : false,
  };
});

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

function renderProvider() {
  return render(
    <TopUpPromptProvider>
      <div>ready</div>
      <LowCreditBanner />
    </TopUpPromptProvider>,
  );
}

beforeEach(() => {
  localStorage.clear();
  isBillingEnabled = true;
});

afterEach(() => {
  localStorage.clear();
});

describe("TopUpPromptProvider daily auto-opener", () => {
  test("auto-opens the top-up dialog once when out of credits", async () => {
    setupCredits({ credits: 0, amount: 0, threshold: 0 });

    renderProvider();

    // The dialog body copy mentions Autopilot, which the banner copy does not,
    // so it unambiguously signals the dialog auto-opened.
    expect(
      await screen.findByText(/keep your agents and Autopilot/i),
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

    expect(screen.queryByText(/keep your agents and Autopilot/i)).toBeNull();
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
    expect(screen.queryByText(/keep your agents and Autopilot/i)).toBeNull();
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
    expect(screen.queryByText(/keep your agents and Autopilot/i)).toBeNull();
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
    expect(screen.queryByText(/keep your agents and Autopilot/i)).toBeNull();
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
    expect(screen.queryByText(/keep your agents and Autopilot/i)).toBeNull();
  });
});
