import { useGetSubscriptionStatus } from "@/app/api/__generated__/endpoints/credits/credits";
import {
  getGetTrialsGetTrialStatusMockHandler200,
  getPostTrialsCancelTrialMockHandler200,
} from "@/app/api/__generated__/endpoints/trials/trials.msw";
import {
  formatTrialEnd,
  formatTrialEndDate,
  formatTrialEndTime,
} from "@/components/organisms/TrialCard/helpers";
import { TrialCard } from "@/components/organisms/TrialCard/TrialCard";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import {
  deferredTrialResponse,
  setTrialUser,
  trialOffer,
  trialResponse,
} from "@/tests/integrations/trial-fixtures";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const mockPush = vi.hoisted(() => vi.fn());
vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: mockPush,
    replace: vi.fn(),
    prefetch: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
    refresh: vi.fn(),
  }),
  usePathname: () => "/settings/billing",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
}));

const posthog = vi.hoisted(() => ({ capture: vi.fn() }));
vi.mock("@posthog/react", () => ({
  usePostHog: () => posthog,
  useFeatureFlagVariantKey: () => undefined,
}));

const HOUR = 60 * 60 * 1000;
const DAY = 24 * HOUR;
const endsAt = new Date("2030-09-17T15:00:00Z");

function active(ends = endsAt) {
  return trialResponse({ ends_at: ends });
}

function cancelPending(ends = endsAt) {
  return trialResponse({
    cancel_at_period_end: true,
    ends_at: ends,
  });
}

function mockCancelFlow(ends = endsAt) {
  const cancel = vi.fn(() => cancelPending(ends));
  server.use(
    getGetTrialsGetTrialStatusMockHandler200(active(ends)),
    getPostTrialsCancelTrialMockHandler200(cancel),
  );
  return cancel;
}

function SubscriptionStatusObserver() {
  useGetSubscriptionStatus();
  return null;
}

async function cancelThroughConfirmation() {
  fireEvent.click(await screen.findByRole("button", { name: "Cancel trial" }));
  const confirm = await screen.findByRole("dialog", {
    name: "Cancel your trial?",
  });
  fireEvent.click(
    within(confirm).getByRole("button", { name: "Cancel trial" }),
  );
  return screen.findByRole("dialog", { name: "Cancellation confirmed" });
}

beforeEach(() => {
  setTrialUser();
  mockPush.mockClear();
  posthog.capture.mockClear();
  vi.useFakeTimers({ toFake: ["Date"] });
  vi.setSystemTime(new Date(endsAt.getTime() - 5 * DAY));
});
afterEach(() => {
  vi.useRealTimers();
  setTrialUser(null);
});

describe("post-cancel popup", () => {
  it("opens once after the cancel and never again on a remount", async () => {
    mockCancelFlow();
    const { unmount } = render(<TrialCard />);
    const popup = await cancelThroughConfirmation();
    expect(popup.textContent).toContain(
      `Your card won't be charged. You still have 5 days of full access, until ${formatTrialEnd(endsAt)}. Nothing changes until then.`,
    );
    expect(within(popup).getByText("5 days").tagName).toBe("STRONG");
    expect(within(popup).getByText("Worth doing before then")).toBeDefined();
    expect(
      within(popup).getByText("Put an Expert on a schedule"),
    ).toBeDefined();
    expect(within(popup).getByText("Your work stays")).toBeDefined();
    expect(
      within(popup).getByText(
        `Pro is $20 / month, cancel anytime. Resume your trial instead and your plan starts ${formatTrialEndDate(endsAt)}.`,
      ),
    ).toBeDefined();
    expect(posthog.capture).toHaveBeenCalledWith("trial_cancel_popup_viewed", {
      days_left: 5,
    });

    fireEvent.click(within(popup).getByRole("button", { name: "Close" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(screen.getByText("Cancellation pending")).toBeDefined();

    unmount();
    server.use(getGetTrialsGetTrialStatusMockHandler200(cancelPending()));
    render(<TrialCard />);
    await screen.findByText("Cancellation pending");
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(
      posthog.capture.mock.calls.filter(
        ([event]) => event === "trial_cancel_popup_viewed",
      ),
    ).toHaveLength(1);
  });

  it("opens without waiting for the plan status refresh", async () => {
    mockCancelFlow();
    const planStatus = deferredTrialResponse<void>();
    server.use(
      http.get("*/api/credits/subscription", async () => {
        await planStatus.promise;
        return HttpResponse.json({ tier: "TRIAL", monthly_cost: 0 });
      }),
    );
    render(
      <>
        <TrialCard />
        <SubscriptionStatusObserver />
      </>,
    );
    const popup = await cancelThroughConfirmation();
    expect(popup).toBeDefined();
    expect(posthog.capture).toHaveBeenCalledWith("trial_cancel_popup_viewed", {
      days_left: 5,
    });
    planStatus.resolve();
  });

  it("closes on Escape and leaves the trial cancel-pending", async () => {
    mockCancelFlow();
    render(<TrialCard />);
    const popup = await cancelThroughConfirmation();
    fireEvent.keyDown(popup, { key: "Escape" });
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(screen.getByRole("button", { name: "Resume trial" })).toBeDefined();
  });

  it("does not open when the cancel finds the trial already over", async () => {
    server.use(
      getGetTrialsGetTrialStatusMockHandler200(trialResponse()),
      getPostTrialsCancelTrialMockHandler200(
        trialResponse({ active: false, status: "canceled" }),
      ),
    );
    render(<TrialCard />);
    fireEvent.click(
      await screen.findByRole("button", { name: "Cancel trial" }),
    );
    fireEvent.click(
      within(
        await screen.findByRole("dialog", { name: "Cancel your trial?" }),
      ).getByRole("button", { name: "Cancel trial" }),
    );
    await screen.findByText("Your trial has ended");
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(posthog.capture).not.toHaveBeenCalledWith(
      "trial_cancel_popup_viewed",
      expect.anything(),
    );
  });

  it.each(["billing", "onboarding"] as const)(
    "sends Subscribe now from the %s card to billing and closes",
    async (returnTo) => {
      mockCancelFlow();
      render(<TrialCard returnTo={returnTo} />);
      const popup = await cancelThroughConfirmation();
      fireEvent.click(
        within(popup).getByRole("button", { name: "Subscribe now" }),
      );
      await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
      expect(mockPush).toHaveBeenCalledExactlyOnceWith("/settings/billing");
      expect(posthog.capture).toHaveBeenCalledWith(
        "trial_subscribe_now_clicked",
        { days_left: 5 },
      );
    },
  );
});

describe("time left in the popup", () => {
  it.each([
    [5 * DAY, "5 days", 5],
    [DAY, "1 day", 1],
    [DAY + HOUR, "2 days", 2],
  ])("reads %ims as %s", async (left, label, daysLeft) => {
    vi.setSystemTime(new Date(endsAt.getTime() - left));
    mockCancelFlow();
    render(<TrialCard />);
    const popup = await cancelThroughConfirmation();
    expect(popup.textContent).toContain(
      `You still have ${label} of full access, until ${formatTrialEnd(endsAt)}.`,
    );
    expect(posthog.capture).toHaveBeenCalledWith("trial_cancel_popup_viewed", {
      days_left: daysLeft,
    });
  });

  it.each([
    ["today", new Date(2030, 8, 17, 11, 10), new Date(2030, 8, 17, 14, 10)],
    ["tomorrow", new Date(2030, 8, 16, 22, 0), new Date(2030, 8, 17, 1, 0)],
  ])(
    "names the clock time %s when under a day is left",
    async (day, now, ends) => {
      vi.setSystemTime(now);
      mockCancelFlow(ends);
      render(<TrialCard />);
      const popup = await cancelThroughConfirmation();
      expect(popup.textContent).toContain(
        `You still have full access until ${formatTrialEndTime(ends)} ${day}. Nothing changes until then.`,
      );
      expect(popup.textContent).not.toMatch(/\bdays?\b of full access/);
      expect(
        within(popup).getByText(
          `Pro is $20 / month, cancel anytime. Resume your trial instead and your plan starts ${day}.`,
        ),
      ).toBeDefined();
      expect(posthog.capture).toHaveBeenCalledWith(
        "trial_cancel_popup_viewed",
        {
          days_left: 1,
        },
      );
    },
  );
});

describe("popup prices", () => {
  it("keeps the cents of a price that has them", async () => {
    mockCancelFlow();
    server.use(
      getPostTrialsCancelTrialMockHandler200(
        trialResponse({
          cancel_at_period_end: true,
          offer: { ...trialOffer, unit_amount: 1999 },
        }),
      ),
    );
    render(<TrialCard />);
    const popup = await cancelThroughConfirmation();
    expect(popup.textContent).toContain(
      "Pro is $19.99 / month, cancel anytime.",
    );
  });
});

describe("popup after the end has passed", () => {
  it("says access has ended instead of naming a time today", async () => {
    vi.setSystemTime(new Date(endsAt.getTime() + 5 * 60 * 1000));
    mockCancelFlow();
    render(<TrialCard />);
    const popup = await cancelThroughConfirmation();
    expect(popup.textContent).toContain(
      "Your card won't be charged. Your trial access has ended.",
    );
    expect(popup.textContent).not.toMatch(/today|Nothing changes/);
    expect(
      within(popup).getByText("Pro is $20 / month, cancel anytime."),
    ).toBeDefined();
  });
});
