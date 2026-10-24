import { formatShortDate } from "@/app/(platform)/settings/billing/helpers";
import { getGetSubscriptionStatusQueryKey } from "@/app/api/__generated__/endpoints/credits/credits";
import { getGetSubscriptionStatusMockHandler401 } from "@/app/api/__generated__/endpoints/credits/credits.msw";
import { getGetV1ListCredentialsMockHandler401 } from "@/app/api/__generated__/endpoints/integrations/integrations.msw";
import {
  getGetTrialsGetTrialStatusMockHandler200,
  getGetTrialsGetTrialStatusMockHandler401,
} from "@/app/api/__generated__/endpoints/trials/trials.msw";
import type { SubscriptionStatusResponse } from "@/app/api/__generated__/models/subscriptionStatusResponse";
import type { TrialStatusResponse } from "@/app/api/__generated__/models/trialStatusResponse";
import { server } from "@/mocks/mock-server";
import {
  GMAIL,
  GMAIL_WORK,
  GOOGLE_CALENDAR,
  MANAGED_CREDENTIAL,
  NOTION,
  SYSTEM_CREDENTIAL,
  deferred,
  mockAccountDeletionAPI,
  subscriptionStatus,
} from "@/tests/integrations/account-deletion-fixtures";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import {
  setTrialUser,
  trialResponse,
} from "@/tests/integrations/trial-fixtures";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { DeleteAccountFlow } from "../DeleteAccountFlow";
import type { DeletionSource } from "../helpers";

vi.mock("@/providers/onboarding/onboarding-provider", () => ({
  default: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  usePathname: () => "/settings/account",
  useSearchParams: () => new URLSearchParams(),
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    user: { id: "user-1", email: "jane@example.com", user_metadata: {} },
    logOut: vi.fn(),
  }),
}));

const REASON = {
  privacy: "I'm worried about my data or privacy",
  cost: "It costs too much",
  not_using: "I'm not using it enough",
  missing_capability: "It couldn't do what I needed",
  limits: "I kept hitting usage limits",
  other: "Something else",
} as const;

type Reason = keyof typeof REASON;

interface Panel {
  copy: string;
  action: string | null;
}

const ENDS_AT = new Date("2026-11-30T12:00:00Z");
const SWITCH_AT = new Date("2026-12-05T12:00:00Z");
const PERIOD_END_SECONDS = Date.UTC(2026, 11, 15, 12) / 1000;
const TRIAL_ENDS_AT = new Date("2026-10-20T12:00:00Z");

const BILLING = "Go to Billing";
const KEEP = "Keep my account";

const NO_APPS: Panel = {
  copy: "You haven't connected any apps, so AutoGPT can't reach your email, calendar or files.",
  action: null,
};
const PRO_CANCEL: Panel = {
  copy: "You can cancel your plan and keep your account. You won't be charged again, and your Experts and chats stay here if you come back.",
  action: BILLING,
};
const TRIAL_CANCEL: Panel = {
  copy: "You can cancel your trial instead. You won't be charged.",
  action: BILLING,
};
const NO_PLAN: Panel = {
  copy: "Your account costs nothing without a plan. Your Experts and chats stay here if you come back.",
  action: KEEP,
};
const MAX_DOWNGRADE =
  "A downgrade keeps your Experts, chats and schedules and starts when your current period ends.";
const LIMITS: Panel = {
  copy: "Max gives you 8.5x the usage of Pro.",
  action: BILLING,
};

function planEnds(date: Date | number): Panel {
  return {
    copy: `Your plan already ends on ${formatShortDate(date)}, and you won't be charged again. Your account stays after that.`,
    action: KEEP,
  };
}

function trialCancel(date: Date): Panel {
  return {
    copy: `You can cancel your trial instead. You won't be charged and you keep access until ${formatShortDate(date)}.`,
    action: BILLING,
  };
}

function trialEnds(date: Date): Panel {
  return {
    copy: `Your trial already ends on ${formatShortDate(date)} and you won't be charged.`,
    action: KEEP,
  };
}

interface Scenario {
  name: string;
  subscription: SubscriptionStatusResponse | "failed";
  trial?: TrialStatusResponse | "failed";
  cost: Panel | null;
  not_using: Panel | null;
  limits: Panel | null;
}

const SCENARIOS: Scenario[] = [
  {
    name: "Max",
    subscription: subscriptionStatus({ tier: "MAX" }),
    cost: { copy: `Pro is $50.00 / month. ${MAX_DOWNGRADE}`, action: BILLING },
    not_using: PRO_CANCEL,
    limits: null,
  },
  {
    name: "Max without a Pro price",
    subscription: subscriptionStatus({
      tier: "MAX",
      tier_costs: { MAX: 32000 },
    }),
    cost: { copy: MAX_DOWNGRADE, action: BILLING },
    not_using: PRO_CANCEL,
    limits: null,
  },
  {
    name: "Max with a pending cancel",
    subscription: subscriptionStatus({
      tier: "MAX",
      pending_tier: "NO_TIER",
      pending_tier_effective_at: ENDS_AT,
    }),
    cost: planEnds(ENDS_AT),
    not_using: planEnds(ENDS_AT),
    limits: null,
  },
  {
    name: "Max with a pending cancel dated only by the period end",
    subscription: subscriptionStatus({
      tier: "MAX",
      pending_tier: "NO_TIER",
      current_period_end: PERIOD_END_SECONDS,
    }),
    cost: planEnds(PERIOD_END_SECONDS * 1000),
    not_using: planEnds(PERIOD_END_SECONDS * 1000),
    limits: null,
  },
  {
    name: "Max with a pending downgrade to Pro",
    subscription: subscriptionStatus({
      tier: "MAX",
      pending_tier: "PRO",
      pending_tier_effective_at: SWITCH_AT,
      pending_billing_cycle: "monthly",
    }),
    cost: {
      copy: `Your switch to Pro is already scheduled for ${formatShortDate(SWITCH_AT)}.`,
      action: null,
    },
    not_using: PRO_CANCEL,
    limits: null,
  },
  {
    name: "Pro",
    subscription: subscriptionStatus({ tier: "PRO" }),
    cost: PRO_CANCEL,
    not_using: PRO_CANCEL,
    limits: LIMITS,
  },
  {
    name: "Pro with a pending cancel",
    subscription: subscriptionStatus({
      tier: "PRO",
      pending_tier: "NO_TIER",
      pending_tier_effective_at: ENDS_AT,
    }),
    cost: planEnds(ENDS_AT),
    not_using: planEnds(ENDS_AT),
    limits: null,
  },
  {
    name: "Trial",
    subscription: subscriptionStatus({ tier: "TRIAL" }),
    trial: trialResponse({
      cancel_at_period_end: false,
      ends_at: TRIAL_ENDS_AT,
    }),
    cost: trialCancel(TRIAL_ENDS_AT),
    not_using: trialCancel(TRIAL_ENDS_AT),
    limits: null,
  },
  {
    name: "Trial whose status has no end date",
    subscription: subscriptionStatus({ tier: "TRIAL" }),
    trial: trialResponse({ cancel_at_period_end: false, ends_at: null }),
    cost: TRIAL_CANCEL,
    not_using: TRIAL_CANCEL,
    limits: null,
  },
  {
    name: "Trial with a pending cancel",
    subscription: subscriptionStatus({ tier: "TRIAL" }),
    trial: trialResponse({
      cancel_at_period_end: true,
      ends_at: TRIAL_ENDS_AT,
    }),
    cost: trialEnds(TRIAL_ENDS_AT),
    not_using: trialEnds(TRIAL_ENDS_AT),
    limits: null,
  },
  {
    name: "Trial whose trial status failed to load",
    subscription: subscriptionStatus({ tier: "TRIAL" }),
    trial: "failed",
    cost: null,
    not_using: null,
    limits: null,
  },
  {
    name: "None (no plan)",
    subscription: subscriptionStatus({ tier: "NO_TIER", tier_costs: {} }),
    cost: NO_PLAN,
    not_using: NO_PLAN,
    limits: null,
  },
  {
    name: "Managed Basic",
    subscription: subscriptionStatus({ tier: "BASIC" }),
    cost: null,
    not_using: null,
    limits: null,
  },
  {
    name: "Managed Team",
    subscription: subscriptionStatus({ tier: "BUSINESS" }),
    cost: null,
    not_using: null,
    limits: null,
  },
  {
    name: "Managed Enterprise with a Stripe schedule",
    subscription: subscriptionStatus({
      tier: "ENTERPRISE",
      pending_tier: "NO_TIER",
      pending_tier_effective_at: ENDS_AT,
    }),
    cost: null,
    not_using: null,
    limits: null,
  },
  {
    name: "a failed plan query",
    subscription: "failed",
    cost: null,
    not_using: null,
    limits: null,
  },
];

function mockSubscription(subscription: Scenario["subscription"]) {
  if (subscription === "failed") {
    server.use(getGetSubscriptionStatusMockHandler401());
    return;
  }
  mockAccountDeletionAPI({ subscription });
}

function mockTrial(trial: Scenario["trial"]) {
  if (!trial) return;
  setTrialUser("user-1");
  server.use(
    trial === "failed"
      ? getGetTrialsGetTrialStatusMockHandler401()
      : getGetTrialsGetTrialStatusMockHandler200(trial),
  );
}

// A QueryClient the test can see, so "no panel" is only asserted once every
// query has answered, not while the data is still on its way.
function renderFlow(source: DeletionSource = "settings") {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  render(
    <QueryClientProvider client={queryClient}>
      <DeleteAccountFlow source={source} open onOpenChange={vi.fn()} />
    </QueryClientProvider>,
  );
  return queryClient;
}

async function settle(queryClient: QueryClient) {
  await waitFor(() => expect(queryClient.isFetching()).toBe(0));
  await act(() => new Promise((resolve) => setTimeout(resolve, 0)));
}

function pickReason(reason: Reason) {
  fireEvent.click(screen.getByRole("radio", { name: REASON[reason] }));
}

function getDialog() {
  return screen.getByRole("dialog");
}

function getPanel() {
  return within(getDialog()).queryByRole("note");
}

function expectPanel(expected: Panel | null) {
  const panel = getPanel();
  if (!expected) {
    expect(panel).toBeNull();
    return;
  }
  expect(panel).not.toBeNull();
  const content = within(panel as HTMLElement);
  expect(content.getByText(expected.copy)).toBeDefined();
  expect(content.queryAllByRole("button").map((b) => b.textContent)).toEqual(
    expected.action ? [expected.action] : [],
  );
}

beforeEach(() => {
  mockAccountDeletionAPI();
});

afterEach(() => {
  setTrialUser(null);
});

describe("DeleteAccountFlow reason panels", () => {
  test.each(SCENARIOS)(
    "every reason shows the right panel for $name",
    async (scenario) => {
      mockSubscription(scenario.subscription);
      mockTrial(scenario.trial);
      const queryClient = renderFlow();
      await settle(queryClient);

      const expected: Record<Reason, Panel | null> = {
        privacy: NO_APPS,
        cost: scenario.cost,
        not_using: scenario.not_using,
        missing_capability: null,
        limits: scenario.limits,
        other: null,
      };
      for (const reason of Object.keys(REASON) as Reason[]) {
        pickReason(reason);
        expect(
          screen.getByRole("radio", { name: REASON[reason], checked: true }),
        ).toBeDefined();
        expectPanel(expected[reason]);
      }
    },
  );

  test("the panel sits right under the picked reason, one at a time", async () => {
    const queryClient = renderFlow();
    await settle(queryClient);

    pickReason("cost");

    const panel = getPanel() as HTMLElement;
    const pickedRow = screen.getByRole("radio", { name: REASON.cost });
    const nextRow = screen.getByRole("radio", { name: REASON.not_using });
    expect(panel.previousElementSibling?.contains(pickedRow)).toBe(true);
    expect(panel.nextElementSibling?.contains(nextRow)).toBe(true);

    pickReason("limits");

    expect(screen.getAllByRole("note")).toHaveLength(1);
    expect(screen.queryByText(PRO_CANCEL.copy)).toBeNull();
    expectPanel(LIMITS);
  });

  test("announces each panel through one status region that is there before any reason is picked", async () => {
    const queryClient = renderFlow();
    await settle(queryClient);
    const status = within(getDialog()).getByRole("status");
    expect(status.textContent).toBe("");
    expect(status.closest('[role="radiogroup"]')).toBeNull();

    pickReason("cost");
    expect(within(getDialog()).getByRole("status")).toBe(status);
    expect(status.textContent).toBe(PRO_CANCEL.copy);

    pickReason("limits");
    expect(within(getDialog()).getByRole("status")).toBe(status);
    expect(status.textContent).toBe(LIMITS.copy);

    pickReason("missing_capability");
    expect(within(getDialog()).getByRole("status")).toBe(status);
    expect(status.textContent).toBe("");
  });

  test("the panel draws a Hugeicons icon, not a lucide one", async () => {
    const queryClient = renderFlow();
    await settle(queryClient);

    pickReason("cost");

    const icon = (getPanel() as HTMLElement).querySelector(":scope > svg");
    expect(icon).not.toBeNull();
    expect(icon?.classList.contains("lucide")).toBe(false);
  });

  test("missing capability and something else change the text box placeholder instead", async () => {
    const queryClient = renderFlow();
    await settle(queryClient);
    const textBox = screen.getByRole("textbox", { name: "Tell us more" });
    expect(textBox.getAttribute("placeholder")).toBe(
      "Anything else we should know? (optional)",
    );

    pickReason("missing_capability");
    expect(textBox.getAttribute("placeholder")).toBe(
      "What were you trying to get done?",
    );

    pickReason("other");
    expect(textBox.getAttribute("placeholder")).toBe("Tell us more (optional)");

    pickReason("cost");
    expect(textBox.getAttribute("placeholder")).toBe(
      "Anything else we should know? (optional)",
    );
  });

  test("shows no edit pencils next to the reasons", async () => {
    const queryClient = renderFlow();
    await settle(queryClient);
    pickReason("cost");

    const dialog = getDialog();
    expect(within(dialog).queryAllByRole("button", { name: /^Edit / })).toEqual(
      [],
    );
    expect(dialog.querySelector('[title="Edit before sending"]')).toBeNull();
    for (const radio of screen.getAllByRole("radio")) {
      const classes = radio.className.split(/\s+/);
      expect(classes).toContain("pr-4");
      expect(classes).not.toContain("pr-12");
    }
  });
});

describe("DeleteAccountFlow privacy panel", () => {
  test("from Settings it names the connected apps and links to them, leaving out managed and system ones", async () => {
    mockAccountDeletionAPI({
      credentials: [
        GMAIL,
        MANAGED_CREDENTIAL,
        GOOGLE_CALENDAR,
        SYSTEM_CREDENTIAL,
      ],
    });
    const queryClient = renderFlow("settings");
    await settle(queryClient);

    pickReason("privacy");

    expectPanel({
      copy: "You can disconnect Gmail and Google Calendar without deleting your account. Our access ends as soon as you do.",
      action: "Manage connected apps",
    });
  });

  test("from the paywall it offers to disconnect them in place, since Settings is behind the paywall", async () => {
    mockAccountDeletionAPI({ credentials: [GMAIL, GOOGLE_CALENDAR] });
    const queryClient = renderFlow("paywall");
    await settle(queryClient);

    pickReason("privacy");

    expectPanel({
      copy: "You can disconnect Gmail and Google Calendar without deleting your account. Our access ends as soon as you do.",
      action: "Disconnect all apps",
    });
  });

  test("names each app once and counts the rest", async () => {
    mockAccountDeletionAPI({
      credentials: [GMAIL, GMAIL_WORK, GOOGLE_CALENDAR, NOTION],
    });
    const queryClient = renderFlow();
    await settle(queryClient);

    pickReason("privacy");

    expectPanel({
      copy: "You can disconnect Gmail, Google Calendar and 1 more app without deleting your account. Our access ends as soon as you do.",
      action: "Manage connected apps",
    });
  });

  test("with only managed and system credentials it says no apps are connected", async () => {
    mockAccountDeletionAPI({
      credentials: [MANAGED_CREDENTIAL, SYSTEM_CREDENTIAL],
    });
    const queryClient = renderFlow();
    await settle(queryClient);

    pickReason("privacy");

    expectPanel(NO_APPS);
  });

  test("shows no panel when the credentials query fails, while the plan panels still work", async () => {
    server.use(getGetV1ListCredentialsMockHandler401());
    const queryClient = renderFlow();
    await settle(queryClient);

    pickReason("privacy");
    expectPanel(null);

    pickReason("cost");
    expectPanel(PRO_CANCEL);
  });

  test("shows no panel and no spinner while the credentials load, then the panel", async () => {
    const credentials = deferred();
    server.use(
      http.get("/api/proxy/api/integrations/credentials", async () => {
        await credentials.promise;
        return HttpResponse.json([GMAIL]);
      }),
    );
    renderFlow();

    pickReason("privacy");
    expectPanel(null);
    expect(within(getDialog()).queryByRole("progressbar")).toBeNull();

    credentials.resolve();

    await waitFor(() =>
      expectPanel({
        copy: "You can disconnect Gmail without deleting your account. Our access ends as soon as you do.",
        action: "Manage connected apps",
      }),
    );
  });
});

describe("DeleteAccountFlow trial panels while the trial status loads", () => {
  test("show nothing until the trial status arrives", async () => {
    const trialStatus = deferred();
    setTrialUser("user-1");
    mockAccountDeletionAPI({
      subscription: subscriptionStatus({ tier: "TRIAL" }),
    });
    server.use(
      http.get("/api/proxy/api/credits/trial", async () => {
        await trialStatus.promise;
        return HttpResponse.json(
          trialResponse({
            cancel_at_period_end: false,
            ends_at: TRIAL_ENDS_AT,
          }),
        );
      }),
    );
    const queryClient = renderFlow();
    await waitFor(() =>
      expect(
        queryClient.getQueryState(getGetSubscriptionStatusQueryKey())?.status,
      ).toBe("success"),
    );
    await act(() => new Promise((resolve) => setTimeout(resolve, 0)));

    pickReason("cost");
    expectPanel(null);

    trialStatus.resolve();

    await waitFor(() => expectPanel(trialCancel(TRIAL_ENDS_AT)));
  });
});
