import { getGetTrialsGetTrialStatusMockHandler200 } from "@/app/api/__generated__/endpoints/trials/trials.msw";
import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import type { DeletionFeedbackRequest } from "@/app/api/__generated__/models/deletionFeedbackRequest";
import type { SubscriptionStatusResponse } from "@/app/api/__generated__/models/subscriptionStatusResponse";
import { server } from "@/mocks/mock-server";
import {
  GMAIL,
  MANAGED_CREDENTIAL,
  NOTION,
  deferred,
  fakeCredentialStore,
  mockAccountDeletionAPI,
  recordFeedback,
  subscriptionStatus,
} from "@/tests/integrations/account-deletion-fixtures";
import {
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
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { DeleteAccountFlow } from "../DeleteAccountFlow";
import type { DeletionSource } from "../helpers";

vi.mock("@/providers/onboarding/onboarding-provider", () => ({
  default: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

const mocks = vi.hoisted(() => ({ push: vi.fn(), replace: vi.fn() }));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: mocks.push, replace: mocks.replace }),
  usePathname: () => "/settings/account",
  useSearchParams: () => new URLSearchParams(),
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    user: { id: "user-1", email: "jane@example.com", user_metadata: {} },
    logOut: vi.fn(),
  }),
}));

const PRIVACY = "I'm worried about my data or privacy";

function renderFlow(source: DeletionSource) {
  const onOpenChange = vi.fn();
  render(
    <DeleteAccountFlow source={source} open onOpenChange={onOpenChange} />,
  );
  return onOpenChange;
}

function getPanel() {
  return within(screen.getByRole("dialog")).getByRole("note");
}

beforeEach(() => {
  mocks.push.mockReset();
  mocks.replace.mockReset();
});

afterEach(() => {
  setTrialUser(null);
});

interface ActionCase {
  name: string;
  subscription: SubscriptionStatusResponse;
  credentials?: CredentialsMetaResponse[];
  trialCancels?: boolean;
  reason: string;
  button: string;
  body: DeletionFeedbackRequest;
  route: string | null;
}

const ACTION_CASES: ActionCase[] = [
  {
    name: "Manage connected apps",
    subscription: subscriptionStatus({ tier: "PRO" }),
    credentials: [GMAIL],
    reason: PRIVACY,
    button: "Manage connected apps",
    body: {
      kind: "alternative_clicked",
      reason: "privacy",
      alternative: "manage_apps",
    },
    route: "/settings/integrations",
  },
  {
    name: "Go to Billing for a Max plan that costs too much",
    subscription: subscriptionStatus({ tier: "MAX" }),
    reason: "It costs too much",
    button: "Go to Billing",
    body: {
      kind: "alternative_clicked",
      reason: "cost",
      alternative: "billing",
    },
    route: "/settings/billing",
  },
  {
    name: "Go to Billing for a Max plan that goes unused",
    subscription: subscriptionStatus({ tier: "MAX" }),
    reason: "I'm not using it enough",
    button: "Go to Billing",
    body: {
      kind: "alternative_clicked",
      reason: "not_using",
      alternative: "billing",
    },
    route: "/settings/billing",
  },
  {
    name: "Go to Billing for Pro limits",
    subscription: subscriptionStatus({ tier: "PRO" }),
    reason: "I kept hitting usage limits",
    button: "Go to Billing",
    body: {
      kind: "alternative_clicked",
      reason: "limits",
      alternative: "billing",
    },
    route: "/settings/billing",
  },
  {
    name: "Go to Billing for a trial",
    subscription: subscriptionStatus({ tier: "TRIAL" }),
    trialCancels: false,
    reason: "It costs too much",
    button: "Go to Billing",
    body: {
      kind: "alternative_clicked",
      reason: "cost",
      alternative: "billing",
    },
    route: "/settings/billing",
  },
  {
    name: "Keep my account without a plan",
    subscription: subscriptionStatus({ tier: "NO_TIER", tier_costs: {} }),
    reason: "It costs too much",
    button: "Keep my account",
    body: {
      kind: "alternative_clicked",
      reason: "cost",
      alternative: "keep_account",
    },
    route: null,
  },
  {
    name: "Keep my account when a Pro plan already ends",
    subscription: subscriptionStatus({
      tier: "PRO",
      pending_tier: "NO_TIER",
      pending_tier_effective_at: new Date("2026-11-30T12:00:00Z"),
    }),
    reason: "I'm not using it enough",
    button: "Keep my account",
    body: {
      kind: "alternative_clicked",
      reason: "not_using",
      alternative: "keep_account",
    },
    route: null,
  },
  {
    name: "Keep my account when a trial already ends",
    subscription: subscriptionStatus({ tier: "TRIAL" }),
    trialCancels: true,
    reason: "It costs too much",
    button: "Keep my account",
    body: {
      kind: "alternative_clicked",
      reason: "cost",
      alternative: "keep_account",
    },
    route: null,
  },
];

describe("DeleteAccountFlow panel buttons", () => {
  test.each(ACTION_CASES)(
    "$name records the click, closes and goes to the right place",
    async (scenario) => {
      mockAccountDeletionAPI({
        subscription: scenario.subscription,
        credentials: scenario.credentials,
      });
      if (scenario.trialCancels !== undefined) {
        setTrialUser("user-1");
        server.use(
          getGetTrialsGetTrialStatusMockHandler200(
            trialResponse({ cancel_at_period_end: scenario.trialCancels }),
          ),
        );
      }
      const feedback = recordFeedback();
      const onOpenChange = renderFlow("settings");

      fireEvent.click(screen.getByRole("radio", { name: scenario.reason }));
      fireEvent.click(
        await within(screen.getByRole("dialog")).findByRole("button", {
          name: scenario.button,
        }),
      );

      expect(onOpenChange.mock.calls).toEqual([[false]]);
      if (scenario.route) {
        expect(mocks.push.mock.calls).toEqual([[scenario.route]]);
      } else {
        expect(mocks.push).not.toHaveBeenCalled();
      }
      expect(mocks.replace).not.toHaveBeenCalled();
      await waitFor(() => expect(feedback).toEqual([scenario.body]));
    },
  );
});

describe("DeleteAccountFlow Disconnect all apps", () => {
  test("disconnects every app with force, shows the button busy, then says they are disconnected", async () => {
    mockAccountDeletionAPI();
    const gate = deferred();
    const store = fakeCredentialStore([GMAIL, MANAGED_CREDENTIAL, NOTION], {
      gate: gate.promise,
    });
    const feedback = recordFeedback();
    const onOpenChange = renderFlow("paywall");

    fireEvent.click(screen.getByRole("radio", { name: PRIVACY }));
    fireEvent.click(
      await screen.findByRole("button", { name: "Disconnect all apps" }),
    );

    const busy = within(getPanel()).getByRole("button", {
      name: "Disconnect all apps",
    }) as HTMLButtonElement;
    expect(busy.disabled).toBe(true);
    await waitFor(() => expect(store.deleteURLs).toHaveLength(2));

    gate.resolve();

    expect(
      await within(getPanel()).findByText("Gmail and Notion are disconnected."),
    ).toBeDefined();
    expect(within(getPanel()).queryByRole("button")).toBeNull();
    expect([...store.deleteURLs].sort()).toEqual([
      "/api/proxy/api/integrations/google/credentials/cred-gmail?force=true",
      "/api/proxy/api/integrations/notion/credentials/cred-notion?force=true",
    ]);
    expect(onOpenChange).not.toHaveBeenCalled();
    expect(mocks.push).not.toHaveBeenCalled();
    expect(feedback).toEqual([
      {
        kind: "alternative_clicked",
        reason: "privacy",
        alternative: "disconnect_all",
      },
    ]);
  });

  test("names only the apps that failed, keeps the button ready, and a retry disconnects just those", async () => {
    mockAccountDeletionAPI();
    const store = fakeCredentialStore([GMAIL, NOTION], {
      failing: [NOTION.id],
    });
    const feedback = recordFeedback();
    const onOpenChange = renderFlow("paywall");

    fireEvent.click(screen.getByRole("radio", { name: PRIVACY }));
    fireEvent.click(
      await screen.findByRole("button", { name: "Disconnect all apps" }),
    );

    expect(
      await within(getPanel()).findByText(
        "Couldn't disconnect Notion. Try again.",
      ),
    ).toBeDefined();
    const retry = within(getPanel()).getByRole("button", {
      name: "Disconnect all apps",
    }) as HTMLButtonElement;
    expect(retry.disabled).toBe(false);
    expect([...store.deleteURLs].sort()).toEqual([
      "/api/proxy/api/integrations/google/credentials/cred-gmail?force=true",
      "/api/proxy/api/integrations/notion/credentials/cred-notion?force=true",
    ]);

    store.stopFailing(NOTION.id);
    fireEvent.click(retry);

    expect(
      await within(getPanel()).findByText("Notion is disconnected."),
    ).toBeDefined();
    expect(store.deleteURLs.slice(2)).toEqual([
      "/api/proxy/api/integrations/notion/credentials/cred-notion?force=true",
    ]);
    expect(onOpenChange).not.toHaveBeenCalled();
    await waitFor(() => expect(feedback).toHaveLength(2));
    expect(feedback).toEqual([
      {
        kind: "alternative_clicked",
        reason: "privacy",
        alternative: "disconnect_all",
      },
      {
        kind: "alternative_clicked",
        reason: "privacy",
        alternative: "disconnect_all",
      },
    ]);
  });

  test("when every app fails it names them all and offers the button again", async () => {
    mockAccountDeletionAPI();
    fakeCredentialStore([GMAIL, NOTION], { failing: [GMAIL.id, NOTION.id] });
    renderFlow("paywall");

    fireEvent.click(screen.getByRole("radio", { name: PRIVACY }));
    fireEvent.click(
      await screen.findByRole("button", { name: "Disconnect all apps" }),
    );

    expect(
      await within(getPanel()).findByText(
        "Couldn't disconnect Gmail and Notion. Try again.",
      ),
    ).toBeDefined();
    const retry = within(getPanel()).getByRole("button", {
      name: "Disconnect all apps",
    }) as HTMLButtonElement;
    expect(retry.disabled).toBe(false);
  });

  test("keeps keyboard focus in the panel when its button goes away", async () => {
    mockAccountDeletionAPI();
    fakeCredentialStore([GMAIL, NOTION]);
    const user = userEvent.setup();
    renderFlow("paywall");
    fireEvent.click(screen.getByRole("radio", { name: PRIVACY }));
    const button = await screen.findByRole("button", {
      name: "Disconnect all apps",
    });
    button.focus();

    await user.keyboard("{Enter}");

    await within(getPanel()).findByText("Gmail and Notion are disconnected.");
    expect(document.activeElement).toBe(getPanel());
    await user.tab();
    expect(document.activeElement).toBe(
      screen.getByRole("textbox", { name: "Tell us more" }),
    );
  });
});
