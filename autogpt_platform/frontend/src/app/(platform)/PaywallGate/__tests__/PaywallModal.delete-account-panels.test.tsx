import { Flag } from "@/services/feature-flags/use-get-flag";
import {
  GMAIL,
  fakeCredentialStore,
  mockAccountDeletionAPI,
  recordFeedback,
  subscriptionStatus,
} from "@/tests/integrations/account-deletion-fixtures";
import {
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, test, vi } from "vitest";

import { PaywallModal } from "../PaywallModal";

const mocks = vi.hoisted(() => ({
  push: vi.fn(),
  replace: vi.fn(),
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    user: { id: "user-1", email: "jane@example.com", user_metadata: {} },
    isLoggedIn: true,
    validateSession: vi.fn().mockResolvedValue(true),
    logOut: vi.fn(),
  }),
}));

vi.mock("@/providers/onboarding/onboarding-provider", () => ({
  default: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: mocks.push,
    replace: mocks.replace,
    prefetch: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
    refresh: vi.fn(),
  }),
  usePathname: () => "/copilot",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
}));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: Flag) => flag === Flag.SELF_SERVE_ACCOUNT_DELETION,
  };
});

async function openDeleteFlow() {
  const user = userEvent.setup();
  render(<PaywallModal />);
  const paywall = screen.getByRole("dialog", { name: "Choose your plan" });
  await user.click(
    within(paywall).getByRole("button", { name: "Delete account" }),
  );
  const flow = await screen.findByRole("dialog", {
    name: "Why are you leaving?",
  });
  return { user, paywall, flow };
}

beforeEach(() => {
  mocks.push.mockReset();
  mocks.replace.mockReset();
  mockAccountDeletionAPI({
    subscription: subscriptionStatus({
      tier: "NO_TIER",
      tier_costs: { PRO: 5000, MAX: 32000 },
    }),
  });
});

describe("PaywallModal delete account flow for a user without a plan", () => {
  test("tells them the account costs nothing, and Keep my account closes only the flow", async () => {
    const feedback = recordFeedback();
    const { user, paywall, flow } = await openDeleteFlow();

    await user.click(
      within(flow).getByRole("radio", { name: "It costs too much" }),
    );
    const panel = await within(flow).findByRole("note");
    expect(panel.textContent).toContain(
      "Your account costs nothing without a plan. Your Experts and chats stay here if you come back.",
    );
    await user.click(
      within(panel).getByRole("button", { name: "Keep my account" }),
    );

    await waitFor(() =>
      expect(
        screen.queryByRole("dialog", { name: "Why are you leaving?" }),
      ).toBeNull(),
    );
    expect(screen.getByRole("dialog", { name: "Choose your plan" })).toBe(
      paywall,
    );
    expect(mocks.push).not.toHaveBeenCalled();
    expect(mocks.replace).not.toHaveBeenCalled();
    await waitFor(() =>
      expect(feedback).toEqual([
        { kind: "opened", surface: "paywall" },
        {
          kind: "alternative_clicked",
          reason: "cost",
          alternative: "keep_account",
        },
      ]),
    );
  });

  test("offers no plan panels for leaving because of usage or limits", async () => {
    const { user, flow } = await openDeleteFlow();

    await user.click(
      within(flow).getByRole("radio", { name: "I'm not using it enough" }),
    );
    expect((await within(flow).findByRole("note")).textContent).toContain(
      "Your account costs nothing without a plan. Your Experts and chats stay here if you come back.",
    );

    await user.click(
      within(flow).getByRole("radio", { name: "I kept hitting usage limits" }),
    );
    expect(within(flow).queryByRole("note")).toBeNull();
  });

  test("privacy disconnects the apps in place instead of linking to Settings behind the paywall", async () => {
    const store = fakeCredentialStore([GMAIL]);
    const { user, flow } = await openDeleteFlow();

    await user.click(
      within(flow).getByRole("radio", {
        name: "I'm worried about my data or privacy",
      }),
    );
    const panel = await within(flow).findByRole("note");
    expect(
      within(panel).queryByRole("button", { name: "Manage connected apps" }),
    ).toBeNull();
    await user.click(
      within(panel).getByRole("button", { name: "Disconnect all apps" }),
    );

    expect(
      await within(panel).findByText("Gmail is disconnected."),
    ).toBeDefined();
    expect(store.deleteURLs).toEqual([
      "/api/proxy/api/integrations/google/credentials/cred-gmail?force=true",
    ]);
    expect(screen.getByRole("dialog", { name: "Why are you leaving?" })).toBe(
      flow,
    );
    expect(mocks.push).not.toHaveBeenCalled();
  });
});
