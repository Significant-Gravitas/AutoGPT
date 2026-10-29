import {
  ImpersonationState,
  setImpersonationViewer,
} from "@/lib/impersonation";
import { Flag } from "@/services/feature-flags/use-get-flag";
import {
  mockAccountDeletionAPI,
  recordDeletion,
  recordFeedback,
} from "@/tests/integrations/account-deletion-fixtures";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import { PaywallModal } from "../PaywallModal";

const mocks = vi.hoisted(() => ({
  useGetFlag: vi.fn(),
  logOut: vi.fn(),
  replace: vi.fn(),
}));

vi.mock("@/app/api/__generated__/endpoints/credits/credits", () => ({
  useGetSubscriptionStatus: () => ({
    data: { tier: "NO_TIER", tier_costs: { PRO: 5000 } },
    isLoading: false,
    isFetching: false,
    refetch: vi.fn(),
  }),
  useUpdateSubscriptionTier: () => ({ mutateAsync: vi.fn(), isPending: false }),
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    user: { id: "user-1", email: "jane@example.com", user_metadata: {} },
    isLoggedIn: true,
    validateSession: vi.fn().mockResolvedValue(true),
    logOut: mocks.logOut,
  }),
}));

const viewport = vi.hoisted(() => ({ isLarge: true }));

vi.mock("@/lib/hooks/useBreakpoint", () => ({
  useBreakpoint: () => (viewport.isLarge ? "lg" : "sm"),
  isLargeScreen: (breakpoint: string) => breakpoint === "lg",
}));

vi.mock("@/providers/onboarding/onboarding-provider", () => ({
  default: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(),
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
  return { ...actual, useGetFlag: mocks.useGetFlag };
});

function setAccountDeletionFlag(enabled: boolean) {
  mocks.useGetFlag.mockImplementation(
    (flag: Flag) => enabled && flag === Flag.SELF_SERVE_ACCOUNT_DELETION,
  );
}

async function openDeleteFlow() {
  const user = userEvent.setup();
  render(<PaywallModal />);
  const paywall = screen.getByRole("dialog", { name: "Choose your plan" });
  const button = within(paywall).getByRole("button", {
    name: "Delete account",
  });
  await user.click(button);
  const flow = await screen.findByRole("dialog", {
    name: "Why are you leaving?",
  });
  return { user, paywall, button, flow };
}

beforeEach(() => {
  viewport.isLarge = true;
  mocks.logOut.mockReset().mockResolvedValue(undefined);
  mocks.replace.mockReset();
  setAccountDeletionFlag(true);
  mockAccountDeletionAPI();
});

afterEach(() => {
  ImpersonationState.clear();
  setImpersonationViewer(null);
});

describe("PaywallModal — delete account", () => {
  test("shows no Delete account button while the flag is off", () => {
    setAccountDeletionFlag(false);

    render(<PaywallModal />);

    expect(screen.getByRole("button", { name: "Log out" })).toBeDefined();
    expect(screen.queryByRole("button", { name: "Delete account" })).toBeNull();
  });

  test("shows no Delete account button while an admin is impersonating the user", () => {
    ImpersonationState.set("user-1", "admin-1");
    setImpersonationViewer({ id: "admin-1", role: "admin" });

    render(<PaywallModal />);

    expect(screen.getByRole("button", { name: "Log out" })).toBeDefined();
    expect(screen.queryByRole("button", { name: "Delete account" })).toBeNull();
  });

  test("opening the flow counts one start from the paywall", async () => {
    const feedback = recordFeedback();

    await openDeleteFlow();

    await waitFor(() =>
      expect(feedback).toEqual([{ kind: "opened", surface: "paywall" }]),
    );
  });

  test("sits next to Log out in the same row and looks the same", () => {
    render(<PaywallModal />);

    const logOut = screen.getByRole("button", { name: "Log out" });
    const deleteAccount = screen.getByRole("button", {
      name: "Delete account",
    });
    expect(deleteAccount.parentElement).toBe(logOut.parentElement);
    expect(deleteAccount.className).toBe(logOut.className);
  });

  test("the paywall still opens with focus on its Close button, with Delete account after Log out", async () => {
    render(<PaywallModal />);

    const close = screen.getByRole("button", { name: "Close" });
    await waitFor(() => expect(document.activeElement).toBe(close));
    const logOut = screen.getByRole("button", { name: "Log out" });
    expect(logOut.nextElementSibling).toBe(
      screen.getByRole("button", { name: "Delete account" }),
    );
  });

  test("Log out still goes to the shared logout page", () => {
    render(<PaywallModal />);

    fireEvent.click(screen.getByRole("button", { name: "Log out" }));

    expect(mocks.replace).toHaveBeenCalledWith("/logout");
  });

  test("opens above the paywall and takes focus", async () => {
    const { paywall, flow } = await openDeleteFlow();

    expect(
      paywall.compareDocumentPosition(flow) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    expect(screen.getAllByRole("dialog")).toEqual([flow]);
    await waitFor(() => expect(paywall.style.pointerEvents).toBe("none"));
    await waitFor(() =>
      expect(flow.contains(document.activeElement)).toBe(true),
    );
  });

  test("below 1024px the flow opens as a drawer and still takes focus", async () => {
    viewport.isLarge = false;
    const user = userEvent.setup();
    render(<PaywallModal />);
    // happy-dom can't run a pointer click inside a vaul drawer, so open it
    // from the keyboard.
    screen.getByRole("button", { name: "Delete account" }).focus();
    await user.keyboard("{Enter}");

    const flow = await screen.findByRole("dialog", {
      name: "Why are you leaving?",
    });
    await waitFor(() =>
      expect(flow.contains(document.activeElement)).toBe(true),
    );
  });

  test("Escape closes only the delete flow and gives focus back to its button", async () => {
    const { user, paywall, button } = await openDeleteFlow();

    await user.keyboard("{Escape}");

    await waitFor(() =>
      expect(
        screen.queryByRole("dialog", { name: "Why are you leaving?" }),
      ).toBeNull(),
    );
    expect(screen.getByRole("dialog", { name: "Choose your plan" })).toBe(
      paywall,
    );
    await waitFor(() => expect(document.activeElement).toBe(button));
  });

  test("Keep account on the confirmation gives focus back to its button", async () => {
    const { user, button } = await openDeleteFlow();

    await user.click(screen.getByRole("button", { name: "Skip" }));
    await user.click(
      await screen.findByRole("button", { name: "Keep account" }),
    );

    await waitFor(() =>
      expect(
        screen.queryByRole("dialog", { name: "Delete your account?" }),
      ).toBeNull(),
    );
    await waitFor(() => expect(document.activeElement).toBe(button));
  });

  test("deletes with source paywall and leaves through /logout", async () => {
    const bodies = recordDeletion();
    const { user } = await openDeleteFlow();

    await user.click(screen.getByRole("button", { name: "Skip" }));
    await user.type(
      await screen.findByRole("textbox", {
        name: "Type jane@example.com to confirm",
      }),
      "jane@example.com",
    );
    const flow = screen.getByRole("dialog", { name: "Delete your account?" });
    await user.click(
      within(flow).getByRole("button", { name: "Delete account" }),
    );

    await waitFor(() => expect(mocks.replace).toHaveBeenCalledWith("/logout"));
    expect(bodies).toEqual([
      { confirm_email: "jane@example.com", source: "paywall" },
    ]);
    expect(mocks.logOut).not.toHaveBeenCalled();
  });
});
