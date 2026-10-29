import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import {
  getGetV1GetNotificationPreferencesMockHandler,
  getGetV1GetUserTimezoneMockHandler,
} from "@/app/api/__generated__/endpoints/auth/auth.msw";
import { Toaster } from "@/components/molecules/Toast/toaster";
import {
  ImpersonationState,
  setImpersonationViewer,
} from "@/lib/impersonation";
import { server } from "@/mocks/mock-server";
import { Flag } from "@/services/feature-flags/use-get-flag";
import {
  mockAccountDeletionAPI,
  recordDeletion,
  recordFeedback,
  rejectDeletion,
} from "@/tests/integrations/account-deletion-fixtures";
import { configureCookiebot } from "@/tests/integrations/cookiebot";
import {
  act,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";

import SettingsPreferencesPage from "../page";

const mocks = vi.hoisted(() => ({
  useGetFlag: vi.fn(),
  logOut: vi.fn(),
  replace: vi.fn(),
}));

vi.mock("@/providers/onboarding/onboarding-provider", () => ({
  default: ({ children }: { children: ReactNode }) => <>{children}</>,
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    user: {
      id: "user-1",
      email: "user@example.com",
      app_metadata: {},
      user_metadata: {},
      aud: "authenticated",
      created_at: "2026-01-01T00:00:00.000Z",
    },
    isLoggedIn: true,
    isUserLoading: false,
    logOut: mocks.logOut,
  }),
}));

vi.mock("next/navigation", async (importOriginal) => {
  const actual = await importOriginal<typeof import("next/navigation")>();
  return {
    ...actual,
    useSearchParams: () => new URLSearchParams(),
    usePathname: () => "/settings/account",
    useRouter: () => ({
      replace: mocks.replace,
      push: vi.fn(),
      back: vi.fn(),
      forward: vi.fn(),
      refresh: vi.fn(),
      prefetch: vi.fn(),
    }),
  };
});

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

type User = ReturnType<typeof userEvent.setup>;

const OPENED = { kind: "opened", surface: "settings" };

async function openFlow(user: User) {
  await user.click(
    await screen.findByRole("button", { name: "Delete account" }),
  );
  return screen.findByRole("dialog", { name: "Why are you leaving?" });
}

async function closeFlow(user: User) {
  await user.click(
    within(screen.getByRole("dialog")).getByRole("button", { name: "Close" }),
  );
  await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
}

function getEmailInput() {
  return screen.getByRole("textbox", {
    name: "Type user@example.com to confirm",
  }) as HTMLInputElement;
}

function getConfirmDeleteButton() {
  return within(
    screen.getByRole("dialog", { name: "Delete your account?" }),
  ).getByRole("button", { name: "Delete account" }) as HTMLButtonElement;
}

async function flushRequests() {
  await act(() => new Promise((resolve) => setTimeout(resolve, 20)));
}

beforeEach(() => {
  mocks.logOut.mockReset().mockResolvedValue(undefined);
  mocks.replace.mockReset();
  setAccountDeletionFlag(true);
  server.use(
    getGetV1GetNotificationPreferencesMockHandler({
      user_id: "user-1",
      email: "user@example.com",
      briefing_frequency: "WEEKLY",
      alerts_enabled: true,
      store_verdicts_enabled: true,
      daily_limit: 0,
    }),
    getGetV1GetUserTimezoneMockHandler({ timezone: "Europe/London" }),
  );
  mockAccountDeletionAPI();
});

afterEach(() => {
  vi.unstubAllEnvs();
  ImpersonationState.clear();
  setImpersonationViewer(null);
});

describe("Settings > Account delete card", () => {
  test("is not shown while the account deletion flag is off", async () => {
    setAccountDeletionFlag(false);

    render(<SettingsPreferencesPage />);

    expect(await screen.findByText("Time zone")).toBeDefined();
    expect(screen.queryByText("Delete account")).toBeNull();
  });

  test("is not shown while an admin is impersonating the user", async () => {
    ImpersonationState.set("user-1", "admin-1");
    setImpersonationViewer({ id: "admin-1", role: "admin" });

    render(<SettingsPreferencesPage />);

    expect(await screen.findByText("Time zone")).toBeDefined();
    expect(screen.queryByText("Delete account")).toBeNull();
  });

  test("is the last card, after the Cookies card", async () => {
    configureCookiebot();

    render(<SettingsPreferencesPage />);

    const button = await screen.findByRole("button", {
      name: "Delete account",
    });
    const classes = button.className.split(/\s+/);
    expect(classes).toContain("text-error-hover");
    expect(classes).not.toContain("text-foreground-emphasis");
    const card = button.closest("section");
    expect(card?.textContent).toContain(
      "Permanently delete your account and everything in it, disconnect your apps and cancel your plan.",
    );

    const sections = Array.from(document.querySelectorAll("section"));
    expect(sections.at(-1)).toBe(card);
    expect(sections.at(-2)?.textContent).toContain("Cookies");
  });

  test("opens the flow, moves focus into it and gives focus back when it closes", async () => {
    const user = userEvent.setup();
    render(<SettingsPreferencesPage />);

    const button = await screen.findByRole("button", {
      name: "Delete account",
    });
    await user.click(button);

    const dialog = await screen.findByRole("dialog", {
      name: "Why are you leaving?",
    });
    await waitFor(() =>
      expect(dialog.contains(document.activeElement)).toBe(true),
    );

    await user.keyboard("{Escape}");

    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    await waitFor(() => expect(document.activeElement).toBe(button));
  });

  test("a 200 deletes with source settings, says so and leaves through /logout", async () => {
    const requests = recordDeletion();
    const user = userEvent.setup();
    render(
      <>
        <SettingsPreferencesPage />
        <Toaster />
      </>,
    );

    await openFlow(user);
    await user.click(await screen.findByRole("button", { name: "Skip" }));
    await user.type(getEmailInput(), "user@example.com");
    await user.click(getConfirmDeleteButton());

    await waitFor(() => expect(mocks.replace).toHaveBeenCalledWith("/logout"));
    expect(requests).toEqual([
      { confirm_email: "user@example.com", source: "settings" },
    ]);
    expect(mocks.logOut).not.toHaveBeenCalled();
    expect(await screen.findByText("Your account is deleted.")).toBeDefined();
    expect(
      screen.getByRole("dialog", { name: "Delete your account?" }),
    ).toBeDefined();
  });

  test("a 502 shows the server's message and keeps the dialog open", async () => {
    const detail =
      "We couldn't cancel your plan. Nothing was deleted. Try again.";
    rejectDeletion(502, detail);
    const user = userEvent.setup();
    render(
      <>
        <SettingsPreferencesPage />
        <Toaster />
      </>,
    );

    await openFlow(user);
    await user.click(await screen.findByRole("button", { name: "Skip" }));
    await user.type(getEmailInput(), "user@example.com");
    await user.click(getConfirmDeleteButton());

    expect(await screen.findByText(detail)).toBeDefined();
    expect(
      screen.getByRole("dialog", { name: "Delete your account?" }),
    ).toBeDefined();
    await waitFor(() => expect(getConfirmDeleteButton().disabled).toBe(false));
    expect(mocks.logOut).not.toHaveBeenCalled();
    expect(mocks.replace).not.toHaveBeenCalled();
  });

  test("reopening starts over at the reason step with nothing picked or typed", async () => {
    const user = userEvent.setup();
    render(<SettingsPreferencesPage />);

    let dialog = await openFlow(user);
    await user.click(
      within(dialog).getByRole("radio", { name: "It costs too much" }),
    );
    await user.type(
      within(dialog).getByRole("textbox", { name: "Tell us more" }),
      "Too much for me",
    );
    await user.click(within(dialog).getByRole("button", { name: "Continue" }));
    await user.type(getEmailInput(), "user@");
    await closeFlow(user);

    dialog = await openFlow(user);
    expect(within(dialog).queryByRole("radio", { checked: true })).toBeNull();
    expect(within(dialog).queryByRole("note")).toBeNull();
    expect(
      (
        within(dialog).getByRole("textbox", {
          name: "Tell us more",
        }) as HTMLInputElement
      ).value,
    ).toBe("");

    await user.click(within(dialog).getByRole("button", { name: "Skip" }));
    expect(getEmailInput().value).toBe("");
    expect(getConfirmDeleteButton().disabled).toBe(true);
  });

  test("every open counts one start, even one closed with the X", async () => {
    const feedback = recordFeedback();
    const user = userEvent.setup();
    render(<SettingsPreferencesPage />);

    await openFlow(user);
    await waitFor(() =>
      expect(feedback).toEqual([{ kind: "opened", surface: "settings" }]),
    );
    await closeFlow(user);
    await openFlow(user);
    await closeFlow(user);
    await flushRequests();

    expect(feedback).toEqual([
      { kind: "opened", surface: "settings" },
      { kind: "opened", surface: "settings" },
    ]);
  });

  test("the close X keeps the account and sends no answer on either step", async () => {
    const feedback = recordFeedback();
    const requests = recordDeletion();
    const user = userEvent.setup();
    render(<SettingsPreferencesPage />);

    const dialog = await openFlow(user);
    await user.click(
      within(dialog).getByRole("radio", { name: "It costs too much" }),
    );
    await closeFlow(user);
    await flushRequests();
    expect(feedback).toEqual([OPENED]);

    await openFlow(user);
    await user.click(screen.getByRole("button", { name: "Skip" }));
    await waitFor(() => expect(feedback).toHaveLength(3));
    await user.type(getEmailInput(), "user@example.com");
    await closeFlow(user);
    await flushRequests();

    expect(requests).toEqual([]);
    expect(feedback).toEqual([
      OPENED,
      OPENED,
      { kind: "answer", reason: "skipped" },
    ]);
    expect(mocks.logOut).not.toHaveBeenCalled();
    expect(mocks.replace).not.toHaveBeenCalled();
  });

  test("every open sends its own answer, once", async () => {
    const feedback = recordFeedback();
    const user = userEvent.setup();
    render(<SettingsPreferencesPage />);

    await openFlow(user);
    await user.click(screen.getByRole("button", { name: "Skip" }));
    await closeFlow(user);
    const dialog = await openFlow(user);
    await user.click(
      within(dialog).getByRole("radio", { name: "Something else" }),
    );
    await user.click(within(dialog).getByRole("button", { name: "Continue" }));
    await flushRequests();

    expect(feedback).toEqual([
      OPENED,
      { kind: "answer", reason: "skipped" },
      OPENED,
      { kind: "answer", reason: "other" },
    ]);
  });
});
