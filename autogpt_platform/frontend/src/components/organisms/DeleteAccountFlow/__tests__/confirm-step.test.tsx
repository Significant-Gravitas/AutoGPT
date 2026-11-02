import { server } from "@/mocks/mock-server";
import {
  CLOSED_TOAST,
  GMAIL,
  GOOGLE_CALENDAR,
  NOTION,
  deferred,
  mockAccountDeletionAPI,
  recordDeletion,
  rejectDeletion,
  subscriptionStatus,
} from "@/tests/integrations/account-deletion-fixtures";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { useState } from "react";
import { beforeEach, describe, expect, test, vi } from "vitest";
import { DeleteAccountFlow } from "../DeleteAccountFlow";

vi.mock("@/providers/onboarding/onboarding-provider", () => ({
  default: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

const mocks = vi.hoisted(() => ({
  logOut: vi.fn(),
  replace: vi.fn(),
  toast: vi.fn(),
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    user: { id: "user-1", email: "jane@example.com", user_metadata: {} },
    logOut: mocks.logOut,
  }),
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: mocks.replace }),
  usePathname: () => "/settings/account",
  useSearchParams: () => new URLSearchParams(),
}));

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: mocks.toast,
  useToast: () => ({ toast: mocks.toast }),
}));

const EMAIL_LABEL = "Type jane@example.com to confirm";
const DELETION_PATH = "/api/proxy/api/account/deletion";

beforeEach(() => {
  mocks.logOut.mockReset().mockResolvedValue(undefined);
  mocks.replace.mockReset();
  mocks.toast.mockReset();
  mockAccountDeletionAPI();
});

function openConfirmStep() {
  render(<DeleteAccountFlow source="settings" open onOpenChange={() => {}} />);
  fireEvent.click(screen.getByRole("button", { name: "Skip" }));
}

function openHostedConfirmStep() {
  const onOpenChange = vi.fn();
  function Host() {
    const [open, setOpen] = useState(true);
    return (
      <DeleteAccountFlow
        source="settings"
        open={open}
        onOpenChange={(next) => {
          onOpenChange(next);
          setOpen(next);
        }}
      />
    );
  }
  render(<Host />);
  fireEvent.click(screen.getByRole("button", { name: "Skip" }));
  return onOpenChange;
}

function gateDeletion(response: () => Response) {
  const gate = deferred();
  server.use(
    http.post(DELETION_PATH, async () => {
      await gate.promise;
      return response();
    }),
  );
  return gate;
}

function getKeepButton() {
  return within(screen.getByRole("dialog")).getByRole("button", {
    name: "Keep account",
  }) as HTMLButtonElement;
}

function pressEscape() {
  fireEvent.keyDown(screen.getByRole("dialog"), { key: "Escape" });
}

function getEmailInput() {
  return screen.getByRole("textbox", { name: EMAIL_LABEL }) as HTMLInputElement;
}

function getDeleteButton() {
  return within(screen.getByRole("dialog")).getByRole("button", {
    name: "Delete account",
  }) as HTMLButtonElement;
}

function typeEmail(value: string) {
  fireEvent.change(getEmailInput(), { target: { value } });
}

describe("DeleteAccountFlow confirm step", () => {
  test("Skip moves to the confirmation card", async () => {
    mockAccountDeletionAPI({
      subscription: subscriptionStatus({ tier: "PRO" }),
    });
    openConfirmStep();

    const dialog = screen.getByRole("dialog");
    expect(dialog.textContent).toContain("Delete your account?");
    expect(dialog.textContent).toContain(
      "This closes your AutoGPT account now and erases it after 14 days.",
    );
    expect(dialog.textContent).toContain(
      "Your chats, Experts, workflows, files and memory are erased after 14 days. Your share links stop working and your Marketplace reviews are removed now.",
    );
    expect(
      await screen.findByText(
        "Your Pro plan ends now. The rest of this billing period isn't refunded.",
      ),
    ).toBeDefined();
    expect(dialog.textContent).toContain(
      "You'll be signed out right away. Changed your mind? Sign in within 14 days and choose Keep my account. Your plan, apps, API keys, schedules, share links, Marketplace listings and reviews don't come back, and notification emails stay off until you switch them back on.",
    );
  });

  test("lists the apps, the plan and the Marketplace listings when they apply", async () => {
    mockAccountDeletionAPI({
      subscription: subscriptionStatus({ tier: "MAX" }),
      credentials: [GMAIL, GOOGLE_CALENDAR, NOTION],
      hasListings: true,
    });
    openConfirmStep();

    const list = await screen.findByRole("list");
    await waitFor(() =>
      expect(
        within(list)
          .getAllByRole("listitem")
          .map((item) => item.textContent),
      ).toEqual([
        "Your chats, Experts, workflows, files and memory are erased after 14 days. Your share links stop working and your Marketplace reviews are removed now.",
        "Connected apps like Gmail, Google Calendar and others are disconnected, and our access to them is revoked.",
        "Your Max plan ends now. The rest of this billing period isn't refunded.",
        "Your published workflows are removed from the Marketplace.",
      ]),
    );
  });

  test("a trial ends without a refund line", async () => {
    mockAccountDeletionAPI({
      subscription: subscriptionStatus({ tier: "TRIAL" }),
    });
    openConfirmStep();

    expect(await screen.findByText("Your trial ends now.")).toBeDefined();
    expect(screen.getByRole("dialog").textContent).not.toContain("refunded");
  });

  test.each([
    [
      "skills only",
      { hasSkillListings: true },
      "Your published skills are removed from the Marketplace.",
    ],
    [
      "workflows and skills",
      { hasListings: true, hasSkillListings: true },
      "Your published workflows and skills are removed from the Marketplace.",
    ],
  ])("names %s in the Marketplace line", async (_, listings, line) => {
    mockAccountDeletionAPI({
      subscription: subscriptionStatus({ tier: "NO_TIER" }),
      ...listings,
    });
    openConfirmStep();

    expect(await screen.findByText(line)).toBeDefined();
  });

  test.each([
    ["nothing", ""],
    ["spaces", "   "],
    ["a shorter address", "jane@example.co"],
    ["a longer address", "jane@example.com.au"],
    ["a space inside", "jane @example.com"],
  ])("Delete account stays disabled for %s", (_, typed) => {
    openConfirmStep();

    typeEmail(typed);

    expect(getDeleteButton().disabled).toBe(true);
  });

  test.each([
    ["the exact address", "jane@example.com"],
    ["another case with spaces around it", "  JANE@Example.COM "],
    ["a trailing tab", "jane@example.com\t"],
  ])("Delete account turns on for %s", (_, typed) => {
    openConfirmStep();

    typeEmail(typed);

    expect(getDeleteButton().disabled).toBe(false);
  });

  test("submitting the form with a wrong email sends nothing", async () => {
    const requests = recordDeletion();
    openConfirmStep();
    typeEmail("jane@example.co");

    fireEvent.submit(getEmailInput().form as HTMLFormElement);

    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(requests).toEqual([]);
  });

  test("Enter with the right email deletes with the trimmed email and the source", async () => {
    const requests = recordDeletion();
    const user = userEvent.setup();
    openConfirmStep();

    await user.type(getEmailInput(), "  Jane@Example.com {Enter}");

    await waitFor(() =>
      expect(requests).toEqual([
        { confirm_email: "Jane@Example.com", source: "settings" },
      ]),
    );
  });

  test("while the request runs the button is busy, the email locked and nothing closes the dialog", async () => {
    const gate = gateDeletion(() => HttpResponse.json({ status: "scheduled" }));
    const onOpenChange = openHostedConfirmStep();
    typeEmail("jane@example.com");

    fireEvent.click(getDeleteButton());

    await waitFor(() => expect(getEmailInput().disabled).toBe(true));
    expect(getDeleteButton().disabled).toBe(true);
    expect(getKeepButton().disabled).toBe(true);
    expect(
      within(screen.getByRole("dialog")).queryByRole("button", {
        name: "Close",
      }),
    ).toBeNull();
    fireEvent.click(getKeepButton());
    pressEscape();
    expect(screen.getByRole("dialog", { name: "Delete your account?" })).toBe(
      screen.getByRole("dialog"),
    );
    expect(onOpenChange).not.toHaveBeenCalled();

    gate.resolve();

    await waitFor(() => expect(mocks.replace).toHaveBeenCalledWith("/logout"));
    expect(onOpenChange).not.toHaveBeenCalled();
  });

  test("after a failed request the dialog can be closed again", async () => {
    const gate = gateDeletion(() =>
      HttpResponse.json(
        { detail: "That email doesn't match your account." },
        { status: 400 },
      ),
    );
    const onOpenChange = openHostedConfirmStep();
    typeEmail("jane@example.com");
    fireEvent.click(getDeleteButton());
    await waitFor(() => expect(getKeepButton().disabled).toBe(true));

    gate.resolve();

    await waitFor(() => expect(getKeepButton().disabled).toBe(false));
    expect(
      within(screen.getByRole("dialog")).getByRole("button", { name: "Close" }),
    ).toBeDefined();
    pressEscape();
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(onOpenChange.mock.calls).toEqual([[false]]);
  });

  test("a 200 says the account is closed until when and hands signing out to the /logout page", async () => {
    const requests = recordDeletion();
    openConfirmStep();
    typeEmail("jane@example.com");

    fireEvent.click(getDeleteButton());

    await waitFor(() => expect(mocks.replace).toHaveBeenCalled());
    expect(requests).toEqual([
      { confirm_email: "jane@example.com", source: "settings" },
    ]);
    expect(mocks.replace.mock.calls).toEqual([["/logout"]]);
    expect(mocks.toast.mock.calls).toEqual([[{ title: CLOSED_TOAST }]]);
    expect(mocks.logOut).not.toHaveBeenCalled();
    expect(getDeleteButton().disabled).toBe(true);
    expect(getKeepButton().disabled).toBe(true);
  });

  test.each([
    [400, "That email doesn't match your account."],
    [502, "We couldn't cancel your plan. Nothing was deleted. Try again."],
  ] as const)(
    "a %i shows the server's message and keeps the dialog open",
    async (status, detail) => {
      const requests = rejectDeletion(status, detail);
      openConfirmStep();
      typeEmail("jane@example.com");

      fireEvent.click(getDeleteButton());

      await waitFor(() =>
        expect(mocks.toast).toHaveBeenCalledWith({
          title: detail,
          variant: "destructive",
        }),
      );
      expect(mocks.toast).toHaveBeenCalledOnce();
      expect(requests).toHaveLength(1);
      expect(mocks.logOut).not.toHaveBeenCalled();
      expect(mocks.replace).not.toHaveBeenCalled();
      expect(screen.getByRole("dialog", { name: "Delete your account?" })).toBe(
        screen.getByRole("dialog"),
      );
      expect(getEmailInput().value).toBe("jane@example.com");
      await waitFor(() => expect(getDeleteButton().disabled).toBe(false));
      expect(getEmailInput().disabled).toBe(false);
    },
  );

  test("a network failure shows the fallback message and keeps the dialog open", async () => {
    server.use(http.post(DELETION_PATH, () => HttpResponse.error()));
    openConfirmStep();
    typeEmail("jane@example.com");

    fireEvent.click(getDeleteButton());

    await waitFor(() =>
      expect(mocks.toast).toHaveBeenCalledWith({
        title: "We couldn't delete your account. Try again.",
        variant: "destructive",
      }),
    );
    expect(mocks.logOut).not.toHaveBeenCalled();
    expect(screen.getByRole("dialog", { name: "Delete your account?" })).toBe(
      screen.getByRole("dialog"),
    );
  });
});
