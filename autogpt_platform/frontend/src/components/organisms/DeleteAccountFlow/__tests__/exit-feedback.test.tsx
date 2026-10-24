import { getPostAccountSendExitFeedbackMockHandler429 } from "@/app/api/__generated__/endpoints/account/account.msw";
import { server } from "@/mocks/mock-server";
import {
  GMAIL,
  fakeCredentialStore,
  mockAccountDeletionAPI,
  recordFeedback,
} from "@/tests/integrations/account-deletion-fixtures";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import { beforeEach, describe, expect, test, vi } from "vitest";
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

function renderFlow(source: DeletionSource = "settings") {
  const onOpenChange = vi.fn();
  render(
    <DeleteAccountFlow source={source} open onOpenChange={onOpenChange} />,
  );
  return onOpenChange;
}

function typeFeedback(value: string) {
  fireEvent.change(screen.getByRole("textbox", { name: "Tell us more" }), {
    target: { value },
  });
}

function pickReason(label: string) {
  fireEvent.click(screen.getByRole("radio", { name: label }));
}

function expectConfirmStep() {
  expect(screen.getByRole("dialog", { name: "Delete your account?" })).toBe(
    screen.getByRole("dialog"),
  );
  expect(
    screen.getByRole("textbox", { name: "Type jane@example.com to confirm" }),
  ).toBeDefined();
}

async function flushRequests() {
  await act(() => new Promise((resolve) => setTimeout(resolve, 20)));
}

beforeEach(() => {
  mockAccountDeletionAPI();
});

describe("DeleteAccountFlow exit feedback", () => {
  test("opens on the reason step with nothing picked", () => {
    renderFlow();

    expect(
      screen.getByRole("dialog", { name: "Why are you leaving?" }).textContent,
    ).toContain(
      "Optional. You can skip straight to deleting, and your answer isn't linked to your account.",
    );
    expect(screen.getAllByRole("radio")).toHaveLength(6);
    expect(screen.queryByRole("radio", { checked: true })).toBeNull();
  });

  test("the text box is 16px below lg and 14px above it", () => {
    renderFlow();

    const classes = screen
      .getByRole("textbox", { name: "Tell us more" })
      .className.split(/\s+/);
    expect(classes).toContain("!text-base");
    expect(classes).toContain("lg:!text-sm");
  });

  test("Skip sends one skipped answer, ignoring any reason or text, and opens the confirmation", async () => {
    const feedback = recordFeedback();
    const onOpenChange = renderFlow();
    pickReason("It costs too much");
    typeFeedback("Too pricey for me");

    fireEvent.click(screen.getByRole("button", { name: "Skip" }));

    expectConfirmStep();
    await waitFor(() =>
      expect(feedback).toEqual([{ kind: "answer", reason: "skipped" }]),
    );
    await flushRequests();
    expect(feedback).toHaveLength(1);
    expect(onOpenChange).not.toHaveBeenCalled();
  });

  test("Continue sends the picked reason and the trimmed text once, then opens the confirmation", async () => {
    const feedback = recordFeedback();
    renderFlow();
    pickReason("It couldn't do what I needed");
    typeFeedback("   Export my runs to Sheets  ");

    fireEvent.click(screen.getByRole("button", { name: "Continue" }));

    expectConfirmStep();
    await waitFor(() =>
      expect(feedback).toEqual([
        {
          kind: "answer",
          reason: "missing_capability",
          text: "Export my runs to Sheets",
        },
      ]),
    );
    await flushRequests();
    expect(feedback).toHaveLength(1);
  });

  test("Continue with a reason and no text sends just the reason", async () => {
    const feedback = recordFeedback();
    renderFlow();
    pickReason("Something else");

    fireEvent.click(screen.getByRole("button", { name: "Continue" }));

    expectConfirmStep();
    await waitFor(() =>
      expect(feedback).toEqual([{ kind: "answer", reason: "other" }]),
    );
  });

  test("Continue with only text sends the text without a reason", async () => {
    const feedback = recordFeedback();
    renderFlow();
    typeFeedback(" I found another tool ");

    fireEvent.click(screen.getByRole("button", { name: "Continue" }));

    expectConfirmStep();
    await waitFor(() =>
      expect(feedback).toEqual([
        { kind: "answer", text: "I found another tool" },
      ]),
    );
  });

  test.each([
    ["nothing", ""],
    ["only spaces", "    "],
  ])(
    "Continue with %s typed and no reason counts as skipped",
    async (_, text) => {
      const feedback = recordFeedback();
      renderFlow();
      typeFeedback(text);

      fireEvent.click(screen.getByRole("button", { name: "Continue" }));

      expectConfirmStep();
      await waitFor(() =>
        expect(feedback).toEqual([{ kind: "answer", reason: "skipped" }]),
      );
    },
  );

  test("Enter on the picked reason continues like the Continue button", async () => {
    const feedback = recordFeedback();
    renderFlow();
    const cost = screen.getByRole("radio", { name: "It costs too much" });
    fireEvent.click(cost);

    fireEvent.keyDown(cost, { key: "Enter" });

    expectConfirmStep();
    await waitFor(() =>
      expect(feedback).toEqual([{ kind: "answer", reason: "cost" }]),
    );
  });

  test("keyboard focus is kept clear of the sticky footer", () => {
    renderFlow();

    const classes = screen.getByRole("dialog").className.split(" ");
    expect(classes).toContain("[&_.overflow-y-auto]:scroll-pb-24");
    // Where the footer stops being sticky the padding would push a focused
    // row above the top of the tiny scroll area instead (400% zoom).
    expect(classes).toContain(
      "[@media(max-height:300px)]:[&_.overflow-y-auto]:scroll-pb-0",
    );
  });

  test("the text box takes at most 1000 characters", () => {
    renderFlow();

    expect(
      screen
        .getByRole("textbox", { name: "Tell us more" })
        .getAttribute("maxLength"),
    ).toBe("1000");
  });

  test("a failed feedback post never blocks the way to the confirmation", async () => {
    server.use(getPostAccountSendExitFeedbackMockHandler429());
    renderFlow();

    fireEvent.click(screen.getByRole("button", { name: "Skip" }));

    expectConfirmStep();
    await flushRequests();
    expectConfirmStep();
  });

  test("an alternative clicked first is sent separately from the answer", async () => {
    fakeCredentialStore([GMAIL]);
    const feedback = recordFeedback();
    renderFlow("paywall");
    pickReason("I'm worried about my data or privacy");
    fireEvent.click(
      await screen.findByRole("button", { name: "Disconnect all apps" }),
    );
    await within(screen.getByRole("note")).findByText("Gmail is disconnected.");

    fireEvent.click(screen.getByRole("button", { name: "Continue" }));

    expectConfirmStep();
    await waitFor(() =>
      expect(feedback).toEqual([
        {
          kind: "alternative_clicked",
          reason: "privacy",
          alternative: "disconnect_all",
        },
        { kind: "answer", reason: "privacy" },
      ]),
    );
  });
});
