import { getGetV2GetCopilotUsageQueryKey } from "@/app/api/__generated__/endpoints/chat/chat";
import {
  getGetV2GetCopilotUsageMockHandler200,
  getPostV2CreateSessionMockHandler200,
} from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { server } from "@/mocks/mock-server";
import {
  assistantTextChunks,
  copilotStreamErrorHandler,
  streamSseResponse,
} from "@/tests/integrations/copilot-sse";
import {
  setTrialUser,
  trialResponse,
} from "@/tests/integrations/trial-fixtures";
import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { resetCopilotChatRegistry } from "../copilotChatRegistry";
import { useCopilotStreamStore } from "../copilotStreamStore";
import { useCopilotUIStore } from "../store";
import {
  budgetHandlers,
  connection,
  fakeAtReset,
  MONDAY_1AM,
  renderChat,
  SATURDAY_1AM,
  SATURDAY_848PM,
  SUNDAY_1AM,
  weeklyExhausted,
} from "./budget-helpers";
import { TEST_BACKEND_BASE_URL, TEST_SESSION_ID } from "./sse-helpers";

vi.mock("@/services/environment", async (importActual) => {
  const actual = await importActual<typeof import("@/services/environment")>();
  return {
    ...actual,
    environment: {
      ...actual.environment,
      getAGPTServerBaseUrl: () => TEST_BACKEND_BASE_URL,
    },
  };
});

vi.mock("../helpers", async (importActual) => {
  const actual = await importActual<typeof import("../helpers")>();
  return {
    ...actual,
    getCopilotAuthHeaders: async () => ({ "x-test-auth": "yes" }),
  };
});

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    user: { id: "test-user" },
    isUserLoading: false,
    isLoggedIn: true,
  }),
}));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) => flag === actual.Flag.ENABLE_PLATFORM_PAYMENT,
    useFlagStatus: () => ({ enabled: false, ready: true }),
  };
});

const toast = vi.hoisted(() => vi.fn());

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast,
  useToast: () => ({ toast, dismiss: vi.fn() }),
}));

const SCHEDULE_PLACEHOLDER = "Write now, and it sends when your budget resets…";

beforeEach(() => {
  toast.mockClear();
  vi.useFakeTimers({ toFake: ["Date"] });
  vi.setSystemTime(SATURDAY_1AM);
  resetCopilotChatRegistry();
  useCopilotStreamStore.getState().resetAll();
  useCopilotUIStore.setState({ initialPrompt: null });
  sessionStorage.clear();
  setTrialUser("test-user");
});

afterEach(() => {
  vi.useRealTimers();
  useAuthStore.setState({ user: null, hasLoadedUser: false });
});

function composer() {
  return screen.getByLabelText(/chat message input/i) as HTMLTextAreaElement;
}

function scheduledChip() {
  return screen.queryByRole("status", { name: "Scheduled message" });
}

async function scheduleDraft(text: string) {
  const user = userEvent.setup();
  const input = await screen.findByPlaceholderText(SCHEDULE_PLACEHOLDER);
  await waitFor(() =>
    expect((input as HTMLTextAreaElement).disabled).toBe(false),
  );
  await user.type(input, text);
  await user.click(screen.getByRole("button", { name: "Send at reset" }));
  return user;
}

describe("Send at reset", () => {
  it("schedules the draft, shows when it sends, and Edit and Cancel take it back", async () => {
    const atReset = fakeAtReset();
    server.use(
      ...budgetHandlers({ usage: weeklyExhausted() }),
      ...atReset.handlers,
    );
    renderChat();

    const user = await scheduleDraft("Draft the launch post");

    const chip = await screen.findByRole("status", {
      name: "Scheduled message",
    });
    expect(within(chip).getByText("Sends Monday at 1:07 AM")).toBeDefined();
    expect(within(chip).getByText("Draft the launch post")).toBeDefined();
    expect(atReset.state.scheduled).toEqual([
      {
        sessionId: TEST_SESSION_ID,
        body: { kind: "send", message: "Draft the launch post" },
      },
    ]);
    expect(composer().value).toBe("");

    await user.click(within(chip).getByRole("button", { name: "Edit" }));
    await waitFor(() => expect(scheduledChip()).toBeNull());
    expect(atReset.state.cancelled).toEqual(["send"]);
    expect(composer().value).toBe("Draft the launch post");

    await user.click(screen.getByRole("button", { name: "Send at reset" }));
    const rescheduled = await screen.findByRole("status", {
      name: "Scheduled message",
    });
    expect(atReset.state.scheduled).toHaveLength(2);
    await user.click(
      within(rescheduled).getByRole("button", { name: "Cancel" }),
    );
    await waitFor(() => expect(scheduledChip()).toBeNull());
    expect(atReset.state.cancelled).toEqual(["send", "send"]);
    expect(atReset.state.jobs).toEqual([]);
    expect(composer().value).toBe("");
  });

  it("adds a second message to the one already waiting instead of replacing it", async () => {
    const atReset = fakeAtReset();
    server.use(
      ...budgetHandlers({ usage: weeklyExhausted() }),
      ...atReset.handlers,
    );
    renderChat();
    await scheduleDraft("Draft the launch post");
    await screen.findByRole("status", { name: "Scheduled message" });

    await scheduleDraft("Also the changelog");

    await waitFor(() => expect(atReset.state.scheduled).toHaveLength(2));
    // The client posts what was written; the server joins it on.
    expect(atReset.state.scheduled[1].body).toEqual({
      kind: "send",
      message: "Also the changelog",
    });
    expect(atReset.state.jobs[0].message).toBe(
      "Draft the launch post\n\nAlso the changelog",
    );
    const chip = await screen.findByRole("status", {
      name: "Scheduled message",
    });
    expect(within(chip).getByText(/Also the changelog/)).toBeDefined();
    expect(within(chip).getByText(/Draft the launch post/)).toBeDefined();
  });

  it("says when a message is too long to join the one waiting, and keeps it", async () => {
    const atReset = fakeAtReset();
    atReset.state.maxMessageChars = 40;
    server.use(
      ...budgetHandlers({ usage: weeklyExhausted() }),
      ...atReset.handlers,
    );
    renderChat();
    await scheduleDraft("Draft the launch post");
    await screen.findByRole("status", { name: "Scheduled message" });

    await scheduleDraft("Also the changelog for this week");

    await waitFor(() =>
      expect(toast).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Couldn’t schedule your message",
          description:
            "Too long to add to the message already waiting — your message is back in the composer.",
        }),
      ),
    );
    expect(composer().value).toBe("Also the changelog for this week");
    expect(atReset.state.jobs[0].message).toBe("Draft the launch post");
  });

  it("still shows the scheduled message after a reload", async () => {
    const atReset = fakeAtReset();
    server.use(
      ...budgetHandlers({ usage: weeklyExhausted() }),
      ...atReset.handlers,
    );
    const first = renderChat();
    await scheduleDraft("Draft the launch post");
    await screen.findByRole("status", { name: "Scheduled message" });
    first.unmount();

    renderChat();
    const chip = await screen.findByRole("status", {
      name: "Scheduled message",
    });
    expect(within(chip).getByText("Sends Monday at 1:07 AM")).toBeDefined();
    expect(within(chip).getByText("Draft the launch post")).toBeDefined();
  });

  it("sends right away when the budget turns out to be back", async () => {
    const atReset = fakeAtReset();
    atReset.state.refusal = "not_out_of_budget";
    const streamed: string[] = [];
    server.use(
      ...budgetHandlers({ usage: weeklyExhausted() }),
      ...atReset.handlers,
      http.post(
        `${TEST_BACKEND_BASE_URL}/api/chat/sessions/${TEST_SESSION_ID}/stream`,
        async ({ request }) => {
          streamed.push(await request.clone().text());
          return streamSseResponse(assistantTextChunks("On it."), {
            abortSignal: request.signal,
          });
        },
      ),
    );
    renderChat();

    await scheduleDraft("Draft the launch post");

    await waitFor(() => expect(streamed).toHaveLength(1));
    expect(streamed[0]).toContain("Draft the launch post");
    expect(scheduledChip()).toBeNull();
  });

  it("schedules the message a refused send put back, from the plan dialog", async () => {
    const atReset = fakeAtReset();
    let usageReads = 0;
    server.use(
      ...budgetHandlers({ usage: weeklyExhausted() }),
      ...atReset.handlers,
      copilotStreamErrorHandler({
        baseUrl: TEST_BACKEND_BASE_URL,
        sessionId: TEST_SESSION_ID,
        status: 429,
        body: {
          detail: {
            kind: "usage_limit",
            message: "You've reached your weekly usage limit.",
            authProvider: "platform",
            credentialId: null,
            resetsAt: null,
            retryable: false,
            reconnectFixesIt: false,
          },
        },
      }),
    );
    server.use(
      getGetV2GetCopilotUsageMockHandler200(() =>
        usageReads++ === 0
          ? weeklyExhausted({
              state: "ok",
              limiting_window: null,
              resets_at: null,
              daily: null,
              weekly: null,
            })
          : weeklyExhausted(),
      ),
    );
    renderChat();

    const user = userEvent.setup();
    const input = (await screen.findByPlaceholderText(
      "What else can I help with?",
    )) as HTMLTextAreaElement;
    await waitFor(() => expect(input.disabled).toBe(false));
    await user.type(input, "Draft the launch post");
    await user.click(screen.getByRole("button", { name: "Submit" }));

    const dialog = await screen.findByRole("dialog", undefined, {
      timeout: 5000,
    });
    await within(dialog).findByText(
      "You’re out of budget for this week",
      undefined,
      { timeout: 5000 },
    );
    await user.click(
      within(dialog).getByRole("button", { name: "Send at reset" }),
    );

    await screen.findByRole("status", { name: "Scheduled message" });
    expect(atReset.state.scheduled).toEqual([
      {
        sessionId: TEST_SESSION_ID,
        body: { kind: "send", message: "Draft the launch post" },
      },
    ]);
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(composer().value).toBe("");
  });

  it("creates the chat first from the new-chat screen, then schedules it", async () => {
    const atReset = fakeAtReset();
    server.use(
      ...budgetHandlers({ usage: weeklyExhausted() }),
      ...atReset.handlers,
      getPostV2CreateSessionMockHandler200({
        id: "new-chat-1",
        created_at: "2026-10-10T06:00:00Z",
        user_id: "test-user",
      }),
    );
    renderChat("");

    await scheduleDraft("Plan the offsite");

    const chip = await screen.findByRole(
      "status",
      { name: "Scheduled message" },
      { timeout: 5000 },
    );
    expect(within(chip).getByText("Sends Monday at 1:07 AM")).toBeDefined();
    expect(atReset.state.scheduled).toEqual([
      {
        sessionId: "new-chat-1",
        body: { kind: "send", message: "Plan the offsite" },
      },
    ]);
  });
});

describe("Out of budget", () => {
  it("gives a top-tier plan the reset time, the meters, and ChatGPT to continue on", async () => {
    server.use(
      ...budgetHandlers({
        usage: weeklyExhausted(),
        offers: [
          connection("platform"),
          connection("microsoft_365_copilot"),
          connection("codex"),
        ],
      }),
    );
    renderChat();

    const notice = await screen.findByRole("status", { name: "Usage notice" });
    await within(notice).findByText("You’re out of budget for this week");
    expect(
      within(notice).getByText(
        "It resets Monday at 1:00 AM, in 2 days. You can still browse, edit workflows and read results.",
      ),
    ).toBeDefined();
    for (const meter of ["Today", "This week"]) {
      expect(
        within(notice)
          .getByRole("progressbar", { name: meter })
          .getAttribute("aria-valuenow"),
      ).toBe("100");
    }
    expect(
      await within(notice).findByRole("button", {
        name: "Continue on ChatGPT",
      }),
    ).toBeDefined();
    expect(screen.getByRole("button", { name: "Send at reset" })).toBeDefined();
    expect(screen.queryByText(/budget left/)).toBeNull();
  });

  it("never offers Microsoft 365 Copilot to continue on", async () => {
    let offersRead = 0;
    server.use(...budgetHandlers({ usage: weeklyExhausted() }));
    server.use(
      http.get("*/api/chat/connections", () => {
        offersRead += 1;
        return HttpResponse.json({
          offers: [connection("platform"), connection("microsoft_365_copilot")],
        });
      }),
    );
    renderChat();

    await screen.findByText("You’re out of budget for this week");
    await waitFor(() => expect(offersRead).toBeGreaterThan(0));
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(screen.queryByRole("button", { name: /Continue on/ })).toBeNull();
  });

  it.each([
    [
      "a spent trial allowance",
      {
        tier: "TRIAL" as const,
        daily: null,
        limiting_window: "trial" as const,
        resets_at: null,
      },
      trialResponse({ usage_policy: "lifetime", allowance_used_percent: 100 }),
    ],
    [
      "an account without a plan",
      { tier: "NO_TIER" as const, resets_at: null },
      undefined,
    ],
  ])(
    "offers only an upgrade for %s, which never refills",
    async (_, over, trial) => {
      server.use(...budgetHandlers({ usage: weeklyExhausted(over), trial }));
      renderChat();

      const notice = await screen.findByRole("status", {
        name: "Usage notice",
      });
      expect(
        await within(notice).findByRole("button", { name: /^Start Pro/ }),
      ).toBeDefined();
      expect(within(notice).queryByText(/resets|refreshes/)).toBeNull();
      expect(
        screen.queryByRole("button", { name: "Send at reset" }),
      ).toBeNull();
      expect(screen.queryByPlaceholderText(SCHEDULE_PLACEHOLDER)).toBeNull();
      expect(
        screen.getByRole("button", { name: "Submit" }).hasAttribute("disabled"),
      ).toBe(true);
    },
  );

  it("lets a trial wait for its own weekly cap, which does refill", async () => {
    server.use(
      ...budgetHandlers({
        usage: weeklyExhausted({
          tier: "TRIAL",
          daily: { percent_used: 40, resets_at: SUNDAY_1AM },
        }),
        trial: trialResponse({
          usage_policy: "lifetime",
          allowance_used_percent: 60,
        }),
      }),
    );
    renderChat();

    await screen.findByText("You’re out of budget for this week");
    expect(
      await screen.findByRole("button", { name: "Send at reset" }),
    ).toBeDefined();
  });

  it("doesn't lock a ChatGPT chat, which its own plan pays for", async () => {
    server.use(
      ...budgetHandlers({ usage: weeklyExhausted(), provider: "codex" }),
    );
    const { queryClient } = renderChat();

    await waitFor(() =>
      expect(
        queryClient.getQueryState(getGetV2GetCopilotUsageQueryKey())?.status,
      ).toBe("success"),
    );
    const user = userEvent.setup();
    const input = await screen.findByPlaceholderText(
      "What else can I help with?",
    );
    await user.type(input, "Keep going");
    expect(
      screen.getByRole("button", { name: "Submit" }).hasAttribute("disabled"),
    ).toBe(false);
    expect(screen.queryByRole("status", { name: "Usage notice" })).toBeNull();
  });
});

describe("Running low", () => {
  it.each([
    ["an open chat", `?sessionId=${TEST_SESSION_ID}`],
    ["the new-chat screen", ""],
  ])("warns under the composer in %s", async (_, searchParams) => {
    vi.setSystemTime(SATURDAY_848PM);
    server.use(
      ...budgetHandlers({
        usage: {
          tier: "PRO",
          daily: { percent_used: 86, resets_at: SUNDAY_1AM },
          weekly: { percent_used: 61, resets_at: MONDAY_1AM },
          reset_cost: 0,
          state: "low",
          limiting_window: "daily",
          resets_at: SUNDAY_1AM,
        },
      }),
    );
    renderChat(searchParams);

    expect(
      await screen.findByText(
        "About 14% of today’s budget left. A long task may pause partway.",
      ),
    ).toBeDefined();
    // One small gauge for the window that is running low, not a meter per
    // window: under the composer, every line costs typing room on a phone.
    expect(
      screen
        .getByRole("progressbar", { name: "Today" })
        .getAttribute("aria-valuenow"),
    ).toBe("86");
    expect(screen.queryByRole("progressbar", { name: "This week" })).toBeNull();
  });
});
