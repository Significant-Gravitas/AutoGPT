import { delay, http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  getGetV2ListChatConnectionsMockHandler200,
  getPutV2ChangeTheConnectionAnExistingChatRunsOnMockHandler200,
} from "@/app/api/__generated__/endpoints/chat/chat.msw";
import type { AIConnectionOffer } from "@/app/api/__generated__/models/aIConnectionOffer";
import { server } from "@/mocks/mock-server";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";

import { useCopilotUIStore } from "../../../store";
import { CopilotChatActionsProvider } from "../../CopilotChatActionsProvider/CopilotChatActionsProvider";
import { CurrentPauseContext } from "../CurrentPauseContext";
import { TurnPausedCard } from "../TurnPausedCard";

const { toast } = vi.hoisted(() => ({ toast: vi.fn() }));

vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/components/molecules/Toast/use-toast")
  >()),
  toast,
}));

vi.mock("@/services/pro-activation/useProActivation", () => ({
  useProActivation: () => ({ start: vi.fn(), isBusy: false, isReady: false }),
}));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/services/feature-flags/use-get-flag")
  >()),
  useGetFlag: () => true,
}));

const SESSION_ID = "chat-1";
// Sunday evening in local time, so the resets below read the same anywhere.
const NOW = new Date(2026, 9, 11, 20, 48);
const TONIGHT_1AM = new Date(2026, 9, 12, 1, 0);
const WEDNESDAY_1AM = new Date(2026, 9, 14, 1, 0);
const RESUMES_AT = new Date(2026, 9, 12, 1, 5);

function pause(over: Record<string, unknown> = {}) {
  return {
    reason: "daily",
    resets_at: TONIGHT_1AM.toISOString(),
    steps: null,
    checkpoint_saved: true,
    started: true,
    ...over,
  };
}

function offer(over: Partial<AIConnectionOffer> = {}): AIConnectionOffer {
  return {
    offer_id: "platform:deployment",
    provider_family: "autogpt",
    auth_provider: "platform",
    display_name: "AutoGPT Platform",
    auth_method: "deployment",
    credential_id: null,
    backed_by_label: "Your AutoGPT plan",
    description: "",
    state: "ready",
    selectable: true,
    is_default: true,
    tiers: [],
    limitations: [],
    lock_reason: null,
    unlock_href: null,
    ...over,
  };
}

const PLATFORM = offer();
const CHATGPT = offer({
  offer_id: "codex:cred-1",
  provider_family: "openai",
  auth_provider: "codex",
  display_name: "ChatGPT",
  auth_method: "chatgpt_oauth",
  credential_id: "cred-1",
  is_default: false,
});
const M365 = offer({
  offer_id: "microsoft_365_copilot:cred-9",
  provider_family: "microsoft",
  auth_provider: "microsoft_365_copilot",
  display_name: "Microsoft 365 Copilot",
  auth_method: "device_code",
  credential_id: "cred-9",
  is_default: false,
});

function backend({
  tier = "PRO",
  offers = [PLATFORM],
  jobs = [] as unknown[],
} = {}) {
  const calls = {
    continued: [] as unknown[],
    scheduled: [] as unknown[],
    switched: [] as unknown[],
    cancelled: 0,
    listed: 0,
    plan: 0,
  };
  let pending = jobs;
  server.use(
    http.get("*/api/chat/usage", () =>
      HttpResponse.json({
        tier,
        daily: { percent_used: 100, resets_at: TONIGHT_1AM.toISOString() },
        weekly: { percent_used: 40, resets_at: WEDNESDAY_1AM.toISOString() },
      }),
    ),
    http.get("*/api/credits/subscription", () => {
      calls.plan += 1;
      return HttpResponse.json({
        tier,
        tier_costs: { PRO: 5000, MAX: 32000 },
        monthly_cost: 5000,
        proration_credit_cents: 0,
      });
    }),
    getGetV2ListChatConnectionsMockHandler200({ offers }),
    http.get("*/api/chat/sessions/:id/at-reset", () => {
      calls.listed += 1;
      return HttpResponse.json({ jobs: pending });
    }),
    http.post("*/api/chat/sessions/:id/at-reset", async ({ request }) => {
      calls.scheduled.push(await request.json());
      const job = {
        kind: "resume",
        run_at: RESUMES_AT.toISOString(),
        message: null,
      };
      pending = [job];
      return HttpResponse.json(job);
    }),
    http.delete("*/api/chat/sessions/:id/at-reset/:kind", () => {
      calls.cancelled += 1;
      pending = [];
      return new HttpResponse(null, { status: 204 });
    }),
    http.post("*/api/chat/sessions/:id/continue", async ({ request }) => {
      calls.continued.push(await request.json());
      return HttpResponse.json({ turn_id: "turn-2" }, { status: 202 });
    }),
    getPutV2ChangeTheConnectionAnExistingChatRunsOnMockHandler200(
      async ({ request }) => {
        calls.switched.push(await request.json());
        return {};
      },
    ),
  );
  return calls;
}

interface CardOptions {
  current?: boolean;
  name?: string;
  sessionId?: string;
  /** Still streaming in: no stored row, so no key that outlives a reload. */
  live?: boolean;
  rowKey?: string;
}

function cardTree(
  data: Record<string, unknown>,
  {
    current = true,
    name = "Otto",
    sessionId = SESSION_ID,
    live = false,
    rowKey = "seq-7",
  }: CardOptions,
  onBackendTurn: () => void,
) {
  return (
    <CopilotChatActionsProvider
      onSend={vi.fn()}
      onBackendTurn={onBackendTurn}
      sessionId={sessionId}
      assistantName={name}
    >
      <CurrentPauseContext.Provider value={current ? data : null}>
        <TurnPausedCard
          data={data}
          pauseKey={live ? "live-turn-3" : rowKey}
          isStoredRow={!live}
        />
      </CurrentPauseContext.Provider>
    </CopilotChatActionsProvider>
  );
}

function renderCard(data: Record<string, unknown>, options: CardOptions = {}) {
  const onBackendTurn = vi.fn();
  const view = render(cardTree(data, options, onBackendTurn));
  return {
    onBackendTurn,
    ...view,
    rerenderCard: (next: Record<string, unknown>, nextOptions: CardOptions) =>
      view.rerender(cardTree(next, nextOptions, onBackendTurn)),
  };
}

/** Exact text whose spaces may be any whitespace: times can carry a narrow
 *  no-break space before "AM", depending on the ICU build. */
function phrase(text: string) {
  const escaped = text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return new RegExp(`^${escaped.replace(/ /g, "\\s")}$`);
}

function buttonsIn(card: HTMLElement) {
  return within(card)
    .queryAllByRole("button")
    .map((button) => (button.textContent ?? "").replace(/\s+/g, " ").trim());
}

beforeEach(() => {
  vi.useFakeTimers({ toFake: ["Date"] });
  vi.setSystemTime(NOW);
  window.localStorage.clear();
  useCopilotUIStore.setState({ copilotLlmModel: "standard" });
});

afterEach(() => {
  vi.useRealTimers();
});

describe("TurnPausedCard — what it says and offers", () => {
  it.each([
    [
      "a daily budget pause",
      pause(),
      "Paused · out of today's budget",
      "Otto saved its progress above. Your budget resets at 1:00 AM, in 4 h 12 min.",
      ["Resume at 1:00 AM", "Upgrade"],
    ],
    [
      "a weekly budget pause",
      pause({ reason: "weekly", resets_at: WEDNESDAY_1AM.toISOString() }),
      "Paused · out of this week's budget",
      "Otto saved its progress above. Your budget resets Wednesday at 1:00 AM, in 2 days.",
      ["Resume Wednesday at 1:00 AM", "Upgrade"],
    ],
    [
      "a pause stopped before its checkpoint",
      pause({ checkpoint_saved: false }),
      "Paused · out of today's budget",
      "Otto was stopped before it could summarize its progress. Your budget resets at 1:00 AM, in 4 h 12 min.",
      ["Resume at 1:00 AM", "Upgrade"],
    ],
    [
      "a turn refused before it ran",
      pause({ started: false }),
      "Not started · out of today's budget",
      "Otto didn't start this because you're out of today's budget. It resets at 1:00 AM, in 4 h 12 min.",
      ["Resume at 1:00 AM", "Upgrade"],
    ],
    // No plan (a trial that is over): the backend sends "weekly" with no
    // reset, so there is no window to name and nothing to wait for.
    [
      "a budget with no plan behind it, which never refills",
      pause({ reason: "weekly", resets_at: null }),
      "Paused · out of budget",
      "Otto saved its progress above.",
      ["Upgrade"],
    ],
    [
      "a turn refused on a budget that never refills",
      pause({ reason: "weekly", resets_at: null, started: false }),
      "Not started · out of budget",
      "Otto didn't start this because you're out of budget.",
      ["Upgrade"],
    ],
    [
      "a spent trial",
      pause({ reason: "trial", resets_at: null }),
      "Paused · trial allowance used",
      "Otto saved its progress above.",
      ["Upgrade"],
    ],
    [
      "a task at its size limit",
      pause({ reason: "task", resets_at: null }),
      "Paused · this task reached its size limit",
      "Otto saved its progress above.",
      ["Keep going", "Stop here"],
    ],
    [
      "a step check-in",
      pause({ reason: "steps", resets_at: null, steps: 450 }),
      "Paused · checking in after a long run",
      "Otto has taken 450 steps on this and summarized its progress above.",
      ["Keep going", "Stop here"],
    ],
  ])("%s", async (_, data, title, body, buttons) => {
    backend();
    renderCard(data);
    const card = await screen.findByRole("region", { name: title });
    expect(within(card).getByText(phrase(body))).toBeDefined();
    await waitFor(() => expect(buttonsIn(card)).toEqual(buttons));
  });

  it("counts a reset just under a day away in hours, as it names the time without a day", async () => {
    // Tuesday, a day less 30 seconds before the reset.
    vi.setSystemTime(new Date(2026, 9, 13, 1, 0, 30));
    backend();
    renderCard(
      pause({ reason: "weekly", resets_at: WEDNESDAY_1AM.toISOString() }),
    );
    const card = await screen.findByRole("region", {
      name: "Paused · out of this week's budget",
    });

    expect(
      within(card).getByText(
        phrase(
          "Otto saved its progress above. Your budget resets at 1:00 AM, in 24 h.",
        ),
      ),
    ).toBeDefined();
  });

  it("offers ChatGPT beside Upgrade on a budget that never refills, and never a resume", async () => {
    const calls = backend({ offers: [PLATFORM, CHATGPT] });
    renderCard(pause({ reason: "weekly", resets_at: null }));
    const card = await screen.findByRole("region", {
      name: "Paused · out of budget",
    });

    await waitFor(() =>
      expect(buttonsIn(card)).toEqual(["Continue on ChatGPT", "Upgrade"]),
    );
    expect(
      within(card).getByText("Otto saved its progress above."),
    ).toBeDefined();
    expect(calls.listed).toBe(0);
  });

  it("speaks as the chat's own expert", async () => {
    backend();
    renderCard(pause({ reason: "task", resets_at: null }), { name: "Nova" });
    expect(screen.getByText("Nova saved its progress above.")).toBeDefined();
  });

  it("leaves Upgrade off a top-tier plan, which keeps Resume and ChatGPT", async () => {
    const calls = backend({ tier: "MAX", offers: [PLATFORM, CHATGPT] });
    renderCard(pause());
    const card = await screen.findByRole("region", {
      name: "Paused · out of today's budget",
    });
    await screen.findByRole("button", { name: "Continue on ChatGPT" });
    await waitFor(() => expect(calls.plan).toBeGreaterThan(0));
    await expect(
      screen.findByRole("button", { name: "Upgrade" }, { timeout: 1000 }),
    ).rejects.toThrow();
    expect(buttonsIn(card)).toEqual([
      "Resume at 1:00 AM",
      "Continue on ChatGPT",
    ]);
  });
});

describe("TurnPausedCard — Keep going", () => {
  it("picks the task up on the composer's model, then follows the turn", async () => {
    const calls = backend();
    useCopilotUIStore.setState({ copilotLlmModel: "advanced" });
    const { onBackendTurn } = renderCard(
      pause({ reason: "steps", resets_at: null, steps: 450 }),
    );

    fireEvent.click(screen.getByRole("button", { name: "Keep going" }));

    await waitFor(() => expect(onBackendTurn).toHaveBeenCalledOnce());
    expect(calls.continued).toEqual([{ model: "advanced" }]);
    expect(
      screen.getByRole("note", {
        name: "Paused · checking in after a long run. Picked back up",
      }),
    ).toBeDefined();
    expect(screen.queryByRole("button")).toBeNull();
  });

  it("follows the turn that is already running instead of failing", async () => {
    backend();
    server.use(
      http.post("*/api/chat/sessions/:id/continue", () =>
        HttpResponse.json({ detail: "turn_in_progress" }, { status: 409 }),
      ),
    );
    const { onBackendTurn } = renderCard(
      pause({ reason: "task", resets_at: null }),
    );

    fireEvent.click(screen.getByRole("button", { name: "Keep going" }));

    await waitFor(() => expect(onBackendTurn).toHaveBeenCalledOnce());
    expect(await screen.findByText("Picked back up")).toBeDefined();
  });

  it("says why when the chat now runs on Microsoft 365 Copilot", async () => {
    backend();
    server.use(
      http.post("*/api/chat/sessions/:id/continue", () =>
        HttpResponse.json(
          { detail: "unsupported_connection" },
          { status: 409 },
        ),
      ),
    );
    const { onBackendTurn } = renderCard(
      pause({ reason: "task", resets_at: null }),
    );

    fireEvent.click(screen.getByRole("button", { name: "Keep going" }));

    await waitFor(() =>
      expect(toast).toHaveBeenCalledWith({
        title: "Couldn't pick this back up",
        description:
          "This chat now runs on Microsoft 365 Copilot, which can't pick up a paused task. Switch its connection back to keep going.",
        variant: "destructive",
      }),
    );
    expect(onBackendTurn).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: "Keep going" })).toBeDefined();
  });

  it("passes on the server's reason when too many tasks are running", async () => {
    backend();
    const tooMany =
      "You have 3 tasks already running. Please wait for one of them to finish before starting a new one.";
    server.use(
      http.post("*/api/chat/sessions/:id/continue", () =>
        HttpResponse.json({ detail: tooMany }, { status: 429 }),
      ),
    );
    const { onBackendTurn } = renderCard(
      pause({ reason: "task", resets_at: null }),
    );

    fireEvent.click(screen.getByRole("button", { name: "Keep going" }));

    await waitFor(() =>
      expect(toast).toHaveBeenCalledWith({
        title: "Couldn't pick this back up",
        description: tooMany,
        variant: "destructive",
      }),
    );
    expect(onBackendTurn).not.toHaveBeenCalled();
  });

  it("points to the plans when picking up needs one", async () => {
    backend();
    server.use(
      http.post("*/api/chat/sessions/:id/continue", () =>
        HttpResponse.json(
          { detail: "Choose a plan to keep using AutoPilot" },
          { status: 402 },
        ),
      ),
    );
    renderCard(pause({ reason: "task", resets_at: null }));

    fireEvent.click(screen.getByRole("button", { name: "Keep going" }));

    await waitFor(() =>
      expect(toast).toHaveBeenCalledWith({
        title: "Couldn't pick this back up",
        description: "Choose a plan to keep going.",
        variant: "destructive",
      }),
    );
  });

  it("turns into the budget pause when the account ran out meanwhile", async () => {
    backend();
    server.use(
      http.post("*/api/chat/sessions/:id/continue", () =>
        HttpResponse.json(
          {
            detail: {
              kind: "usage_limit",
              message: "You've reached your daily usage limit.",
              authProvider: "platform",
              credentialId: null,
              resetsAt: TONIGHT_1AM.getTime() / 1000,
              retryable: false,
              reconnectFixesIt: false,
            },
          },
          { status: 429 },
        ),
      ),
    );
    const { onBackendTurn } = renderCard(
      pause({ reason: "task", resets_at: null }),
    );

    fireEvent.click(screen.getByRole("button", { name: "Keep going" }));

    const card = await screen.findByRole("region", {
      name: "Paused · out of today's budget",
    });
    expect(
      within(card).getByText(
        phrase(
          "Otto saved its progress above. Your budget resets at 1:00 AM, in 4 h 12 min.",
        ),
      ),
    ).toBeDefined();
    await waitFor(() =>
      expect(buttonsIn(card)).toEqual(["Resume at 1:00 AM", "Upgrade"]),
    );
    expect(onBackendTurn).not.toHaveBeenCalled();
  });

  it("is offered once the budget has refilled", async () => {
    backend();
    renderCard(
      pause({ resets_at: new Date(2026, 9, 11, 20, 0).toISOString() }),
    );

    expect(
      screen.getByText("Otto saved its progress above. Your budget has reset."),
    ).toBeDefined();
    expect(screen.getByRole("button", { name: "Keep going" })).toBeDefined();
    expect(screen.queryByRole("button", { name: /^Resume/ })).toBeNull();
  });

  it("keeps counting down while the chat sits open, and is offered once the reset passes", async () => {
    vi.useFakeTimers({ toFake: ["Date", "setInterval", "clearInterval"] });
    vi.setSystemTime(NOW);
    backend();
    renderCard(pause());
    const card = await screen.findByRole("region", {
      name: "Paused · out of today's budget",
    });
    await waitFor(() =>
      expect(buttonsIn(card)).toEqual(["Resume at 1:00 AM", "Upgrade"]),
    );

    // Synchronous from here: the usage poll fires but none of its responses
    // land, so only the card's own clock can move the text.
    act(() => {
      vi.advanceTimersByTime(10 * 60_000);
    });

    expect(
      within(card).getByText(
        phrase(
          "Otto saved its progress above. Your budget resets at 1:00 AM, in 4 h 2 min.",
        ),
      ),
    ).toBeDefined();

    act(() => {
      vi.setSystemTime(TONIGHT_1AM);
      vi.advanceTimersByTime(1000);
    });

    expect(
      within(card).getByText(
        "Otto saved its progress above. Your budget has reset.",
      ),
    ).toBeDefined();
    expect(buttonsIn(card)).toEqual(["Keep going", "Upgrade"]);
  });
});

describe("TurnPausedCard — Resume at the reset", () => {
  it("sets the resume, says when it runs, and cancels it", async () => {
    const calls = backend();
    renderCard(pause());

    fireEvent.click(
      await screen.findByRole("button", { name: phrase("Resume at 1:00 AM") }),
    );

    expect(await screen.findByText(phrase("Resumes at 1:05 AM"))).toBeDefined();
    expect(calls.scheduled).toEqual([{ kind: "resume", message: null }]);
    expect(screen.queryByRole("button", { name: /^Resume/ })).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));

    expect(
      await screen.findByRole("button", { name: phrase("Resume at 1:00 AM") }),
    ).toBeDefined();
    expect(calls.cancelled).toBe(1);
    expect(screen.queryByText(/Resumes at/)).toBeNull();
  });

  it("shows a resume that was already set when the chat opens", async () => {
    backend({
      jobs: [
        { kind: "send", run_at: RESUMES_AT.toISOString(), message: "Hi" },
        { kind: "resume", run_at: RESUMES_AT.toISOString(), message: null },
      ],
    });
    renderCard(pause());

    expect(await screen.findByText(phrase("Resumes at 1:05 AM"))).toBeDefined();
    expect(screen.getByRole("button", { name: "Cancel" })).toBeDefined();
    expect(screen.queryByRole("button", { name: /^Resume/ })).toBeNull();
  });

  it("offers Keep going instead when the budget turns out to be back", async () => {
    backend();
    server.use(
      http.post("*/api/chat/sessions/:id/at-reset", () =>
        HttpResponse.json({ detail: "not_out_of_budget" }, { status: 409 }),
      ),
    );
    renderCard(pause());

    fireEvent.click(
      await screen.findByRole("button", { name: phrase("Resume at 1:00 AM") }),
    );

    expect(
      await screen.findByRole("button", { name: "Keep going" }),
    ).toBeDefined();
    expect(
      screen.getByText("Otto saved its progress above. Your budget has reset."),
    ).toBeDefined();
  });

  it.each([
    {
      status: 409,
      detail: "no_reset",
      title: "This allowance doesn't reset",
      description: "Upgrade to keep going.",
    },
    {
      status: 503,
      detail: "Rate limit service degraded, retry shortly",
      title: "Couldn't check your budget",
      description: "Try again in a moment.",
    },
  ])(
    "says why it couldn't set the resume ($status $detail)",
    async ({ status, detail, title, description }) => {
      backend();
      server.use(
        http.post("*/api/chat/sessions/:id/at-reset", () =>
          HttpResponse.json({ detail }, { status }),
        ),
      );
      renderCard(pause());

      fireEvent.click(
        await screen.findByRole("button", {
          name: phrase("Resume at 1:00 AM"),
        }),
      );

      await waitFor(() =>
        expect(toast).toHaveBeenCalledWith({
          title,
          description,
          variant: "destructive",
        }),
      );
      expect(
        screen.getByRole("button", { name: phrase("Resume at 1:00 AM") }),
      ).toBeDefined();
    },
  );
});

describe("TurnPausedCard — Continue on ChatGPT", () => {
  it("moves the chat to the ChatGPT account, then picks the task up there", async () => {
    const calls = backend({ offers: [PLATFORM, M365, CHATGPT] });
    const { onBackendTurn } = renderCard(pause());

    fireEvent.click(
      await screen.findByRole("button", { name: "Continue on ChatGPT" }),
    );

    await waitFor(() => expect(onBackendTurn).toHaveBeenCalledOnce());
    expect(calls.switched).toEqual([
      { llm_auth_provider: "codex", llm_credential_id: "cred-1" },
    ]);
    expect(calls.continued).toEqual([{ model: "standard" }]);
  });

  it("spins on the button that was clicked while the task is picked up", async () => {
    backend({ offers: [PLATFORM, CHATGPT] });
    let pickUps = 0;
    server.use(
      http.post("*/api/chat/sessions/:id/continue", async () => {
        pickUps += 1;
        await delay("infinite");
        return HttpResponse.json({});
      }),
    );
    // The budget is back, so Keep going sits beside Continue on ChatGPT.
    renderCard(
      pause({ resets_at: new Date(2026, 9, 11, 20, 0).toISOString() }),
    );

    fireEvent.click(
      await screen.findByRole("button", { name: "Continue on ChatGPT" }),
    );

    // Past the switch, the pick-up itself is running.
    await waitFor(() => expect(pickUps).toBe(1));
    await waitFor(() =>
      expect(
        screen
          .getByRole("button", { name: "Continue on ChatGPT" })
          .querySelector(".animate-spin"),
      ).not.toBeNull(),
    );
    expect(
      screen
        .getByRole("button", { name: "Keep going" })
        .querySelector(".animate-spin"),
    ).toBeNull();
  });

  it("never offers Microsoft 365 Copilot, which runs no tools, nor a locked ChatGPT", async () => {
    backend({
      offers: [
        PLATFORM,
        M365,
        { ...CHATGPT, offer_id: "codex:locked", credential_id: null },
        { ...CHATGPT, offer_id: "codex:cred-2", selectable: false },
      ],
    });
    renderCard(pause());

    await screen.findByRole("button", { name: "Upgrade" });
    await expect(
      screen.findByRole("button", { name: /continue on/i }, { timeout: 1000 }),
    ).rejects.toThrow();
  });

  it("is not offered for a pause the account still has budget for", async () => {
    backend({ offers: [PLATFORM, CHATGPT] });
    renderCard(pause({ reason: "task", resets_at: null }));

    await expect(
      screen.findByRole("button", { name: /continue on/i }, { timeout: 1000 }),
    ).rejects.toThrow();
  });
});

describe("TurnPausedCard — once it's history", () => {
  it("stops here, and stays stopped when the chat is opened again", () => {
    backend();
    const data = pause({ reason: "steps", resets_at: null, steps: 450 });
    const first = renderCard(data);

    fireEvent.click(screen.getByRole("button", { name: "Stop here" }));

    expect(screen.getByText("Stopped here")).toBeDefined();
    expect(screen.queryByRole("button")).toBeNull();

    first.unmount();
    renderCard(data);
    expect(screen.getByText("Stopped here")).toBeDefined();
    expect(screen.queryByRole("button")).toBeNull();
  });

  it("keeps a stop made while the pause was live once its stored row takes over, and only for that pause", () => {
    backend();
    const sessionId = "chat-stopped-live";
    const data = pause({ reason: "steps", resets_at: null, steps: 450 });
    const live = renderCard(data, { sessionId, live: true });

    fireEvent.click(screen.getByRole("button", { name: "Stop here" }));
    expect(screen.getByText("Stopped here")).toBeDefined();

    // The stream ends and the chat rebuilds the pause from its stored row.
    live.unmount();
    const stored = renderCard({ ...data }, { sessionId, rowKey: "seq-9" });
    expect(screen.getByText("Stopped here")).toBeDefined();
    expect(screen.queryByRole("button")).toBeNull();

    // Kept under that row from now on, as for a stop made on the row itself.
    stored.unmount();
    const reopened = renderCard({ ...data }, { sessionId, rowKey: "seq-9" });
    expect(screen.getByText("Stopped here")).toBeDefined();

    // A later pause in the same chat is its own, however alike.
    reopened.unmount();
    renderCard({ ...data }, { sessionId, rowKey: "seq-12" });
    expect(screen.getByRole("button", { name: "Stop here" })).toBeDefined();
  });

  it("forgets a live stop the chat moved past before the pause's row arrived", () => {
    backend();
    const sessionId = "chat-moved-past";
    const data = pause({ reason: "task", resets_at: null });
    const live = renderCard(data, { sessionId, live: true });

    fireEvent.click(screen.getByRole("button", { name: "Stop here" }));
    // A new message lands under it while it is still the live part.
    live.rerenderCard(data, { sessionId, live: true, current: false });
    live.unmount();

    renderCard({ ...data }, { sessionId, rowKey: "seq-12" });
    expect(screen.getByRole("button", { name: "Keep going" })).toBeDefined();
  });

  it("reads as one static line once the chat has moved past it", () => {
    backend();
    renderCard(pause(), { current: false });

    expect(
      screen.getByRole("note", {
        name: "Paused · out of today's budget. Picked back up",
      }),
    ).toBeDefined();
    expect(screen.queryByRole("button")).toBeNull();
  });

  it("shows just the title where the chat can't be acted on", () => {
    render(
      <CopilotChatActionsProvider onSend={vi.fn()} chatSurface="share">
        <TurnPausedCard data={pause()} pauseKey="seq-7" isStoredRow />
      </CopilotChatActionsProvider>,
    );

    expect(
      screen.getByRole("note", { name: "Paused · out of today's budget" }),
    ).toBeDefined();
    expect(screen.queryByRole("button")).toBeNull();
  });
});
