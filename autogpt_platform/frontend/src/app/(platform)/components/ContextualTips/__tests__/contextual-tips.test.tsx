import { act, render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import type { ToolUIPart } from "ai";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const capture = vi.hoisted(() => vi.fn());
vi.mock("posthog-js", () => ({ default: { capture } }));

const auth = vi.hoisted(() => ({ userID: "user-1" }));
vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ user: { id: auth.userID }, isLoggedIn: true }),
}));

const onboarding = vi.hoisted(() => ({
  current: {
    state: { completedSteps: [] as string[] } as {
      completedSteps: string[];
    } | null,
    completeStep: vi.fn(),
  },
}));
vi.mock("@/providers/onboarding/onboarding-provider", async (original) => ({
  ...(await original<object>()),
  useOnboarding: () => onboarding.current,
}));

const billing = vi.hoisted(() => ({ enabled: true, resolved: true }));
vi.mock("@/hooks/useIsBillingEnabled", () => ({
  useIsBillingEnabled: () => billing.enabled,
  useBillingStatus: () => ({
    isBillingEnabled: billing.enabled,
    isResolved: billing.resolved,
  }),
}));

vi.mock("@/services/usageExperience/useUsageExperience", () => ({
  useUsageExperience: () => ({ experience: { isActiveTrial: true } }),
}));

const sessionList = vi.hoisted(() => ({
  current: {
    sessions: [] as { id: string }[],
    isLoading: false,
    isListLoaded: true,
  } as {
    sessions: { id: string }[];
    isLoading: boolean;
    isListLoaded: boolean;
    dataUpdatedAt?: number;
  },
}));
vi.mock("@/app/(platform)/copilot/useSessionList", () => ({
  useSessionList: () => sessionList.current,
}));

import { ExpertChangeGroup } from "@/app/(platform)/copilot/components/ToolChain/ExpertCards";
import { BudgetLowHint } from "@/app/(platform)/copilot/components/UsageLimits/BudgetLowHint";
import type { UsageSnapshot } from "@/services/usageExperience/useUsageExperience";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { isKey } from "@/lib/keyboard";
import { useOrgTeamStore } from "@/services/org-team/store";
import type { User } from "@/lib/auth/types";
import { tourTarget } from "@/lib/tour-targets";
import { useFirstRunTourStore } from "../../FirstRunTour/store";
import { ContextualTips } from "../ContextualTips";
import { isFirstSecondChat } from "../helpers";
import { requestTip, useContextualTipStore } from "../store";
import { mentionsCost } from "../tips";
import { latestReplyMentionsCost, useCostReplyMoment } from "../useTipMoment";

// jsdom lays nothing out: tagged elements get a box unless listed here
// (the wallet on an install without billing, the phone's menu on desktop).
const hidden = new Set<string>();
// Targets laid out somewhere other than the default box.
const placed = new Map<string, DOMRectInit>();
function stubLayout() {
  vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(
    function (this: HTMLElement) {
      const target = this.getAttribute("data-tour-target");
      if (!target || hidden.has(target))
        return DOMRect.fromRect({ x: 0, y: 0, width: 0, height: 0 });
      return DOMRect.fromRect(
        placed.get(target) ?? { x: 10, y: 100, width: 240, height: 40 },
      );
    },
  );
}

function Page() {
  return (
    <div>
      <div {...tourTarget("chats")}>Recent chats</div>
      <div {...tourTarget("team")}>Team</div>
      <div {...tourTarget("wallet")}>Wallet</div>
      <div {...tourTarget("menu")}>Menu</div>
      <label>
        Message
        <textarea />
      </label>
      <ContextualTips />
    </div>
  );
}

beforeEach(() => {
  auth.userID = "user-1";
  hidden.clear();
  placed.clear();
  billing.enabled = true;
  billing.resolved = true;
  window.localStorage.clear();
  capture.mockReset();
  onboarding.current = { state: { completedSteps: [] }, completeStep: vi.fn() };
  sessionList.current = { sessions: [], isLoading: false, isListLoaded: true };
  act(() => useContextualTipStore.setState({ requested: [], shown: [] }));
  stubLayout();
});

afterEach(() => {
  act(() => {
    useFirstRunTourStore.getState().close();
    useFirstRunTourStore.getState().setConnectToolsOpen(false);
  });
  vi.restoreAllMocks();
});

describe("contextual tips — when a cost comes up", () => {
  it.each([
    "That costs a few cents per image.",
    "This run uses about 2 credits.",
    "Each image is 1 credit.",
    "It comes out of your trial allowance, not your card.",
    "Automation credits pay for scheduled runs.",
    "That leaves 40% of your allowance for the week.",
    "You have 120 credits left.",
  ])("counts Otto's billing words: %s", (text) => {
    expect(mentionsCost(text)).toBe(true);
  });

  it.each([
    "I can cut your shipping costs by batching orders.",
    "Your $500 marketing budget is enough for two campaigns.",
    "Update the credit card on file in Stripe.",
    "We take credit cards and bank transfers.",
    "Give credit to the photographer in the caption.",
    "EVs can qualify for up to $7,500 in federal tax credits.",
    "The stock pays a dividend of 30 cents a share.",
    "Check the baggage allowance before you book the flight.",
    "We bill 5 cents per kilowatt hour on the shop's meter.",
  ])("ignores money talk that is about the business: %s", (text) => {
    expect(mentionsCost(text)).toBe(false);
  });

  it("only reads Otto's latest finished reply", () => {
    const reply = (text: string) => ({
      role: "assistant",
      parts: [{ type: "text", text }],
    });
    const cost = reply("That costs a few cents per image.");

    expect(latestReplyMentionsCost([cost], false)).toBe(true);
    expect(latestReplyMentionsCost([cost], true)).toBe(false);
    expect(
      latestReplyMentionsCost([cost, { role: "user", parts: [] }], false),
    ).toBe(false);
    expect(latestReplyMentionsCost([cost, reply("Done.")], false)).toBe(false);
  });

  it("only counts a reply that finishes in front of the user, not chat history", () => {
    const history = [
      { role: "user", parts: [{ type: "text", text: "Photos please" }] },
      {
        role: "assistant",
        parts: [{ type: "text", text: "Each image costs a few cents." }],
      },
    ];
    function Chat({ isStreaming }: { isStreaming: boolean }) {
      useCostReplyMoment({ messages: history, isStreaming, enabled: true });
      return null;
    }

    const { rerender } = render(<Chat isStreaming={false} />);
    expect(useContextualTipStore.getState().requested).toEqual([]);

    rerender(<Chat isStreaming />);
    rerender(<Chat isStreaming={false} />);
    expect(useContextualTipStore.getState().requested).toEqual(["first-cost"]);
  });

  it("treats running low on the allowance as the moment too", () => {
    const usage = {
      budget: { state: "low", window: "trial", percentUsed: 85 },
      experience: { isLifetimeTrial: true, trialPercent: 85 },
    } as unknown as UsageSnapshot;

    render(<BudgetLowHint usage={usage} />);

    expect(useContextualTipStore.getState().requested).toEqual(["first-cost"]);
  });
});

describe("contextual tips — when an expert is proposed", () => {
  function proposal(kind: "raise" | "hire"): ToolUIPart {
    return {
      type: "tool-raise_expert",
      state: "output-available",
      toolCallId: "call-1",
      input: {},
      output: {
        type: "expert_change_proposed",
        applied: false,
        confirmation_id: "c-1",
        preview: { kind, name: "Fiona", role: "Bookkeeper" },
      },
    } as ToolUIPart;
  }

  it("asks for the tip when Otto proposes an expert in front of the user", () => {
    render(
      <ExpertChangeGroup parts={[proposal("hire")]} isCurrentlyStreaming />,
    );

    expect(useContextualTipStore.getState().requested).toEqual([
      "first-expert",
    ]);
  });

  it("stays quiet on a proposal read back from an old chat", () => {
    // A decided proposal keeps applied: false in the transcript, so an old
    // chat would otherwise look like a fresh proposal every time it opens.
    render(<ExpertChangeGroup parts={[proposal("hire")]} />);

    expect(useContextualTipStore.getState().requested).toEqual([]);
  });

  it("stays quiet on a read-only transcript", () => {
    render(<ExpertChangeGroup parts={[proposal("raise")]} readOnly />);

    expect(useContextualTipStore.getState().requested).toEqual([]);
  });
});

describe("contextual tips — the first second chat", () => {
  it("counts going from one chat to two, and nothing else", () => {
    expect(isFirstSecondChat(1, 2)).toBe(true);
    expect(isFirstSecondChat(0, 2)).toBe(true);
    expect(isFirstSecondChat(0, 3)).toBe(true);
    expect(isFirstSecondChat(2, 3)).toBe(false);
    expect(isFirstSecondChat(null, 2)).toBe(false);
    expect(isFirstSecondChat(0, 1)).toBe(false);
    expect(isFirstSecondChat(5, 6)).toBe(false);
  });

  it("does not count the list dropping to one and coming back to two", async () => {
    // A delete that fails puts the chat back; deleting down to one and
    // starting another is not someone's first second chat either.
    for (const counts of [
      [2, 1, 2],
      [3, 1, 2],
    ]) {
      act(() => useContextualTipStore.setState({ requested: [], shown: [] }));
      const sessionsOf = (n: number) =>
        Array.from({ length: n }, (_, i) => ({ id: `s${i}` }));
      sessionList.current = {
        sessions: sessionsOf(counts[0]),
        isLoading: false,
        isListLoaded: true,
      };
      const { rerender, unmount } = render(<Page />);
      for (const count of counts.slice(1)) {
        sessionList.current = {
          sessions: sessionsOf(count),
          isLoading: false,
          isListLoaded: true,
        };
        rerender(<Page />);
      }
      await act(async () => {});

      expect(useContextualTipStore.getState().requested).toEqual([]);
      unmount();
    }
  });

  it("starts counting afresh for another account", async () => {
    sessionList.current = {
      sessions: [{ id: "a" }],
      isLoading: false,
      isListLoaded: true,
    };
    const { rerender } = render(<Page />);

    // Someone else signs in here and already has two chats.
    auth.userID = "user-2";
    sessionList.current = {
      sessions: [{ id: "x" }, { id: "y" }],
      isLoading: false,
      isListLoaded: true,
    };
    rerender(<Page />);
    await act(async () => {});

    expect(useContextualTipStore.getState().requested).toEqual([]);
  });

  it("counts going straight from no chats to two, but only from a list that loaded", async () => {
    // A new account's first two chats can land in one refresh. A list that
    // failed to load reads as empty too, and must not make someone with
    // several chats look new.
    sessionList.current = {
      sessions: [],
      isLoading: false,
      isListLoaded: false,
    };
    const { rerender } = render(<Page />);
    sessionList.current = {
      sessions: [{ id: "a" }, { id: "b" }, { id: "c" }],
      isLoading: false,
      isListLoaded: true,
    };
    rerender(<Page />);
    await act(async () => {});
    expect(useContextualTipStore.getState().requested).toEqual([]);

    // Same component, an account with no chats yet that then gets two.
    auth.userID = "user-2";
    sessionList.current = {
      sessions: [],
      isLoading: false,
      isListLoaded: true,
    };
    rerender(<Page />);
    sessionList.current = {
      sessions: [{ id: "x" }, { id: "y" }],
      isLoading: false,
      isListLoaded: true,
    };
    rerender(<Page />);
    await waitFor(() =>
      expect(useContextualTipStore.getState().requested).toContain(
        "second-chat",
      ),
    );
  });

  it("keeps counting when the page's organization first loads", async () => {
    // On load the active org is unknown until the org list arrives; the
    // chats shown meanwhile are the same list, so its second chat counts.
    act(() => useOrgTeamStore.setState({ activeOrgID: null }));
    sessionList.current = {
      sessions: [{ id: "a" }],
      isLoading: false,
      isListLoaded: true,
      dataUpdatedAt: 1,
    };
    const { rerender } = render(<Page />);

    act(() => useOrgTeamStore.setState({ activeOrgID: "org-personal" }));
    await act(async () => {});
    sessionList.current = {
      sessions: [{ id: "a" }, { id: "b" }],
      isLoading: false,
      isListLoaded: true,
      dataUpdatedAt: Date.now() + 60_000,
    };
    rerender(<Page />);

    await waitFor(() =>
      expect(useContextualTipStore.getState().requested).toContain(
        "second-chat",
      ),
    );
    act(() => useOrgTeamStore.setState({ activeOrgID: null }));
  });

  it("starts counting afresh in another organization", async () => {
    // The chat list is per organization: switching from a personal org with
    // one chat to a team's with three is not starting a second chat.
    act(() => useOrgTeamStore.setState({ activeOrgID: "org-personal" }));
    sessionList.current = {
      sessions: [{ id: "a" }],
      isLoading: false,
      isListLoaded: true,
      dataUpdatedAt: 1,
    };
    const { rerender } = render(<Page />);

    // The switch renders once with the last org's list still on screen,
    // before the reset refetches it for the new one.
    act(() => useOrgTeamStore.setState({ activeOrgID: "org-team" }));
    await act(async () => {});
    sessionList.current = {
      sessions: [{ id: "x" }, { id: "y" }, { id: "z" }],
      isLoading: false,
      isListLoaded: true,
      dataUpdatedAt: Date.now() + 60_000,
    };
    rerender(<Page />);
    await act(async () => {});

    expect(useContextualTipStore.getState().requested).toEqual([]);
    act(() => useOrgTeamStore.setState({ activeOrgID: null }));
  });

  it("points at the chat list when a second chat appears", async () => {
    sessionList.current = {
      sessions: [{ id: "a" }],
      isLoading: false,
      isListLoaded: true,
    };
    const { rerender } = render(<Page />);
    expect(screen.queryByTestId("contextual-tip")).toBeNull();

    sessionList.current = {
      sessions: [{ id: "b" }, { id: "a" }],
      isLoading: false,
      isListLoaded: true,
    };
    rerender(<Page />);

    expect(await screen.findByText("You have two chats now")).toBeDefined();
  });

  it("says nothing to someone who already has several chats", async () => {
    sessionList.current = {
      sessions: [{ id: "a" }, { id: "b" }, { id: "c" }],
      isLoading: false,
      isListLoaded: true,
    };
    const { rerender } = render(<Page />);
    sessionList.current = {
      sessions: [{ id: "d" }, { id: "a" }, { id: "b" }, { id: "c" }],
      isLoading: false,
      isListLoaded: true,
    };
    rerender(<Page />);

    await act(async () => {});
    expect(screen.queryByTestId("contextual-tip")).toBeNull();
  });
});

describe("contextual tips — showing one", () => {
  it("shows beside its target without taking focus, and is gone for good after Got it", async () => {
    render(<Page />);
    const textarea = screen.getByRole("textbox", { name: "Message" });
    textarea.focus();

    act(() => requestTip("first-expert"));

    expect(await screen.findByText("Experts can own tasks")).toBeDefined();
    expect(document.activeElement).toBe(textarea);

    await userEvent.click(screen.getByRole("button", { name: "Got it" }));

    await waitFor(() =>
      expect(screen.queryByTestId("contextual-tip")).toBeNull(),
    );
    expect(onboarding.current.completeStep).toHaveBeenCalledWith("EXPERT_TIP");
    expect(
      window.localStorage.getItem("autogpt:contextual-tip-seen:first-expert"),
    ).toBe("user-1");

    act(() => requestTip("first-expert"));
    await act(async () => {});
    expect(screen.queryByTestId("contextual-tip")).toBeNull();
  });

  it("says where costs come from on a trial", async () => {
    render(<Page />);
    act(() => requestTip("first-cost"));

    expect(
      await screen.findByText(
        "Chats and experts use your trial allowance. Automations use credits. You can check both here.",
      ),
    ).toBeDefined();
  });

  it("never shows a tip this user already has on another device", async () => {
    onboarding.current.state = { completedSteps: ["CHATS_TIP"] };
    render(<Page />);

    act(() => requestTip("second-chat"));

    await act(async () => {});
    expect(screen.queryByTestId("contextual-tip")).toBeNull();
    expect(useContextualTipStore.getState().requested).toEqual([]);
  });

  it("waits while the walkthrough is open, then shows", async () => {
    render(<Page />);
    act(() => useFirstRunTourStore.getState().open("menu"));
    act(() => requestTip("first-cost"));

    await act(async () => {});
    expect(screen.queryByTestId("contextual-tip")).toBeNull();

    act(() => useFirstRunTourStore.getState().close());
    expect(await screen.findByText("Where costs come from")).toBeDefined();
  });

  it("gets out of the way of the connect-your-tools dialog, even once it is showing", async () => {
    render(<Page />);
    act(() => requestTip("first-cost"));
    expect(await screen.findByText("Where costs come from")).toBeDefined();

    act(() => useFirstRunTourStore.getState().setConnectToolsOpen(true));
    await waitFor(() =>
      expect(screen.queryByTestId("contextual-tip")).toBeNull(),
    );

    act(() => useFirstRunTourStore.getState().setConnectToolsOpen(false));
    expect(await screen.findByText("Where costs come from")).toBeDefined();
  });

  it("counts a tip as shown once, however often it is put back on screen", async () => {
    // Anything opening over the tip (another dialog, the walkthrough) takes
    // it down and puts it back once it closes.
    render(<Page />);
    act(() => requestTip("first-cost"));
    expect(await screen.findByText("Where costs come from")).toBeDefined();

    act(() => useFirstRunTourStore.getState().open("menu"));
    await waitFor(() =>
      expect(screen.queryByTestId("contextual-tip")).toBeNull(),
    );
    act(() => useFirstRunTourStore.getState().close());
    expect(await screen.findByText("Where costs come from")).toBeDefined();

    const shown = capture.mock.calls.filter(
      ([event]) => event === "contextual_tip_shown",
    );
    expect(shown).toEqual([["contextual_tip_shown", { tip: "first-cost" }]]);
  });

  it("steps aside while a popover is open and comes back once it closes, for later tips too", async () => {
    // Opening the wallet or the profile menu (both role="dialog") used to
    // shelve tips until the next page: closing it brought nothing back, and
    // switching chats (?sessionId=) never changes the page.
    render(<Page />);
    act(() => requestTip("second-chat"));
    expect(await screen.findByText("You have two chats now")).toBeDefined();

    const popover = document.createElement("div");
    popover.setAttribute("role", "dialog");
    act(() => {
      document.body.appendChild(popover);
    });
    await waitFor(() =>
      expect(screen.queryByTestId("contextual-tip")).toBeNull(),
    );

    act(() => popover.remove());
    expect(await screen.findByText("You have two chats now")).toBeDefined();

    await userEvent.click(screen.getByRole("button", { name: "Got it" }));
    act(() => requestTip("first-expert"));
    expect(await screen.findByText("Experts can own tasks")).toBeDefined();
  });

  it("is recorded the moment it shows, so leaving without Got it does not bring it back", async () => {
    const { unmount } = render(<Page />);
    act(() => requestTip("second-chat"));
    await screen.findByText("You have two chats now");

    expect(onboarding.current.completeStep).toHaveBeenCalledWith("CHATS_TIP");
    expect(
      window.localStorage.getItem("autogpt:contextual-tip-seen:second-chat"),
    ).toBe("user-1");

    // A reload: a fresh page, the same person, the same moment again.
    unmount();
    act(() => useContextualTipStore.setState({ requested: [], shown: [] }));
    render(<Page />);
    act(() => requestTip("second-chat"));
    await act(async () => {});

    expect(screen.queryByTestId("contextual-tip")).toBeNull();
  });

  it("keeps its queue through a dropped session, and drops it for another account", () => {
    act(() => useAuthStore.setState({ user: { id: "user-1" } as User }));
    act(() => requestTip("first-expert"));

    act(() => useAuthStore.setState({ user: null }));
    act(() => useAuthStore.setState({ user: { id: "user-1" } as User }));
    expect(useContextualTipStore.getState().requested).toEqual([
      "first-expert",
    ]);

    act(() => useAuthStore.setState({ user: { id: "user-2" } as User }));
    expect(useContextualTipStore.getState().requested).toEqual([]);
    act(() => useAuthStore.setState({ user: null }));
  });

  it("shows one tip at a time, oldest first", async () => {
    render(<Page />);
    act(() => {
      requestTip("first-expert");
      requestTip("first-cost");
    });

    expect(await screen.findByText("Experts can own tasks")).toBeDefined();
    expect(screen.queryByText("Where costs come from")).toBeNull();

    await userEvent.click(screen.getByRole("button", { name: "Got it" }));

    expect(await screen.findByText("Where costs come from")).toBeDefined();
  });

  it("never queues the cost tip where billing is off, so it cannot hold up the others", async () => {
    billing.enabled = false;
    hidden.add("wallet");
    hidden.add("menu");
    render(<Page />);

    act(() => {
      requestTip("first-cost");
      requestTip("first-expert");
    });

    expect(await screen.findByText("Experts can own tasks")).toBeDefined();
    await waitFor(() =>
      expect(useContextualTipStore.getState().requested).toEqual([
        "first-expert",
      ]),
    );
  });

  it("keeps the cost tip waiting while the billing flag is still on its way", async () => {
    // A self-hosted install reads its payments flag from a vendor: until it
    // answers, billing reads as off, which is not the same as being off.
    billing.enabled = false;
    billing.resolved = false;
    const { rerender } = render(<Page />);
    act(() => requestTip("first-cost"));
    await act(async () => {});

    expect(useContextualTipStore.getState().requested).toEqual(["first-cost"]);
    expect(screen.queryByTestId("contextual-tip")).toBeNull();

    billing.enabled = true;
    billing.resolved = true;
    rerender(<Page />);
    expect(await screen.findByText("Where costs come from")).toBeDefined();
  });

  it("steps aside when its target is not on screen, letting the next tip show", async () => {
    hidden.add("wallet");
    hidden.add("menu");
    render(<Page />);

    act(() => {
      requestTip("first-cost");
      requestTip("first-expert");
    });

    expect(
      await screen.findByText("Experts can own tasks", {}, { timeout: 4000 }),
    ).toBeDefined();
    expect(screen.queryByText("Where costs come from")).toBeNull();
    // Still waiting its turn, not dropped: it shows once its target exists.
    expect(useContextualTipStore.getState().requested).toContain("first-cost");
    // And it was never counted as shown while it had nothing to point at.
    expect(capture).toHaveBeenCalledWith("contextual_tip_shown", {
      tip: "first-expert",
    });
    expect(capture).not.toHaveBeenCalledWith("contextual_tip_shown", {
      tip: "first-cost",
    });
  });

  it("does not point at a target scrolled out of view", async () => {
    // A tip shows while the user is busy and never scrolls the page, so a
    // target below the fold waits like a missing one.
    placed.set("wallet", {
      x: 10,
      y: window.innerHeight + 300,
      width: 240,
      height: 40,
    });
    hidden.add("menu");
    render(<Page />);

    act(() => {
      requestTip("first-cost");
      requestTip("first-expert");
    });

    expect(
      await screen.findByText("Experts can own tasks", {}, { timeout: 4000 }),
    ).toBeDefined();
    expect(screen.queryByText("Where costs come from")).toBeNull();
  });

  it("tells a screen reader to open the menu when the tip points there", async () => {
    hidden.add("chats");
    render(<Page />);
    act(() => requestTip("second-chat"));
    await screen.findByText("You have two chats now");

    expect(screen.getByRole("status").textContent).toMatch(
      /Open the menu to find it\.$/,
    );
  });

  it("puts the tip away with Escape pressed inside it", async () => {
    render(<Page />);
    act(() => requestTip("second-chat"));
    await screen.findByText("You have two chats now");

    screen.getByRole("button", { name: "Got it" }).focus();
    await userEvent.keyboard("{Escape}");

    await waitFor(() =>
      expect(screen.queryByTestId("contextual-tip")).toBeNull(),
    );
    expect(onboarding.current.completeStep).toHaveBeenCalledWith("CHATS_TIP");
  });

  it("lets an Escape outside it through untouched, for the card or menu it belongs to", async () => {
    // Radix's dismiss layer marks every Escape on the page as handled, so
    // the expert card's "Skip questions · Esc" (which skips a handled one)
    // and any open menu never heard it while a tip was up.
    render(<Page />);
    act(() => requestTip("second-chat"));
    await screen.findByText("You have two chats now");
    let outsideEscapeWasHandled: boolean | null = null;
    function onKeyDown(event: KeyboardEvent) {
      if (isKey(event, "Escape"))
        outsideEscapeWasHandled = event.defaultPrevented;
    }
    window.addEventListener("keydown", onKeyDown);
    try {
      screen.getByRole("textbox", { name: "Message" }).focus();
      await userEvent.keyboard("{Escape}");
    } finally {
      window.removeEventListener("keydown", onKeyDown);
    }

    expect(outsideEscapeWasHandled).toBe(false);
    expect(screen.getByTestId("contextual-tip")).toBeDefined();
  });

  it("stays put when Escape belongs to something else on the page", async () => {
    // The tip never takes focus, so an Escape elsewhere (closing the
    // composer's mention list) is not the user putting the tip away.
    render(<Page />);
    act(() => requestTip("second-chat"));
    await screen.findByText("You have two chats now");

    screen.getByRole("textbox", { name: "Message" }).focus();
    await userEvent.keyboard("{Escape}");
    await act(async () => {});

    expect(screen.getByTestId("contextual-tip")).toBeDefined();
    expect(useContextualTipStore.getState().requested).toContain("second-chat");
  });

  it("gives way to a tip that arrives later, after it has waited alone", async () => {
    hidden.add("wallet");
    hidden.add("menu");
    render(<Page />);
    act(() => requestTip("first-cost"));
    // Alone in the queue past its grace period: there was no one to give
    // way to then.
    // Outside act(): inside it React holds the tip's mount until the wait
    // is over, and the timer never runs alone.
    await new Promise((resolve) => setTimeout(resolve, 2500));

    act(() => requestTip("first-expert"));

    expect(
      await screen.findByText("Experts can own tasks", {}, { timeout: 4000 }),
    ).toBeDefined();
  }, 15000);
});
